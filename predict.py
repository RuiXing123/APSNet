import os
import sys
from collections import defaultdict

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ==================================================
# Global configuration
# ==================================================

# ------------------
# Paths
# ------------------
CKPT_PATH = 'output/Challenge'
DATA_ROOT = 'data/APS_dataset'
VAL_DIR = os.path.join(DATA_ROOT, 'test')

# ------------------
# Hardware
# ------------------
DEVICE_ID = 1
DEVICE = torch.device(f'cuda:{DEVICE_ID}' if torch.cuda.is_available() else 'cpu')

# ------------------
# Image
# ------------------
IMAGE_SIZE = 512

# ------------------
# Normalization
# ------------------
MEAN = (0.6618, 0.6510, 0.6353)
STD  = (0.1053, 0.1120, 0.1130)

# ------------------
# Grad-CAM
# ------------------
SAVE_CAM = False
CAM_SAVE_DIR = CKPT_PATH + '/gradcam_results'


# ==================================================
# Transform
# ==================================================
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ==================================================
# Dataset
# ==================================================
TESTSET = torchvision.datasets.ImageFolder(
    root=VAL_DIR,
    transform=TRANSFORM
)

CLASS_NAMES = TESTSET.classes
NUM_CLASSES = len(CLASS_NAMES)

print(f'[INFO] Discovered {NUM_CLASSES} classes: {CLASS_NAMES}')


# ==================================================
# Load model
# ==================================================
net = torch.load(CKPT_PATH + "/model_acc_com.pth", map_location=DEVICE, weights_only=False)
net.eval()
net.to(DEVICE)


# ==================================================
# Utility functions
# ==================================================
@torch.no_grad()
def predict_one(image_path):
    img = Image.open(image_path).convert('RGB')

    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    img = img.crop((left, top, left + side, top + side))

    x = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    out1, out2, out3, out_cat, _ = net(x)
    out_ensemble = out1 + out2 + out3 + out_cat

    prob = torch.softmax(out_ensemble, dim=1)
    top5_prob, top5_idx = torch.topk(prob, k=min(5, NUM_CLASSES), dim=1)

    return [
        (CLASS_NAMES[i], float(top5_prob[0, j]))
        for j, i in enumerate(top5_idx[0])
    ]


def get_last_conv_layer(model: torch.nn.Module):
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d layer found. Grad-CAM requires a convolutional layer.")
    return last_conv


class WrapperModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out1, out2, out3, out_cat, _ = self.model(x)
        return out1 + out2 + out3 + out_cat


@torch.no_grad()
def predict_one_gradcam(image_path):
    img_pil = Image.open(image_path).convert('RGB')

    width, height = img_pil.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    img_pil = img_pil.crop((left, top, left + side, top + side))

    x = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)

    out1, out2, out3, out_cat, _ = net(x)
    out_ensemble = out1 + out2 + out3 + out_cat

    prob = torch.softmax(out_ensemble, dim=1)
    top5_prob, top5_idx = torch.topk(prob, k=min(5, NUM_CLASSES), dim=1)

    preds = [
        (CLASS_NAMES[i], float(top5_prob[0, j]))
        for j, i in enumerate(top5_idx[0])
    ]

    if SAVE_CAM:
        wrapped_net = WrapperModel(net)
        target_layer = get_last_conv_layer(net)

        cam = GradCAM(
            model=wrapped_net,
            target_layers=[target_layer]
        )

        grayscale_cam = cam(input_tensor=x)[0]

        rgb_img = np.array(
            img_pil.resize((IMAGE_SIZE, IMAGE_SIZE))
        ).astype(np.float32) / 255.0

        visualization = show_cam_on_image(
            rgb_img, grayscale_cam, use_rgb=True
        )

        os.makedirs(CAM_SAVE_DIR, exist_ok=True)
        save_path = os.path.join(CAM_SAVE_DIR, os.path.basename(image_path))
        Image.fromarray(visualization).save(save_path)

        print(f"[Grad-CAM] Saved heatmap to {save_path}")

    return preds


# ==================================================
# Main evaluation
# ==================================================
if __name__ == '__main__':

    if not os.path.isdir(VAL_DIR):
        print('Validation directory not found.')
        sys.exit(1)

    total = 0
    correct = 0

    class_stats = defaultdict(lambda: [0, 0])
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    for class_name in os.listdir(VAL_DIR):
        class_dir = os.path.join(VAL_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        gt_idx = CLASS_NAMES.index(class_name)

        for file_name in os.listdir(class_dir):
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_path = os.path.join(class_dir, file_name)

            preds = predict_one_gradcam(image_path)
            pred_class = preds[0][0]
            pred_idx = CLASS_NAMES.index(pred_class)

            conf_mat[gt_idx, pred_idx] += 1

            is_correct = (pred_class == class_name)
            correct += int(is_correct)
            total += 1

            class_stats[class_name][1] += 1
            if is_correct:
                class_stats[class_name][0] += 1

            print(
                f"{file_name:<20} "
                f"Predicted: {pred_class:<20} "
                f"GT: {class_name:<20} "
                f"{'OK' if is_correct else 'WRONG'}"
            )

    # ------------------
    # Class-wise accuracy
    # ------------------
    print("\n[CLASS-WISE ACCURACY]")
    for cls, (right, total_cls) in sorted(class_stats.items()):
        acc = right / total_cls if total_cls > 0 else 0.0
        print(f"{cls:<20} {right:>3}/{total_cls:<3}  Acc: {acc:.4f}")

    # ------------------
    # Class-wise precision
    # ------------------
    print("\n[CLASS-WISE PRECISION]")
    for i, cls_name in enumerate(CLASS_NAMES):
        TP = conf_mat[i, i]
        FP = np.sum(conf_mat[:, i]) - TP
        precision = TP / (TP + FP + 1e-8)
        print(f"{cls_name:<20} Precision: {precision:.4f}")

    # ------------------
    # Overall metrics
    # ------------------
    TP = np.diag(conf_mat)
    FP = np.sum(conf_mat, axis=0) - TP
    FN = np.sum(conf_mat, axis=1) - TP

    precision = np.mean(TP / (TP + FP + 1e-8))
    recall = np.mean(TP / (TP + FN + 1e-8))
    f1_score = np.mean(2 * precision * recall / (precision + recall + 1e-8))
    accuracy = np.sum(TP) / np.sum(conf_mat)

    print("\n[OVERALL METRICS]")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1_score:.4f}")


