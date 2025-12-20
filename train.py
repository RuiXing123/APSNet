import os
import numpy as np
import torch
import torch.optim as optim
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm
from ASPNet import load_model
from ContrastiveLoss import con_loss
from test import test
from Jigsaw import jigsaw_generator

# ==================================================
# Global configuration
# ==================================================

# ------------------
# Hardware
# ------------------
DEVICE_ID = 1
DEVICE = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")

# ------------------
# Data
# ------------------
DATA_PATH = 'data/APS_dataset/'
BATCH_SIZE = 16
NUM_WORKERS = 8

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
# Training
# ------------------
NB_EPOCH = 200
PATIENCE = 30

# ------------------
# Model
# ------------------
MODEL_NAME = 'APSNet'
PRETRAIN = False
REQUIRE_GRAD = True

# ------------------
# Experiment
# ------------------
STORE_NAME = 'output/Challenge'
RESUME = False
MODEL_PATH = ''
START_EPOCH = 0

# ------------------
# Optimizer learning rates
# ------------------
LR_GROUPS = [
    0.001,   # classifier_concat
    0.001,   # conv_block1
    0.001,   # classifier1
    0.001,   # conv_block2
    0.001,   # classifier2
    0.001,   # conv_block3
    0.001,   # classifier3
    0.0001   # features
]
WEIGHT_DECAY = 5e-4

# ==================================================
# Learning rate scheduler
# ==================================================
def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % nb_epoch) / nb_epoch
    cos_out = np.cos(cos_inner) + 1
    return float(lr / 2 * cos_out)

# ==================================================
# Training function
# ==================================================
def train(
    nb_epoch,
    batch_size,
    store_name,
    data_path,
    device,
    num_workers,
    patience,
    model_name,
    pretrain,
    require_grad,
    weight_decay,
    lr_groups,
    resume,
    start_epoch,
    model_path
):

    os.makedirs(store_name, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    print('Preparing training data...')

    transform_train = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 100, IMAGE_SIZE + 100)),
        transforms.RandomCrop(IMAGE_SIZE, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    trainset = torchvision.datasets.ImageFolder(
        root=data_path + '/train',
        transform=transform_train
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    if resume:
        net = torch.load(model_path, weights_only=False)
    else:
        net = load_model(
            model_name=model_name,
            pretrain=pretrain,
            require_grad=require_grad
        )

    net.to(device)
    netp = torch.nn.DataParallel(net, device_ids=[device.index])

    # --------------------------------------------------
    # Loss & optimizer
    # --------------------------------------------------
    CELoss = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        [
            {'params': net.classifier_concat.parameters(),         'lr': lr_groups[0]},
            {'params': net.conv_block1.parameters(),               'lr': lr_groups[1]},
            {'params': net.classifier1.parameters(),               'lr': lr_groups[2]},
            {'params': net.conv_block2.parameters(),               'lr': lr_groups[3]},
            {'params': net.classifier2.parameters(),               'lr': lr_groups[4]},
            {'params': net.conv_block3.parameters(),               'lr': lr_groups[5]},
            {'params': net.classifier_decoupled.parameters(),      'lr': lr_groups[6]},
            {'params': net.features.parameters(),                  'lr': lr_groups[7]},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay
    )

    max_val_acc = 0
    max_val_acc_com = 0
    patience_counter = 0

    # ==================================================
    # Training loop
    # ==================================================
    for epoch in range(start_epoch, nb_epoch):

        net.train()
        print(f"\nEpoch [{epoch + 1}/{nb_epoch}]")

        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0

        correct = 0
        total = 0

        pbar = tqdm(enumerate(trainloader),total=len(trainloader),ncols=120,desc='Train')

        for batch_idx, (inputs, targets) in pbar:

            if inputs.size(0) < batch_size:
                continue

            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)

            inputs, targets = Variable(inputs), Variable(targets)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = \
                    cosine_anneal_schedule(epoch, nb_epoch, lr_groups[i])

            optimizer.zero_grad()
            inputs1 = jigsaw_generator(inputs, 8)
            output_1, _, _, _, _ = netp(inputs1)
            loss1 = CELoss(output_1, targets)
            loss1.backward()
            optimizer.step()

            optimizer.zero_grad()
            inputs2 = jigsaw_generator(inputs, 4)
            _, output_2, _, _, _ = netp(inputs2)
            loss2 = CELoss(output_2, targets)
            loss2.backward()
            optimizer.step()

            optimizer.zero_grad()
            inputs3 = jigsaw_generator(inputs, 2)
            _, _, output_3, _, ConRes = netp(inputs3)
            loss3 = CELoss(output_3, targets)
            ConResloss = con_loss(ConRes, targets)
            decoupled_loss = loss3 + ConResloss
            decoupled_loss.backward()
            optimizer.step()

            optimizer.zero_grad()
            _, _, _, output_concat, _ = netp(inputs)
            concat_loss = CELoss(output_concat, targets) * 2
            concat_loss.backward()
            optimizer.step()

            _, predicted = torch.max(output_concat.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            train_loss += (
                loss1.item() +
                loss2.item() +
                decoupled_loss.item() +
                concat_loss.item()
            )

            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += decoupled_loss.item()
            train_loss4 += concat_loss.item()

            acc = 100. * float(correct) / total

            pbar.set_postfix({
                'L1': f'{train_loss1/(batch_idx+1):.3f}',
                'L2': f'{train_loss2/(batch_idx+1):.3f}',
                'Ldc': f'{train_loss3/(batch_idx+1):.3f}',
                'Lcon': f'{train_loss4/(batch_idx+1):.3f}',
                'Acc': f'{acc:.2f}%'
            })

        train_acc = 100. * float(correct) / total
        train_loss /= (batch_idx + 1)

        with open(store_name + '/results_train.txt', 'a') as f:
            f.write(
                f"Epoch {epoch} | "
                f"Train Acc: {train_acc:.5f} | "
                f"Train Loss: {train_loss:.5f}\n"
            )

        val_acc, val_acc_com, val_loss = test(
            net=net,
            criterion=CELoss,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            data_path=data_path,
            image_size=IMAGE_SIZE,
            mean=MEAN,
            std=STD,
            num_workers=NUM_WORKERS
        )

        improved = False
        if val_acc_com >= max_val_acc_com:
            max_val_acc_com = val_acc_com
            torch.save(net, store_name + '/model_acc_com.pth')
            improved = True

        if val_acc >= max_val_acc:
            max_val_acc = val_acc
            torch.save(net, store_name + '/model_acc.pth')
            improved = True

        if improved:
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{patience}")

        with open(store_name + '/results_test.txt', 'a') as f:
            f.write(
                f"Epoch {epoch}, "
                f"Val Acc: {val_acc:.5f}, "
                f"Val Acc Combined: {val_acc_com:.5f}, "
                f"Val Loss: {val_loss:.6f}\n"
            )

        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs.")
            break


# ==================================================
# Main
# ==================================================
if __name__ == '__main__':
    train(
        nb_epoch=NB_EPOCH,
        batch_size=BATCH_SIZE,
        store_name=STORE_NAME,
        data_path=DATA_PATH,
        device=DEVICE,
        num_workers=NUM_WORKERS,
        patience=PATIENCE,
        model_name=MODEL_NAME,
        pretrain=PRETRAIN,
        require_grad=REQUIRE_GRAD,
        weight_decay=WEIGHT_DECAY,
        lr_groups=LR_GROUPS,
        resume=RESUME,
        start_epoch=START_EPOCH,
        model_path=MODEL_PATH
    )
