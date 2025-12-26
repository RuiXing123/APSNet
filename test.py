import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm


def test(
    net,
    criterion,
    batch_size,
    device,
    data_path,
    image_size,
    mean,
    std,
    num_workers
):
    net.eval()

    test_loss = 0.0
    correct = 0
    correct_com = 0
    total = 0

    transform_test = transforms.Compose([
        transforms.Resize((image_size + 102, image_size + 102)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    testset = torchvision.datasets.ImageFolder(
        root=f"{data_path}/testraw",
        transform=transform_test
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    progress_bar = tqdm(testloader,desc="Validation",ncols=120,leave=True)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(device)
            targets = targets.to(device)

            output_1, output_2, output_3, _, output_concat = net(inputs)
            outputs_com = output_1 + output_2 + output_3 + output_concat

            loss = criterion(output_concat, targets)
            test_loss += loss.item()

            _, predicted = torch.max(output_concat, 1)
            _, predicted_com = torch.max(outputs_com, 1)

            batch_size_cur = targets.size(0)
            total += batch_size_cur
            correct += predicted.eq(targets).sum().item()
            correct_com += predicted_com.eq(targets).sum().item()

            progress_bar.set_postfix({
                "loss": f"{test_loss / (batch_idx + 1):.4f}",
                "acc": f"{100.0 * correct / total:.2f}%",
                "acc_com": f"{100.0 * correct_com / total:.2f}%"
            })

    test_acc = 100.0 * correct / total
    test_acc_com = 100.0 * correct_com / total
    test_loss /= len(testloader)

    return test_acc, test_acc_com, test_loss
