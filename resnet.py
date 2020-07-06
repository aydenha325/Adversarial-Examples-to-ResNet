import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models


# ResNet 정의
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# CIFAR-10 준비
def cifar_10(BATCH_SIZE):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./',
                         train=True,
                         download=True,
                         transform=transforms.Compose([
                             transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./',
                         train=False,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=BATCH_SIZE, shuffle=True)
    return train_loader, test_loader


# 모델 학습
def train(model, train_loader, optimizer, epoch, DEVICE):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


# 모델 테스트
def evaluate(model, test_loader, DEVICE, fgsm=False):
    model.eval()
    test_loss = 0
    correct = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            if fgsm:
                pred = output.max(1)[1]
                correct_class = (pred == target)
                for i in range(16):
                    label = target[i]
                    class_correct[label] += int(correct_class[i])
                    class_total[label] += 1

    try:
        test_loss /= len(test_loader.dataset)
        test_accuracy = 100. * correct / len(test_loader.dataset)
    except:
        test_loss /= 10000
        test_accuracy = 100. * correct / 10000

    if fgsm:
        class_accuracy = []
        for i in range(10):
            class_accuracy.append(100 * class_correct[i] / class_total[i])
        return test_loss, test_accuracy, class_accuracy
    else:
        return test_loss, test_accuracy


# 학습/테스트, 저장
def save_model(EPOCHS, model, optimizer, scheduler, train_loader, test_loader, DEVICE):
    for epoch in range(1, EPOCHS+1):
        scheduler.step()
        train(model, train_loader, optimizer, epoch, DEVICE)
        test_loss, test_accuracy = evaluate(model, test_loader, DEVICE)

        print(f'[{epoch}] Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')

        # 체크포인트 저장
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }, f'./save_p_eph/epoch{epoch}.pth')

    # 전체 모델 저장
    torch.save(model, './model.pth')
    torch.save(model.state_dict(), './model_state_dict.pth')


# 실행
if __name__ == '__main__':
    # CUDA 사용 가능 여부 확인
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    # 하이퍼파라미터
    EPOCHS = 300
    BATCH_SIZE = 128

    # CIFAR-10 로드
    train_loader, test_loader = cifar_10(BATCH_SIZE)

    # 학습/테스트 준비
    model = ResNet().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # 학습/테스트 실행
    save_model(EPOCHS, model, optimizer, scheduler, train_loader, test_loader, DEVICE)
    print('Done!')

