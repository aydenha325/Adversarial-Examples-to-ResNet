import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt

# resnet, fgsm 임포트
from resnet import *
from fgsm import *


def plot(examples, total_loss, total_accuracy, total_class_accuracy):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


if __name__ == '__main__':
    # CUDA 사용 가능 여부 확인
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    # 모델 불러오기
    model = ResNet().to(DEVICE)
    model.load_state_dict(torch.load('./model_state_dict.pth'))
    model.eval()

    examples = []
    total_loss = []
    total_accuracy = []
    total_class_accuracy = []
    epsilons = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    for epsilon in epsilons:
        with open(f'./examples/epslion_{epsilon}.pth', 'rb') as f:
            test_loader = torch.load(f)
        examples.append(test_loader[0])

        test_result = evaluate(model, test_loader, DEVICE, fgsm=True)

        total_loss.append(test_result[0])
        total_accuracy.append(test_result[1])
        total_class_accuracy.append(test_result[2])

    plot(examples, total_loss, total_accuracy, total_class_accuracy)
    print('Done!')
