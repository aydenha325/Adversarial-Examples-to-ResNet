import torch
import torch.nn.functional as F
from torchvision import transforms, datasets, models, utils
import numpy as np
import matplotlib.pyplot as plt

# resnet 임포트
from resnet import *


# FGSM 정의
def fgsm_attack(data, epsilon, gradient):
    if epsilon:
        # 기울기값 원소의 sign값
        sign_gradient = gradient.sign()
        # 각 픽셀값을 sign_gradient 방향으로 epsilon 만큼 조절
        perturbed_image = data + epsilon * sign_gradient
        # [0,1] 범위를 벗어나는 값 조절
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
    else:
        perturbed_image = data
    return perturbed_image


# 적대적 예제 생성
def make_example(model, test_loader, epsilon, DEVICE):
    adversarial_examples = []

    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        data.requires_grad_(True)
        output = model(data)

        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()

        gradient = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, gradient)
        adversarial_examples.append([perturbed_data, target])

    return adversarial_examples


# 앱실론값별 적대적 예제 세트 저장
def loop_epsilon(model, test_loader, epsilons, DEVICE):
    for epsilon in epsilons:
        print(f'epslion {epsilon}...')
        adversarial_examples = make_example(model, test_loader, epsilon, DEVICE)
        with open(f'./examples/epslion_{epsilon}.pth', 'wb') as f:
            torch.save(adversarial_examples, f)
        print('done')


# 추론 결과 저장
def save_result(model, test_loader, DEVICE, epsilon):
    test_loss, test_accuracy, class_accuracy = evaluate(model, test_loader, DEVICE, fgsm=True)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    with open(f'./result/result_{epsilon}.txt', 'w') as f:
        line = f' - epsilon {epsilon} - \n'
        f.write(line)
        print(line[:-1])

        line = f'loss : {test_loss}\n'
        f.write(line)
        print(line[:-1])

        line = f'accuracy : {test_accuracy}\n'
        f.write(line)
        print(line[:-1])

        for idx in range(10):
            line = f'- {classes[idx]:<5} : {class_accuracy[idx]}\n'
            f.write(line)
            print(line[:-1])
    print()


if __name__ == '__main__':
    # CUDA 사용 가능 여부 확인
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    # CIFAR-10 준비
    test_loader = cifar_10(16)[1]
    print()

    # 모델 불러오기
    model = ResNet().to(DEVICE)
    model.load_state_dict(torch.load('./model_state_dict.pth'))
    model.eval()

    # 앱실론 0 : 원본
    epsilons = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    loop_epsilon(model, test_loader, epsilons, DEVICE)
    print('datasets generated successfully.\n')

    for epsilon in epsilons:
        with open(f'./examples/epslion_{epsilon}.pth', 'rb') as f:
            test_loader = torch.load(f)
        save_result(model, test_loader, DEVICE, epsilon)

    print('All Done!')
