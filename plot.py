import torch
from torchvision import utils
import numpy as np
import matplotlib.pyplot as plt

# resnet 임포트
from resnet import *


# 테스트 데이터를 파일로 저장
def save_result():
    # CUDA 사용 가능 여부 확인
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    # 모델 불러오기
    model = ResNet().to(DEVICE)
    model.load_state_dict(torch.load('./model_state_dict.pth'))
    model.eval()

    # 결괏값 저장
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

    total_result_data = [examples, total_loss, total_accuracy, total_class_accuracy]
    with open('./result/total_result_data', 'wb') as f:
        torch.save(total_result_data, f)
    print('saved successfully')


# 적대적 예제 샘플 시각화
def plot_examples(examples):
    for images in examples:
        images = utils.make_grid(images[0])
        images = images / 2 + 0.5
        images = images.to('cpu')
        npimg = images.detach().numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


# 손실함수 그래프화
def plot_loss(total_loss, epsilons):
    plt.plot(epsilons, total_loss, '*-')
    plt.yticks(np.arange(0, 4.9, step=0.4))
    plt.xticks(np.arange(0, 0.11, step=0.01))
    plt.title('Total Loss')
    plt.xlabel('Epsilon')
    plt.ylabel('Loss')
    plt.show()


# 인식률 그래프화
def plot_accuracy(total_accuracy, epsilons):
    plt.plot(epsilons, total_accuracy, '*-')
    plt.yticks(np.arange(0, 101, step=10))
    plt.xticks(np.arange(0, 0.11, step=0.01))
    plt.title('Total Accuracy')
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy')
    plt.show()


# 클래스별 인식률 그래프화
def plot_class_accuracy(accuracy, epsilons, class_name):
    plt.plot(epsilons, accuracy, '*-')
    plt.yticks(np.arange(0, 101, step=10))
    plt.xticks(np.arange(0, 0.11, step=0.01))
    plt.title(class_name)
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    epsilons = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    save_result()

    with open('./result/total_result_data', 'rb') as f:
        examples, total_loss, total_accuracy, total_class_accuracy = torch.load(f)

    plot_examples(examples)
    plot_loss(total_loss, epsilons)
    plot_accuracy(total_accuracy, epsilons)
    for index in range(len(classes)):
        accuracy = []
        for epsilon in total_class_accuracy:
            accuracy.append(epsilon[index])
        plot_class_accuracy(accuracy, epsilons, classes[index])

    print('Done!')
