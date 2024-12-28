import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# 定义SimpleCNN模型（与训练时的模型保持一致）
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


# 加载测试数据
def create_test_loader(test_dir, batch_size=32, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader


# 测试模型并输出预测结果和真实结果
def test_model(model, test_loader, class_names):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # 切换为评估模式

    running_corrects = 0
    total_images = 0

    with torch.no_grad():  # 在测试时关闭梯度计算
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)
            total_images += labels.size(0)

            # 输出每张图像的预测和真实结果
            for i in range(inputs.size(0)):
                label = labels[i].item()  # 获取真实标签
                pred = preds[i].item()  # 获取预测标签
                print(f'Pred: {class_names[pred]} | Actual: {class_names[label]}')

    accuracy = running_corrects.double() / total_images
    return accuracy


def main():
    # 1. 设置路径和参数
    model_path = 'trained_model.pth'  # 训练模型路径
    test_dir = './chest_xray/test'  # 测试集的路径
    class_names = ['Pneumonia', 'Normal']

    # 2. 加载模型
    model = SimpleCNN(num_classes=2)  # 确保与训练时模型结构一致
    model.load_state_dict(torch.load(model_path))

    # 3. 加载测试数据
    test_loader = create_test_loader(test_dir)

    # 4. 测试模型并输出结果
    accuracy = test_model(model, test_loader, class_names)
    print(f'Model accuracy on test set: {accuracy:.4f}')


if __name__ == "__main__":
    main()
