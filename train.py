import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt  # 导入matplotlib


# 定义CNN模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        # 加入Dropout层以防止过拟合
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# 数据预处理和加载的函数
def create_dataloaders(data_dir, batch_size=32, num_workers=4):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
        transforms.RandomRotation(15),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 颜色调整
        transforms.RandomResizedCrop(224),  # 随机裁剪
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = f'{data_dir}/train'
    val_dir = f'{data_dir}/val'
    test_dir = f'{data_dir}/test'

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


# 训练模型的函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=1, model_save_path='best_model.pth',
                patience=5, lr_scheduler=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Early stopping计数器
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            if phase == 'train':
                # 使用 tqdm 显示训练进度
                pbar = tqdm(total=len(data_loader))
                for inputs, labels in data_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    with torch.set_grad_enabled(True):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    pbar.update(1)
                    pbar.set_postfix({'loss': f'{running_loss / (inputs.size(0) * pbar.n):.4f}',
                                      'acc': f'{running_corrects.double() / (inputs.size(0) * pbar.n):.4f}'})

                pbar.close()

            else:  # val阶段不使用tqdm
                with torch.no_grad():
                    for inputs, labels in data_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.item())

            # 保存最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                early_stopping_counter = 0  # 如果验证精度提升，则重置计数器
            elif phase == 'val' and epoch_acc <= best_acc:
                early_stopping_counter += 1

            # 早停条件
            if early_stopping_counter >= patience:
                print(f'Early stopping triggered at epoch {epoch + 1}')
                break

        if early_stopping_counter >= patience:
            break

        # 如果存在学习率调度器，更新学习率
        if lr_scheduler:
            lr_scheduler.step()

    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)

    # 保存最佳模型
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

    # 绘制损失和精度曲线
    plot_training_curve(train_losses, val_losses, train_accuracies, val_accuracies)

    return model


# 调整优化器和学习率调度器
def adjust_optimizer_and_scheduler(model, lr=0.001, weight_decay=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 使用学习率调度器，随着训练进行逐步衰减学习率
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    return optimizer, scheduler

# 绘制训练过程中的损失和精度曲线
def plot_training_curve(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Val Loss', color='red')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制精度曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(epochs, val_accuracies, label='Val Accuracy', color='red')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 显示图表
    plt.tight_layout()
    plt.show()

# 主函数
if __name__ == '__main__':
    data_dir = './chest_xray'
    batch_size = 32
    num_workers = 4

    train_loader, val_loader, test_loader = create_dataloaders(data_dir, batch_size, num_workers)

    model = SimpleCNN(num_classes=2)
    criterion = nn.CrossEntropyLoss()

    # 调整优化器和学习率调度器
    optimizer, scheduler = adjust_optimizer_and_scheduler(model, lr=0.0001, weight_decay=1e-4)  # 学习率较低
    model_save_path = 'trained_model.pth'

    # 训练模型
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, model_save_path=model_save_path, lr_scheduler=scheduler)
