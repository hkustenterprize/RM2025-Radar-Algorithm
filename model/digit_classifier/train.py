import matplotlib

matplotlib.use("Agg")  # 使用非交互式后端以避免显示窗口
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from pathlib import Path
from model.digit_classifier.model import (
    ResNet18Classifier,
    EfficientNetClassifier,
    MobileNetClassifier,
)
import cv2
import numpy as np
from PIL import Image
import random
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model.digit_classifier.transform import (
    RandomErase,
    RandomMask,
    RandomLines,
    RandomShadow,
    RandomNoise,
    PadToSquare,
    PadToSquareTensor,
)
import datetime
import argparse


# 新的类别映射 - 6类（将B0和R0映射为Q）
class_mapping = {
    # 数字1
    "B1": 0,
    "R1": 0,
    # 数字2
    "B2": 1,
    "R2": 1,
    # 数字3
    "B3": 2,
    "R3": 2,
    # 数字4
    "B4": 3,
    "R4": 3,
    # 哨兵
    "BS": 4,
    "RS": 4,
    # 前哨站Q（实际在B0/R0文件夹中）
    "B0": 5,
    "R0": 5,
}

class_names_new = ["1", "2", "3", "4", "S", "Q"]  # 添加Q

# 原始类别名称
original_class_names = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B0",
    "BS",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R0",
    "RS",
]


class ArmorDigitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: PyTorch格式的数据集根目录 (train/ 或 val/)
            transform: 数据变换
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        # 获取所有有效的样本
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        """加载所有有效的样本"""
        # 遍历所有原始类别文件夹
        for original_class in original_class_names:
            class_dir = self.root_dir / original_class
            if not class_dir.exists():
                continue

            # 检查是否是我们需要的类别
            if original_class in class_mapping:
                new_class_id = class_mapping[original_class]

                # 获取该类别下的所有图像文件
                image_files = (
                    list(class_dir.glob("*.jpg"))
                    + list(class_dir.glob("*.jpeg"))
                    + list(class_dir.glob("*.png"))
                )

                for image_file in image_files:
                    self.samples.append((str(image_file), new_class_id))

        print(f"Loaded {len(self.samples)} valid samples")

        # 打印类别分布
        class_counts = {}
        for _, class_id in self.samples:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1

        print("Class distribution:")
        for class_id, count in sorted(class_counts.items()):
            print(f"  {class_names_new[class_id]}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, class_id = self.samples[idx]

        # 加载图像
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, class_id


class ArmorDigitTrainer:
    def __init__(
        self,
        train_dir,
        val_dir,
        batch_size=32,
        learning_rate=0.001,
        input_size=64,
        device=None,
        model_type=EfficientNetClassifier,
        use_class_weight=True,
        workspace_root="./training_workspace",  # 新增工作区根目录
    ):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_class_weight = use_class_weight
        self.input_size = input_size

        # 创建工作区
        self.workspace_root = Path(workspace_root)
        self.setup_workspace()

        # 设备设置
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"Using device: {self.device}")
        print(f"Workspace: {self.workspace}")

        # 数据增强
        self.train_transform = transforms.Compose(
            [
                # 几何变换（在padding之前）
                transforms.RandomRotation(degrees=10),  # 减少旋转角度，避免过度变形
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=5,  # 添加轻微的剪切变换
                ),
                # Padding成正方形
                PadToSquare(fill=0, padding_mode="constant"),  # 黑色填充
                # 现在resize到目标尺寸
                transforms.Resize((self.input_size, self.input_size)),
                # 随机翻转（概率降低，因为数字有方向性）
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.RandomVerticalFlip(p=0.05),
                # 透视变换
                transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
                # 颜色变换
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
                ),
                # 转换为tensor
                # 自定义遮挡变换（在tensor格式下）
                RandomErase(
                    p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"
                ),
                RandomMask(p=0.2, mask_ratio=(0.05, 0.15)),
                RandomLines(p=0.15, num_lines=(1, 2), line_width=(1, 2)),
                RandomNoise(p=0.15, noise_level=(0.02, 0.08)),
                RandomShadow(p=0.15, shadow_strength=(0.5, 0.8)),
                # 标准化
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # 验证集变换 - 只padding和resize，不做数据增强
        self.val_transform = transforms.Compose(
            [
                PadToSquare(fill=0, padding_mode="constant"),
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # 用于可视化的验证变换（不包含标准化）
        self.vis_transform = transforms.Compose(
            [
                PadToSquare(fill=0, padding_mode="constant"),
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
            ]
        )

        # 创建数据集和数据加载器
        self._create_datasets()

        # 创建模型
        self.model = model_type(num_classes=6, pretrained=True).to(self.device)  # 改为6
        self.model_name = self.model.__class__.__name__

        # 损失函数和优化器
        if self.use_class_weight:
            class_weights = self.calculate_class_weights()
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-4
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.3, patience=4, verbose=True
        )

        # 训练历史
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def setup_workspace(self):
        """设置训练工作区"""
        # 创建时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建工作区目录
        self.workspace = self.workspace_root / f"training_{timestamp}"
        self.workspace.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        (self.workspace / "models").mkdir(exist_ok=True)
        (self.workspace / "visualizations").mkdir(exist_ok=True)
        (self.workspace / "confusion_matrices").mkdir(exist_ok=True)
        (self.workspace / "sample_predictions").mkdir(exist_ok=True)
        (self.workspace / "logs").mkdir(exist_ok=True)

        print(f"Created workspace: {self.workspace}")

    def _create_datasets(self):
        """创建训练和验证数据集"""
        # PyTorch格式：直接传入train/val目录
        self.train_dataset = ArmorDigitDataset(
            self.train_dir, transform=self.train_transform
        )

        self.val_dataset = ArmorDigitDataset(self.val_dir, transform=self.val_transform)

        # 用于可视化的验证数据集（不标准化）
        self.val_vis_dataset = ArmorDigitDataset(
            self.val_dir, transform=self.vis_transform
        )

        # 数据加载器
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

        print(f"Train dataset: {len(self.train_dataset)} samples")
        print(f"Val dataset: {len(self.val_dataset)} samples")

    def calculate_class_weights(self):
        """计算类别权重用于处理类别不平衡"""
        # 统计每个类别的样本数量
        class_counts = {i: 0 for i in range(6)}  # 改为6个类别

        for _, class_id in self.train_dataset.samples:
            class_counts[class_id] += 1

        # 计算总样本数
        total_samples = sum(class_counts.values())

        # 计算权重：使用inverse frequency
        class_weights = []
        for i in range(6):  # 改为6
            if class_counts[i] > 0:
                # 权重 = total_samples / (num_classes * class_count)
                weight = total_samples / (6 * class_counts[i])
            else:
                weight = 0.0
            class_weights.append(weight)

        # 打印权重信息
        print("\nClass weights calculation:")
        print("-" * 40)
        for i, (count, weight) in enumerate(zip(class_counts.values(), class_weights)):
            print(f"  {class_names_new[i]}: {count:>6} samples, weight: {weight:.4f}")

        return torch.FloatTensor(class_weights).to(self.device)

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images, targets = images.to(self.device), targets.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()

            # 打印进度
            if batch_idx % 10 == 0:
                print(
                    f"Batch [{batch_idx}/{len(self.train_loader)}], Loss: {loss.item():.4f}"
                )

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct_predictions / total_predictions

        return epoch_loss, epoch_acc

    def validate(self, epoch=None):
        """验证模型"""
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets in self.val_loader:
                images, targets = images.to(self.device), targets.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_targets, all_predictions)

        # 生成可视化
        if epoch is not None:
            self.visualize_predictions(epoch, all_targets, all_predictions)
            self.save_confusion_matrix(epoch, all_targets, all_predictions)

        return epoch_loss, epoch_acc, all_predictions, all_targets

    def save_confusion_matrix(self, epoch, targets, predictions):
        """保存混淆矩阵"""
        try:
            cm = confusion_matrix(targets, predictions)

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names_new,
                yticklabels=class_names_new,
                cbar_kws={"label": "Count"},
            )

            # 计算每个类别的准确率
            class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)  # 避免除零

            plt.title(
                f"Confusion Matrix - Epoch {epoch}\nOverall Acc: {accuracy_score(targets, predictions):.3f}"
            )
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")

            # 添加类别准确率信息
            textstr = "\n".join(
                [
                    f"{class_names_new[i]}: {acc:.3f}"
                    for i, acc in enumerate(class_acc)
                    if not np.isnan(acc)
                ]
            )
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            plt.figtext(
                0.02, 0.98, textstr, fontsize=9, verticalalignment="top", bbox=props
            )

            plt.tight_layout()

            # 保存图像
            save_path = (
                self.workspace
                / "confusion_matrices"
                / f"epoch_{epoch:03d}_confusion_matrix.png"
            )
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"Saved confusion matrix: {save_path}")

        except Exception as e:
            print(f"Error saving confusion matrix: {e}")
            import traceback

            traceback.print_exc()

    def visualize_predictions(self, epoch, targets, predictions, num_samples=20):
        """可视化预测结果 - 重点显示错误样本"""
        # 找出所有错误预测的索引
        error_indices = [i for i in range(len(targets)) if targets[i] != predictions[i]]
        correct_indices = [
            i for i in range(len(targets)) if targets[i] == predictions[i]
        ]

        # 选择样本：优先显示错误样本，然后是正确样本
        selected_indices = []

        # 先添加错误样本（最多占一半）
        max_errors = min(num_samples // 2, len(error_indices))
        if error_indices:
            selected_indices.extend(random.sample(error_indices, max_errors))

        # 再添加正确样本
        remaining_slots = num_samples - len(selected_indices)
        if correct_indices and remaining_slots > 0:
            max_correct = min(remaining_slots, len(correct_indices))
            selected_indices.extend(random.sample(correct_indices, max_correct))

        # 如果错误样本不够，用正确样本补充
        if len(selected_indices) < num_samples and correct_indices:
            additional_needed = num_samples - len(selected_indices)
            additional_correct = random.sample(
                correct_indices, min(additional_needed, len(correct_indices))
            )
            selected_indices.extend(additional_correct)

        # 创建图像网格
        cols = 4
        rows = (len(selected_indices) + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else []

        for i, result_idx in enumerate(selected_indices):
            if i >= len(axes):
                break

            # 获取对应的图像和预测结果
            image, true_label = self.val_vis_dataset[result_idx]
            pred_label = predictions[result_idx]

            # 转换图像为可显示格式
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).numpy()

            # 显示图像
            axes[i].imshow(image)

            # 设置标题
            true_class = class_names_new[true_label]
            pred_class = class_names_new[pred_label]

            if true_label == pred_label:
                title = f"✓ True: {true_class}, Pred: {pred_class}"
                color = "green"
            else:
                title = f"✗ True: {true_class}, Pred: {pred_class}"
                color = "red"

            axes[i].set_title(title, color=color, fontsize=10, weight="bold")
            axes[i].axis("off")

        # 隐藏多余的子图
        for i in range(len(selected_indices), len(axes)):
            axes[i].axis("off")

        # 添加总体信息
        error_count = len(error_indices)
        total_count = len(targets)
        error_rate = error_count / total_count * 100

        fig.suptitle(
            f"Epoch {epoch} - Predictions (Errors: {error_count}/{total_count} = {error_rate:.1f}%)",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        # 保存图像
        save_path = (
            self.workspace / "sample_predictions" / f"epoch_{epoch:03d}_predictions.png"
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(
            f"Saved prediction samples: {save_path} (Errors: {len([i for i in selected_indices if i in error_indices])}/{len(selected_indices)})"
        )

    def train(self, num_epochs=50, save_best=True, visualize_every=5):
        """训练模型"""
        best_val_acc = 0.0
        best_model_path = (
            self.workspace / "models" / f"best_armor_classifier_{self.model_name}.pth"
        )

        print("Starting training...")
        print("=" * 50)

        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print("-" * 30)

            # 训练
            train_loss, train_acc = self.train_epoch()

            # 验证 - 传入epoch用于可视化
            should_visualize = (epoch + 1) % visualize_every == 0 or epoch == 0
            val_loss, val_acc, val_preds, val_targets = self.validate(
                epoch + 1 if should_visualize else None
            )

            # 学习率调度
            self.scheduler.step(val_loss)

            # 记录历史
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # 打印结果
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # 保存最佳模型
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_val_acc": best_val_acc,
                        "class_mapping": class_mapping,
                        "class_names": class_names_new,
                        "workspace": str(self.workspace),
                    },
                    best_model_path,
                )
                print(f"New best model saved! Val Acc: {best_val_acc:.4f}")

            # 保存训练历史
            self.save_training_history(epoch + 1)

        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")

        # 最终验证报告
        self.print_classification_report(val_targets, val_preds)

        return str(best_model_path)

    def save_training_history(self, current_epoch):
        """保存训练历史图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 损失曲线
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(
            epochs, self.train_losses, label="Train Loss", marker="o", markersize=3
        )
        ax1.plot(epochs, self.val_losses, label="Val Loss", marker="s", markersize=3)
        ax1.set_title(f"Training and Validation Loss (Epoch {current_epoch})")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 准确率曲线
        ax2.plot(
            epochs,
            self.train_accuracies,
            label="Train Accuracy",
            marker="o",
            markersize=3,
        )
        ax2.plot(
            epochs, self.val_accuracies, label="Val Accuracy", marker="s", markersize=3
        )
        ax2.set_title(f"Training and Validation Accuracy (Epoch {current_epoch})")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存到可视化目录
        save_path = self.workspace / "visualizations" / "training_history.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def print_classification_report(self, targets, predictions):
        """打印分类报告"""
        print("\n" + "=" * 50)
        print("CLASSIFICATION REPORT")
        print("=" * 50)

        # 分类报告
        report = classification_report(
            targets, predictions, target_names=class_names_new
        )
        print(report)

        # 保存报告到文件
        report_path = self.workspace / "logs" / "final_classification_report.txt"
        with open(report_path, "w") as f:
            f.write("FINAL CLASSIFICATION REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(report)

        # 最终混淆矩阵
        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names_new,
            yticklabels=class_names_new,
        )
        plt.title("Final Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        final_cm_path = self.workspace / "visualizations" / "final_confusion_matrix.png"
        plt.savefig(final_cm_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Final confusion matrix saved: {final_cm_path}")

    def plot_training_history(self):
        """绘制训练历史（最终版本）"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        epochs = range(1, len(self.train_losses) + 1)

        # 损失曲线
        ax1.plot(epochs, self.train_losses, label="Train Loss", linewidth=2)
        ax1.plot(epochs, self.val_losses, label="Val Loss", linewidth=2)
        ax1.set_title("Training and Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 准确率曲线
        ax2.plot(epochs, self.train_accuracies, label="Train Accuracy", linewidth=2)
        ax2.plot(epochs, self.val_accuracies, label="Val Accuracy", linewidth=2)
        ax2.set_title("Training and Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        final_history_path = (
            self.workspace / "visualizations" / "final_training_history.png"
        )
        plt.savefig(final_history_path, dpi=300, bbox_inches="tight")
        plt.show()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train Armor Digit Classifier')

    # 数据集路径
    parser.add_argument('--dataset-path', type=str, 
                       default='/localdata/szhoubx/rm/data/pytorch_split',
                       help='Path to the dataset directory containing train/ and val/ subdirs')

    # 批次大小
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training (default: 32)')

    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()

    # 数据路径
    dataset_path = Path(args.dataset_path)
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"

    # 检查数据路径
    if not train_dir.exists() or not val_dir.exists():
        print("Error: Train or validation directory not found!")
        print(f"Expected train dir: {train_dir}")
        print(f"Expected val dir: {val_dir}")
        return

    # 检查是否有预期的类别文件夹
    print("Checking dataset structure...")
    for split_name, split_dir in [("Train", train_dir), ("Val", val_dir)]:
        print(f"\n{split_name} directory contents:")
        if split_dir.exists():
            class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
            for class_dir in sorted(class_dirs):
                image_count = (
                    len(list(class_dir.glob("*.jpg")))
                    + len(list(class_dir.glob("*.jpeg")))
                    + len(list(class_dir.glob("*.png")))
                )
                if class_dir.name in class_mapping:
                    mapped_class = class_names_new[class_mapping[class_dir.name]]
                    print(f"  {class_dir.name} -> {mapped_class}: {image_count} images")
                else:
                    print(f"  {class_dir.name} (ignored): {image_count} images")

    # 创建训练器
    trainer = ArmorDigitTrainer(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=args.batch_size,  # 使用参数
        learning_rate=0.0003,
        model_type=MobileNetClassifier,
        workspace_root="./training_workspace",
    )

    # 训练模型
    best_model_path = trainer.train(
        num_epochs=100, save_best=True, visualize_every=5
    )

    # 绘制最终训练历史
    trainer.plot_training_history()

    print(f"Best model saved at: {best_model_path}")
    print(f"Training workspace: {trainer.workspace}")


if __name__ == "__main__":
    main()
