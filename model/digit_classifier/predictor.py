from model.digit_classifier.model import (
    MobileNetClassifier,
    EfficientNetClassifier,
    ResNet18Classifier,
    build_model,
)
from torch.nn import functional as F
from model.digit_classifier.transform import NormalTransform
import torch

# import torch_tensorrt


class DigitClassifier:
    def __init__(self, model_type, weights_path, class_names=["1", "2", "3", "4", "S", "Q"]):
        """
        Initialize the digit classifier with the specified model type and load weights.
        """
        assert model_type in [
            "resnet18",
            "efficientnet",
            "mobilenet",
        ], f"Unsupported model type: {model_type}"

        num_classes = len(class_names)
        self.class_names = class_names
        self.model_type = model_type
        # Load the model weights
        self.model = build_model(model_type, num_classes, weights_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model = self.model.half()
        self.model.eval()

        # self.model = torch.compile(
        #     self.model,
        #     mode="default",  # 最大性能优化
        #     backend="inductor",  # 使用 Inductor 后端
        #     fullgraph=True,  # 强制编译整个图
        #     dynamic=True,  # 支持动态输入大小
        # )
        # self.model = torch.compile(
        #     self.model,
        #     backend="torch_tensorrt",
        #     options={
        #         "precision": torch.float16,  # 可选：使用 FP16 精度
        #         "workspace_size": 1 << 30,  # 可选：设置工作空间大小（1GB）
        #     },
        # )

        self.transform = NormalTransform(input_size=64)

    def predict(self, image, return_names=False):
        """
        Predict the class of the input image.
        """
        # Apply transformations
        image = self.transform(image)

        # Add batch dimension and convert to tensor
        image_tensor = image.unsqueeze(0)
        image_tensor = image_tensor.to(self.device, dtype=torch.float16)

        # Forward pass through the model
        with torch.no_grad():
            output = self.model(image_tensor)

        # Get the predicted class index
        _, predicted_idx = torch.max(output, 1)

        # Convert index to class name
        if return_names:
            predicted_class = self.class_names[predicted_idx.item()]
        else:
            predicted_class = predicted_idx.item()

        return predicted_class, F.softmax(output, dim=1).squeeze().tolist()

    def predict_batch(self, images, return_names=False):
        """
        Predict the classes of a batch of images.
        """

        # Apply transformations to each image
        transformed_images = [self.transform(image) for image in images]

        # Stack images into a single tensor
        image_tensor = torch.stack(transformed_images)
        image_tensor = image_tensor.to(self.device, dtype=torch.float16)

        # Forward pass through the model

        with torch.no_grad():
            outputs = self.model(image_tensor)

        # Get the predicted class indices
        _, predicted_indices = torch.max(outputs, 1)
        # Convert indices to class names
        if return_names:
            predicted_classes = [
                self.class_names[idx.item()] for idx in predicted_indices
            ]
        else:
            predicted_classes = [idx.item() for idx in predicted_indices]

        # Get probabilities for each class
        probabilities = F.softmax(outputs, dim=1).tolist()
        return predicted_classes, probabilities

    def get_class_names(self):
        """
        Get the class names of the classifier.
        """
        return self.class_names


if __name__ == "__main__":
    import os
    import random
    from PIL import Image
    import matplotlib.pyplot as plt
    from pathlib import Path

    # 设置matplotlib后端
    import matplotlib

    matplotlib.use("Agg")

    # 配置
    model_type = "mobilenet"  # 修改为字符串
    weights_path = "weights/MOBILENET_best_save0.pth"
    dataset_path = Path(
        "/home/fallengold/extra/pure_armor_dataset/pytorch_split/val"
    )  # 验证集路径

    # 初始化分类器
    classifier = DigitClassifier(model_type, weights_path)

    # 从数据集中采样图像
    def sample_images_from_dataset(dataset_path, num_samples=12):
        """从数据集中随机采样图像"""
        all_images = []
        class_names = ["B1", "B2", "B3", "B4", "BS", "R1", "R2", "R3", "R4", "RS"]

        for class_name in class_names:
            class_dir = dataset_path / class_name
            if class_dir.exists():
                image_files = list(class_dir.glob("*.jpg")) + list(
                    class_dir.glob("*.png")
                )
                for img_file in image_files:
                    # 映射到新的类别
                    if class_name in ["B1", "R1"]:
                        true_class = "1"
                    elif class_name in ["B2", "R2"]:
                        true_class = "2"
                    elif class_name in ["B3", "R3"]:
                        true_class = "3"
                    elif class_name in ["B4", "R4"]:
                        true_class = "4"
                    elif class_name in ["BS", "RS"]:
                        true_class = "S"
                    else:
                        continue

                    all_images.append((str(img_file), true_class, class_name))

        # 随机采样
        if len(all_images) > num_samples:
            sampled = random.sample(all_images, num_samples)
        else:
            sampled = all_images

        return sampled

    # 采样图像
    print("Sampling images from dataset...")
    sampled_images = sample_images_from_dataset(dataset_path, num_samples=16)
    print(f"Found {len(sampled_images)} images")

    if not sampled_images:
        print("No images found in dataset!")
        exit()

    # 加载图像并进行预测
    images = []
    true_labels = []
    original_classes = []

    for img_path, true_label, original_class in sampled_images:
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            true_labels.append(true_label)
            original_classes.append(original_class)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

    if not images:
        print("No valid images loaded!")
        exit()

    # 批量预测
    print("Making predictions...")
    predicted_classes, probabilities = classifier.predict_batch(images)

    # 可视化结果
    print("Creating visualization...")
    cols = 4
    rows = (len(images) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else []

    correct_count = 0

    for i, (img, true_label, pred_label, prob, orig_class) in enumerate(
        zip(images, true_labels, predicted_classes, probabilities, original_classes)
    ):
        if i >= len(axes):
            break

        # 显示图像
        axes[i].imshow(img)

        # 计算预测置信度
        pred_idx = classifier.class_names.index(pred_label)
        confidence = prob[pred_idx]

        # 设置标题
        if true_label == pred_label:
            title = f"✓ {orig_class} → {pred_label}\nConf: {confidence:.3f}"
            color = "green"
            correct_count += 1
        else:
            title = f"✗ {orig_class} → {pred_label}\nTrue: {true_label}, Conf: {confidence:.3f}"
            color = "red"

        axes[i].set_title(title, color=color, fontsize=10, weight="bold")
        axes[i].axis("off")

    # 隐藏多余的子图
    for i in range(len(images), len(axes)):
        axes[i].axis("off")

    # 添加总体统计信息
    accuracy = correct_count / len(images) * 100
    fig.suptitle(
        f"Armor Digit Classification Results\n"
        f"Accuracy: {correct_count}/{len(images)} = {accuracy:.1f}%\n"
        f"Model: {model_type.upper()}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    # 保存结果
    output_path = "prediction_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Results saved to: {output_path}")

    # 打印详细结果
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"Total images: {len(images)}")
    print(f"Correct predictions: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("\nDetailed results:")

    for i, (true_label, pred_label, prob, orig_class) in enumerate(
        zip(true_labels, predicted_classes, probabilities, original_classes)
    ):
        pred_idx = classifier.class_names.index(pred_label)
        confidence = prob[pred_idx]
        status = "✓" if true_label == pred_label else "✗"
        print(
            f"{i+1:2d}. {status} {orig_class:3s} → Pred: {pred_label:3s} (True: {true_label:3s}) Conf: {confidence:.3f}"
        )

    # 显示类别分布
    print(f"\nClass distribution in sample:")
    from collections import Counter

    true_dist = Counter(true_labels)
    pred_dist = Counter(predicted_classes)

    print("True labels:")
    for class_name in classifier.class_names:
        count = true_dist.get(class_name, 0)
        print(f"  {class_name}: {count}")

    print("Predicted labels:")
    for class_name in classifier.class_names:
        count = pred_dist.get(class_name, 0)
        print(f"  {class_name}: {count}")

    plt.close()
