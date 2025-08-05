import numpy as np
from PIL import Image, ImageDraw
import random
import torch
from torchvision import transforms


class PadToSquare(object):
    """将图像padding成正方形"""

    def __init__(self, fill=0, padding_mode="constant"):
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): 输入图像
        Returns:
            PIL Image: Padding后的正方形图像
        """
        w, h = img.size
        max_size = max(w, h)

        # 计算padding大小
        pad_w = max_size - w
        pad_h = max_size - h

        # 计算左右上下的padding
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        # 应用padding
        padding = (pad_left, pad_top, pad_right, pad_bottom)

        if self.padding_mode == "constant":
            # 使用常数填充
            if isinstance(self.fill, int):
                fill_color = (self.fill, self.fill, self.fill)
            else:
                fill_color = self.fill

            # 创建新的正方形图像
            new_img = Image.new(img.mode, (max_size, max_size), fill_color)
            new_img.paste(img, (pad_left, pad_top))
            return new_img
        else:
            # 使用PIL的pad功能（支持reflect, edge等模式）
            return transforms.functional.pad(img, padding, self.fill, self.padding_mode)


class PadToSquareTensor(object):
    """将tensor图像padding成正方形（用于tensor格式）"""

    def __init__(self, fill=0, padding_mode="constant"):
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): 输入tensor，形状为 (C, H, W)
        Returns:
            Tensor: Padding后的正方形tensor
        """
        _, h, w = tensor.shape
        max_size = max(w, h)

        # 计算padding大小
        pad_w = max_size - w
        pad_h = max_size - h

        # 计算左右上下的padding
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        # PyTorch的pad格式: (pad_left, pad_right, pad_top, pad_bottom)
        padding = (pad_left, pad_right, pad_top, pad_bottom)

        return torch.nn.functional.pad(tensor, padding, self.padding_mode, self.fill)


class RandomErase(object):
    """随机擦除变换"""

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, img):
        if random.random() > self.p:
            return img

        area = img.size[0] * img.size[1]

        for _ in range(100):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            h = int(round((target_area * aspect_ratio) ** 0.5))
            w = int(round((target_area / aspect_ratio) ** 0.5))

            if w < img.size[0] and h < img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                if isinstance(self.value, (int, float)):
                    # 单色填充
                    img_array = np.array(img)
                    img_array[y1 : y1 + h, x1 : x1 + w] = self.value
                    return Image.fromarray(img_array)
                else:
                    # 随机噪声填充
                    img_array = np.array(img)
                    img_array[y1 : y1 + h, x1 : x1 + w] = np.random.randint(
                        0, 256, (h, w, 3)
                    )
                    return Image.fromarray(img_array)

        return img


class RandomMask(object):
    """随机遮罩变换"""

    def __init__(self, p=0.3, mask_ratio=(0.1, 0.3)):
        self.p = p
        self.mask_ratio = mask_ratio

    def __call__(self, img):
        if random.random() > self.p:
            return img

        img_array = np.array(img)
        h, w = img_array.shape[:2]

        # 随机选择遮罩比例
        ratio = random.uniform(*self.mask_ratio)
        mask_h = int(h * ratio)
        mask_w = int(w * ratio)

        # 随机选择遮罩位置
        x = random.randint(0, max(1, w - mask_w))
        y = random.randint(0, max(1, h - mask_h))

        # 应用遮罩（随机颜色）
        mask_color = np.random.randint(0, 256, 3)
        img_array[y : y + mask_h, x : x + mask_w] = mask_color

        return Image.fromarray(img_array)


class RandomLines(object):
    """随机线条遮挡"""

    def __init__(self, p=0.3, num_lines=(1, 3), line_width=(1, 3)):
        self.p = p
        self.num_lines = num_lines
        self.line_width = line_width

    def __call__(self, img):
        if random.random() > self.p:
            return img

        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw)

        num_lines = random.randint(*self.num_lines)

        for _ in range(num_lines):
            # 随机线条颜色
            color = tuple(np.random.randint(0, 256, 3))

            # 随机线条宽度
            width = random.randint(*self.line_width)

            # 随机线条位置（可能是水平、垂直或斜线）
            line_type = random.choice(["horizontal", "vertical", "diagonal"])

            if line_type == "horizontal":
                y = random.randint(0, img.size[1])
                draw.line([(0, y), (img.size[0], y)], fill=color, width=width)
            elif line_type == "vertical":
                x = random.randint(0, img.size[0])
                draw.line([(x, 0), (x, img.size[1])], fill=color, width=width)
            else:  # diagonal
                x1, y1 = random.randint(0, img.size[0]), random.randint(0, img.size[1])
                x2, y2 = random.randint(0, img.size[0]), random.randint(0, img.size[1])
                draw.line([(x1, y1), (x2, y2)], fill=color, width=width)

        return img_draw


class RandomNoise(object):
    """随机噪声"""

    def __init__(self, p=0.3, noise_level=(0.05, 0.15)):
        self.p = p
        self.noise_level = noise_level

    def __call__(self, img):
        if random.random() > self.p:
            return img

        img_array = np.array(img).astype(np.float32)

        # 添加高斯噪声
        noise_std = random.uniform(*self.noise_level) * 255
        noise = np.random.normal(0, noise_std, img_array.shape)

        noisy_img = img_array + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy_img)


class RandomShadow(object):
    """随机阴影"""

    def __init__(self, p=0.3, shadow_strength=(0.3, 0.7)):
        self.p = p
        self.shadow_strength = shadow_strength

    def __call__(self, img):
        if random.random() > self.p:
            return img

        img_array = np.array(img).astype(np.float32)
        h, w = img_array.shape[:2]

        # 创建随机阴影区域
        shadow_ratio = random.uniform(0.2, 0.8)

        if random.random() > 0.5:
            # 水平阴影
            shadow_h = int(h * shadow_ratio)
            y_start = random.randint(0, h - shadow_h)
            shadow_mask = np.ones((h, w))
            shadow_mask[y_start : y_start + shadow_h, :] = random.uniform(
                *self.shadow_strength
            )
        else:
            # 垂直阴影
            shadow_w = int(w * shadow_ratio)
            x_start = random.randint(0, w - shadow_w)
            shadow_mask = np.ones((h, w))
            shadow_mask[:, x_start : x_start + shadow_w] = random.uniform(
                *self.shadow_strength
            )

        # 应用阴影
        for c in range(3):
            img_array[:, :, c] *= shadow_mask

        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)


class NormalTransform:

    def __init__(self, input_size=224):
        self.input_size = input_size
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

    def __call__(self, image):
        """
        Args:
            image (PIL Image): 输入图像
        Returns:
            Tensor: 预处理后的图像tensor
        """
        return self.val_transform(image)
