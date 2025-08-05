# -*- coding: utf-8 -*-
"""
Average Meter utility for RM2025 Radar Algorithm System
Copyright (c) 2025 香港科技大学ENTERPRIZE战队（HKUST ENTERPRIZE Team）

Licensed under the MIT License. See LICENSE file in the project root for license information.
"""


class AverageMeter:
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有统计量"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """更新统计量
        Args:
            val: 当前值
            n: 当前值的计数，默认为1
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

    def __str__(self):
        """返回平均值的字符串表示"""
        return f"{self.avg:.4f}"

