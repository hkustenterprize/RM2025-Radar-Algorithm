from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class ZoomableDraggableLabel(QLabel):
    """支持缩放和拖拽的图像标签"""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 300)
        self.setAlignment(Qt.AlignCenter)

        # 缩放和拖拽相关变量
        self.scale_factor = 1.0
        self.min_scale = 0.1
        self.max_scale = 5.0
        self.scale_step = 0.1

        # 拖拽相关变量
        self.dragging = False
        self.last_pan_point = QPoint()
        self.image_offset = QPoint(0, 0)

        # 原始图像
        self.original_pixmap = None
        self.scaled_pixmap = None

        # 设置鼠标追踪和接受拖拽事件
        self.setMouseTracking(True)
        self.setAcceptDrops(False)

        # 设置样式
        self.setStyleSheet("border: 2px solid #333; background-color: #f0f0f0;")

    def setPixmap(self, pixmap):
        """设置图像并重置缩放"""
        self.original_pixmap = pixmap
        self.scale_factor = 1.0
        self.image_offset = QPoint(0, 0)
        self.update_display()

    def update_display(self):
        """更新图像显示"""
        if self.original_pixmap is None:
            return

        # 计算缩放后的尺寸
        scaled_size = self.original_pixmap.size() * self.scale_factor
        self.scaled_pixmap = self.original_pixmap.scaled(
            scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # 创建显示图像
        self.create_display_image()

    def create_display_image(self):
        """创建用于显示的图像"""
        if self.scaled_pixmap is None:
            return

        # 获取控件尺寸
        widget_size = self.size()

        # 创建背景图像
        display_image = QPixmap(widget_size)
        display_image.fill(QColor(240, 240, 240))  # 浅灰色背景

        # 计算图像在控件中的位置
        image_rect = self.scaled_pixmap.rect()

        # 应用偏移
        image_rect.translate(self.image_offset)

        # 居中显示（当图像小于控件时）
        if self.scaled_pixmap.width() < widget_size.width():
            image_rect.moveLeft((widget_size.width() - self.scaled_pixmap.width()) // 2)
        if self.scaled_pixmap.height() < widget_size.height():
            image_rect.moveTop(
                (widget_size.height() - self.scaled_pixmap.height()) // 2
            )

        # 绘制图像
        painter = QPainter(display_image)
        painter.drawPixmap(image_rect, self.scaled_pixmap)

        # 绘制缩放信息
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.setFont(QFont("Arial", 12))
        scale_text = f"缩放: {self.scale_factor:.1f}x"
        painter.drawText(10, 25, scale_text)

        painter.end()

        # 显示最终图像
        super().setPixmap(display_image)

    def wheelEvent(self, event):
        """鼠标滚轮缩放"""
        if self.original_pixmap is None:
            return

        # 获取鼠标位置
        mouse_pos = event.pos()

        # 计算缩放前鼠标在图像中的相对位置
        old_scale = self.scale_factor

        # 根据滚轮方向调整缩放
        if event.angleDelta().y() > 0:
            self.scale_factor = min(self.max_scale, self.scale_factor + self.scale_step)
        else:
            self.scale_factor = max(self.min_scale, self.scale_factor - self.scale_step)

        # 如果缩放发生变化，调整偏移以保持鼠标位置不变
        if old_scale != self.scale_factor:
            scale_ratio = self.scale_factor / old_scale

            # 计算新的偏移
            widget_center = QPoint(self.width() // 2, self.height() // 2)
            mouse_offset = mouse_pos - widget_center - self.image_offset
            new_mouse_offset = mouse_offset * scale_ratio
            self.image_offset = mouse_pos - widget_center - new_mouse_offset

            self.update_display()

    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.last_pan_point = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.dragging and self.original_pixmap is not None:
            # 计算偏移量
            delta = event.pos() - self.last_pan_point
            self.image_offset += delta
            self.last_pan_point = event.pos()

            # 限制拖拽范围（可选）
            self.constrain_offset()

            self.update_display()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.setCursor(
                Qt.OpenHandCursor if self.original_pixmap else Qt.ArrowCursor
            )
        super().mouseReleaseEvent(event)

    def enterEvent(self, event):
        """鼠标进入控件"""
        if self.original_pixmap:
            self.setCursor(Qt.OpenHandCursor)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """鼠标离开控件"""
        self.setCursor(Qt.ArrowCursor)
        super().leaveEvent(event)

    def constrain_offset(self):
        """限制偏移范围，防止图像完全移出视野"""
        if self.scaled_pixmap is None:
            return

        widget_size = self.size()
        image_size = self.scaled_pixmap.size()

        # 计算允许的最大偏移
        # 当图像大于控件时，允许的偏移范围应该让图像边缘能够到达控件边缘
        if image_size.width() > widget_size.width():
            # 图像比控件宽，可以左右拖拽
            # 最右侧：图像左边缘到达控件右边缘 -> offset_x = widget_width - image_width
            # 最左侧：图像右边缘到达控件左边缘 -> offset_x = 0
            min_offset_x = widget_size.width() - image_size.width()
            max_offset_x = 0
            self.image_offset.setX(
                max(min_offset_x, min(max_offset_x, self.image_offset.x()))
            )
        else:
            # 图像比控件小，保持居中
            self.image_offset.setX(0)

        if image_size.height() > widget_size.height():
            # 图像比控件高，可以上下拖拽
            # 最下侧：图像上边缘到达控件下边缘 -> offset_y = widget_height - image_height
            # 最上侧：图像下边缘到达控件上边缘 -> offset_y = 0
            min_offset_y = widget_size.height() - image_size.height()
            max_offset_y = 0
            self.image_offset.setY(
                max(min_offset_y, min(max_offset_y, self.image_offset.y()))
            )
        else:
            # 图像比控件小，保持居中
            self.image_offset.setY(0)

    def reset_view(self):
        """重置视图到初始状态"""
        self.scale_factor = 1.0
        self.image_offset = QPoint(0, 0)
        self.update_display()

    def fit_to_window(self):
        """适应窗口大小"""
        if self.original_pixmap is None:
            return

        widget_size = self.size()
        image_size = self.original_pixmap.size()

        # 计算适应窗口的缩放比例
        scale_x = widget_size.width() / image_size.width()
        scale_y = widget_size.height() / image_size.height()
        self.scale_factor = min(scale_x, scale_y, 1.0)  # 不放大，只缩小

        self.image_offset = QPoint(0, 0)
        self.update_display()

    def resizeEvent(self, event):
        """控件大小改变事件"""
        super().resizeEvent(event)
        if self.original_pixmap:
            self.update_display()
