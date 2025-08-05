import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from .interfactive_display import ZoomableDraggableLabel
import numpy as np
import cv2
import logging
from driver.referee.referee_comm import RefereeCommManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class RadarStationMainWindow(QMainWindow):
    """雷达站主界面"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RoboMaster 2025 雷达站控制界面")
        self.setGeometry(100, 100, 2048, 1534)

        # 初始化UI组件
        self.init_ui()
        self.show()  # Show the window first
        QApplication.processEvents()  # Process all pending events
        self.repaint()  # Force repaint
        QApplication.processEvents()  # Proce
        # 初始化数据更新定时器
        self.setup_timers()
        self.logger = logging.getLogger("RadarStation UI")
        import yaml
        with open("config/params.yaml", "r") as file:
            config = yaml.safe_load(file)
        
        if config["debug"]["inference_video"]:
            from driver.hik_camera.mock_hik import SimpleHikCamera
        else:
            from driver.hik_camera.hik import SimpleHikCamera
        self.camera = SimpleHikCamera._instance
        
        self.referee = RefereeCommManager._instance
        assert (
            self.referee is not None
        ), "Referee Comm Manager instance is not initialized."
        assert self.camera is not None, "Hik Camera instance is not initialized."
        self.camera.register_group("radar_station")
        self.is_setting_exposure = False

    def init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QHBoxLayout(central_widget)

        # 左侧区域：相机图像和雷达点云
        left_panel = self.create_image_panel()

        # 中间区域：场地投影和车辆坐标
        center_panel = self.create_field_projection_panel()

        # 右侧区域：控制面板和状态监控
        right_panel = self.create_control_panel()

        # 添加到主布局
        main_layout.addWidget(left_panel, 2)  # 左侧占2/5
        main_layout.addWidget(center_panel, 2)  # 中间占2/5
        main_layout.addWidget(right_panel, 1)  # 右侧占1/5

    # 修改原有的RadarStationMainWindow类中的create_image_panel方法
    def create_image_panel(self):
        """创建图像显示面板"""
        panel = QGroupBox("图像显示")
        layout = QVBoxLayout(panel)

        # 相机图像显示
        
        self.target_camera_size = QSize(819, 700)  # 2/5 of 2048px width
        self.target_tracker_size = QSize(819, 700)
        
        self.camera_label = QLabel("相机图像")
        self.camera_label.setMinimumSize(self.target_camera_size)
        self.camera_label.setStyleSheet("border: 1px solid black")
        self.camera_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.camera_label)

        # 雷达点云显示（可选）
        self.tracker_label = QLabel("Tracker")
        self.tracker_label.setMinimumSize(self.target_tracker_size)
        self.tracker_label.setStyleSheet("border: 1px solid black")
        self.tracker_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.tracker_label)
        


        # 相机控制按钮
        camera_controls = QHBoxLayout()

        # Basic control of camera
        basic_controls = QHBoxLayout()
      
        self.keypoint_btn = QPushButton("标关键点")

        basic_controls.addWidget(self.keypoint_btn)
        camera_controls.addLayout(basic_controls)

        # Exposure controls
        exposure_controls = QHBoxLayout()
        exposure_label = QLabel("曝光值:")
        exposure_label.setFixedWidth(50)
        exposure_controls.addWidget(exposure_label)
        # 曝光值输入框
        self.exposure_input = QSpinBox()
        self.exposure_input.setRange(1, 100000)  # 曝光时间范围，单位微秒
        self.exposure_input.setValue(10000)  # 默认值15ms
        self.exposure_input.setSuffix(" μs")
        self.exposure_input.setFixedWidth(100)
        exposure_controls.addWidget(self.exposure_input)
        # 曝光确认按钮
        self.exposure_confirm_btn = QPushButton("设置曝光")
        self.exposure_confirm_btn.setFixedWidth(80)
        self.exposure_confirm_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; }"
        )
        exposure_controls.addWidget(self.exposure_confirm_btn)
        # 当前曝光显示
        self.current_exposure_label = QLabel("当前: 10000μs")
        self.current_exposure_label.setStyleSheet("color: gray; font-size: 12px;")
        exposure_controls.addWidget(self.current_exposure_label)

        exposure_controls.addStretch()  # 添加弹性空间
        camera_controls.addLayout(exposure_controls)

        # Gain controls
        gain_controls = QHBoxLayout()
        gain_label = QLabel("增益值:")
        gain_label.setFixedWidth(50)
        gain_controls.addWidget(gain_label)
        # 增益值输入框
        self.gain_input = QSpinBox()
        self.gain_input.setRange(0, 100)  # 增益范围，单位dB
        self.gain_input.setValue(23)  # 默认值23dB
        self.gain_input.setSuffix(" dB")
        self.gain_input.setFixedWidth(100)
        gain_controls.addWidget(self.gain_input)
        # 增益确认按钮
        self.gain_confirm_btn = QPushButton("设置增益")
        self.gain_confirm_btn.setFixedWidth(80)
        self.gain_confirm_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; font-weight: bold; }"
        )
        gain_controls.addWidget(self.gain_confirm_btn)
        # 当前增益显示
        self.current_gain_label = QLabel("当前: 23dB")
        self.current_gain_label.setStyleSheet("color: gray; font-size: 12px;")
        gain_controls.addWidget(self.current_gain_label)

        gain_controls.addStretch()  # 添加弹性空间
        camera_controls.addLayout(gain_controls)

        layout.addLayout(camera_controls)

        # 连接信号槽
        self.exposure_confirm_btn.clicked.connect(self.on_exposure_confirm)
        self.exposure_input.valueChanged.connect(self.on_exposure_value_changed)
        self.gain_confirm_btn.clicked.connect(self.on_gain_confirm)
        self.gain_input.valueChanged.connect(self.on_gain_value_changed)
        self.keypoint_btn.clicked.connect(self.open_keypoint_calibration)  # 新增

        return panel

    def create_field_projection_panel(self):
        """创建场地投影面板"""
        panel = QGroupBox("场地实时投影")
        layout = QVBoxLayout(panel)

        # 场地投影显示区域
        self.field_view = FieldProjectionWidget()
        layout.addWidget(self.field_view)

        # 投影控制选项
        controls = QHBoxLayout()
        self.show_enemy_cb = QCheckBox("显示敌方车辆")
        self.show_ally_cb = QCheckBox("显示己方车辆")
        self.show_trajectory_cb = QCheckBox("显示轨迹")

        self.show_enemy_cb.setChecked(True)
        self.show_ally_cb.setChecked(True)

        controls.addWidget(self.show_enemy_cb)
        controls.addWidget(self.show_ally_cb)
        controls.addWidget(self.show_trajectory_cb)
        layout.addLayout(controls)

        self.show_enemy_cb.stateChanged.connect(self.update_display_settings)
        self.show_ally_cb.stateChanged.connect(self.update_display_settings)

        return panel

    def create_control_panel(self):
        """创建控制和状态面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 状态监控面板
        status_panel = self.create_status_panel()
        layout.addWidget(status_panel)

        # 校准控制面板
        calibration_panel = self.create_calibration_panel()
        layout.addWidget(calibration_panel)

        # 系统信息面板
        system_panel = self.create_system_panel()
        layout.addWidget(system_panel)

        return panel

    def update_display_settings(self):
        """更新显示设置"""
        self.field_view.show_enemy = self.show_enemy_cb.isChecked()
        self.field_view.show_ally = self.show_ally_cb.isChecked()
        self.field_view.update()

    def create_status_panel(self):
        """创建设备状态监控面板"""
        panel = QGroupBox("设备状态")
        layout = QVBoxLayout(panel)

        # 连接状态指示器
        self.referee_status = StatusIndicator("裁判系统rx")
        self.sentry_received = StatusIndicator("裁判系统tx")
        self.sentry_status = StatusIndicator("哨兵通讯")
        self.camera_status = StatusIndicator("相机连接")

        layout.addWidget(self.referee_status)
        layout.addWidget(self.sentry_received)
        layout.addWidget(self.sentry_status)
        layout.addWidget(self.camera_status)

        # 相机帧率显示
        self.fps_label = QLabel("相机帧率: 0 FPS")
        layout.addWidget(self.fps_label)

        # 当前阵营显示
        self.team_label = QLabel("当前阵营: 未知")
        self.team_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.team_label)

        return panel

    def create_calibration_panel(self):
        """创建校准控制面板"""
        panel = QGroupBox("校准控制")
        layout = QVBoxLayout(panel)

        # 校准按钮
        self.calibrate_btn = QPushButton("开始外参校准")
        self.calibrate_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }"
        )
        layout.addWidget(self.calibrate_btn)

        # 校准状态显示
        self.calibration_status = QLabel("雷达标定状态: 未校准")
        layout.addWidget(self.calibration_status)

        # 校准结果显示
        self.calibration_result = QTextEdit()
        self.calibration_result.setMaximumHeight(100)
        self.calibration_result.setPlaceholderText("校准结果将在此显示...")
        layout.addWidget(self.calibration_result)

        # 开始启动主进程
        self.start_tracking_btn = QPushButton("启动雷达站")
        self.start_tracking_btn.setEnabled(False)
        self.start_tracking_btn.clicked.connect(self.start_tracking)
        layout.addWidget(self.start_tracking_btn)
        self.main_event_loop = None
        self.calibrated_R = None
        self.calibrate_T = None

        return panel

    def create_system_panel(self):
        """创建系统信息面板"""
        panel = QGroupBox("系统状态")
        layout = QVBoxLayout(panel)

        # 双倍易伤状态
        self.double_damage_status = QLabel("双倍易伤: 未触发")
        self.double_damage_status.setStyleSheet("color: green; font-weight: bold;")
        layout.addWidget(self.double_damage_status)

        # 双倍易伤触发次数
        self.double_damage_count = QLabel("可触发次数:")
        self.double_damage_count.setStyleSheet("color: blue; font-weight: bold;")
        layout.addWidget(self.double_damage_count)

        # 机器人标记状态面板
        robot_mark_panel = QGroupBox("机器人标记状态")
        robot_mark_layout = QGridLayout(robot_mark_panel)
        
        # 创建机器人状态指示器字典
        self.robot_status_indicators = {}
        robot_types = [
            ("1Hero", "1英雄"),
            ("2Engineer", "2工程"),
            ("3Standard", "3步兵"),
            ("4Standard", "4步兵"),
            ("7Sentry", "7哨兵")
        ]
        
        # 按2列布局排列机器人状态
        for i, (robot_id, robot_name) in enumerate(robot_types):
            row = i // 2
            col = i % 2
            
            # 创建状态指示器容器
            robot_widget = QWidget()
            robot_layout = QHBoxLayout(robot_widget)
            robot_layout.setContentsMargins(5, 2, 5, 2)
            
            # 状态指示圆点
            status_dot = QLabel("●")
            status_dot.setFixedSize(20, 20)
            status_dot.setAlignment(Qt.AlignCenter)
            robot_layout.addWidget(status_dot)
            
            # 机器人名称标签
            name_label = QLabel(f"{robot_name}")
            name_label.setFixedWidth(80)
            robot_layout.addWidget(name_label)
            
            # 存储到字典中以便后续更新
            self.robot_status_indicators[robot_id] = {
                'dot': status_dot,
                'label': name_label,
                'is_marked': False
            }
            
            # 设置初始状态（未标记 = 红色）
            self.update_robot_indicator_status(robot_id, False)
            
            robot_mark_layout.addWidget(robot_widget, row, col)
        
        layout.addWidget(robot_mark_panel)

        # 系统日志
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setPlaceholderText("系统日志...")
        layout.addWidget(self.log_text)
        
        self.reset_referee_btn = QPushButton("重置裁判状态")
        self.reset_referee_btn.setStyleSheet(
            "QPushButton { background-color: #e53935; color: white; font-weight: bold; }"
        )
        self.reset_referee_btn.clicked.connect(self.on_reset_referee)
        layout.addWidget(self.reset_referee_btn)

        return panel

    def setup_timers(self):
        """设置定时器用于数据更新"""
        # 图像更新定时器
        self.image_timer = QTimer()
        self.image_timer.timeout.connect(self.update_images)
        self.image_timer.start(33)  # 30 FPS

        # 状态更新定时器
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(100)  # 1秒更新一次

    def start_tracking(self):
        # if self.calibrated_R is not None and self.calibrated_T is not None:
        from main_event_loop import MainEventLoop
        self.main_event_loop: MainEventLoop = MainEventLoop._instance
        self.main_event_loop.set_calibrated_param(
            self.calibrated_R, self.calibrated_T
        )

        self.main_event_loop.run()
        # 禁用按钮，避免重复启动
        self.start_tracking_btn.setEnabled(False)
        self.start_tracking_btn.setText("雷达站运行中...")
        # else:
        #     QMessageBox.warning(
        #         self,
        #         "校准未完成",
        #         "请先完成外参校准后再启动雷达站。",
        #     )

    def display_image(self, image, label_widget):
        """显示图像到指定的标签控件"""
        if image is not None:
            # 将OpenCV图像转换为QImage
            height, width, channel = image.shape
            bytes_per_line = channel * width
            q_image = QImage(
                image.data, width, height, bytes_per_line, QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(q_image)
            pixmap = pixmap.scaled(
                self.target_camera_size, Qt.IgnoreAspectRatio, Qt.FastTransformation
            )
            label_widget.setPixmap(pixmap)
        else:
            label_widget.setText("图像获取失败")

    def update_images(self):
        """更新图像显示"""
        image, time_stamp = self.camera.get_image_latest("radar_station", timeout=1)
        if image is not None:
            if (
                hasattr(self, "main_event_loop")
                and self.main_event_loop is not None
                and hasattr(self.main_event_loop, "detect_vis_img")
            ):
                if (detect_img := self.main_event_loop.detect_vis_img) is not None:
                    # 显示检测结果图像
                    self.display_image(detect_img, self.camera_label)
                else:
                    self.camera_label.setText("等待检测数据")
            else:
                # 显示相机图像
                self.display_image(image, self.camera_label)
        else:
            self.camera_label.setText("相机图像获取失败")

        if (
            hasattr(self, "main_event_loop")
            and self.main_event_loop is not None
            and hasattr(self.main_event_loop, "track_vis_img")
        ):
            if (track_img := self.main_event_loop.track_vis_img) is not None:
                self.display_image(track_img, self.tracker_label)
            else:
                self.tracker_label.setText("等待追踪数据")
        else:
            self.tracker_label.setText("Tracker还未启动")
        # # TODO: 实现相机图像获取和显示
        # # TODO: 实现雷达点云图像显示
        pass

    def update_robot_indicator_status(self, robot_id, is_marked):
        """更新机器人标记状态指示器"""
        if robot_id not in self.robot_status_indicators:
            return
        
        indicator = self.robot_status_indicators[robot_id]
        indicator['is_marked'] = is_marked
        
        if is_marked:
            # 已标记 - 绿色
            indicator['dot'].setStyleSheet("color: #4CAF50; font-size: 16px;")
            indicator['label'].setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            # 未标记 - 红色
            indicator['dot'].setStyleSheet("color: #F44336; font-size: 16px;")
            indicator['label'].setStyleSheet("color: #F44336; font-weight: normal;")

    def update_double_damage_status(self, is_active):
        """更新双倍易伤状态显示"""
        if is_active:
            self.double_damage_status.setText("双倍易伤: 已触发")
            self.double_damage_status.setStyleSheet("color: red; font-weight: bold; background-color: #ffeeee;")
        else:
            self.double_damage_status.setText("双倍易伤: 未触发")
            self.double_damage_status.setStyleSheet("color: green; font-weight: bold;")

    def update_double_damage_count(self, used_count, total_count):
        """更新双倍易伤触发次数显示"""
        remaining_count = total_count - used_count
        self.double_damage_count.setText(f"可触发次数: {used_count}/{total_count}")
        
        # 根据剩余次数设置颜色
        if remaining_count <= 0:
            self.double_damage_count.setStyleSheet("color: red; font-weight: bold")
        elif remaining_count <= 1:
            self.double_damage_count.setStyleSheet("color: orange; font-weight: bold")
        else:
            self.double_damage_count.setStyleSheet("color: blue; font-weight: bold")

    def get_robot_mark_status(self, robot_id):
        """
        获取指定机器人的标记状态
        
        Args:
            robot_id (str): 机器人ID，如 "1Hero", "2Engineer" 等
        
        Returns:
            bool: True表示已标记，False表示未标记
        """
        if hasattr(self, 'referee') and self.referee is not None:
            try:
                # 获取雷达标记进度消息
                mark_progress = self.referee.radar_mark_progress_msg
                
                # 根据机器人ID映射到对应的标记状态字段
                robot_mark_mapping = {
                    "1Hero": mark_progress.enemy_hero,
                    "2Engineer": mark_progress.enemy_engineer, 
                    "3Standard": mark_progress.enemy_standard_3,
                    "4Standard": mark_progress.enemy_standard_4,
                    "7Sentry": mark_progress.enemy_sentry
                }
                
                # 获取对应机器人的标记状态
                if robot_id in robot_mark_mapping:
                    return bool(robot_mark_mapping[robot_id])
                else:
                    # 如果robot_id不在映射中，返回False
                    return False
                    
            except AttributeError as e:
                # 如果访问字段失败，返回False
                return False
        
        # 默认返回未标记状态
        return False

    def update_robot_mark_status(self):
        """更新所有机器人标记状态"""
        if not hasattr(self, 'robot_status_indicators'):
            return
        
        # 遍历所有机器人类型，获取其标记状态
        for robot_id in self.robot_status_indicators.keys():
            # 获取标记状态
            is_marked = self.get_robot_mark_status(robot_id)
            # 更新显示
            self.update_robot_indicator_status(robot_id, is_marked)
    def update_status(self):
        """更新设备状态"""
        if hasattr(self.camera, "is_connected"):
            camera_connected = self.camera.is_connected()
            self.camera_status.update_status(camera_connected)
            if camera_connected:
                self.get_camera_fps()

        if hasattr(self.referee, "is_connected"):
            referee_connected = self.referee.is_connected()
            self.referee_status.update_status(referee_connected)

        if hasattr(self.referee, "is_sentry_connected"):
            sentry_connected = self.referee.is_sentry_connected
            self.sentry_status.update_status(sentry_connected)

        if hasattr(self.referee, "sentry_received_flag"):
            sentry_received = self.referee.sentry_received_flag
            self.sentry_received.update_status(sentry_received)

        from driver.referee.referee_comm import FACTION

        if hasattr(self.referee, "get_faction"):
            match self.referee.get_faction():
                case FACTION.BLUE:
                    self.team_label.setStyleSheet("color: blue; font-size: 12px")
                    self.team_label.setText("当前阵营：蓝方")
                case FACTION.RED:
                    self.team_label.setStyleSheet("color: red; font-size: 12px")
                    self.team_label.setText("当前阵营：红方")

        # 新增：更新双倍易伤状态
        if hasattr(self.referee, "is_double_vulnerability"):
            self.update_double_damage_status(self.referee.is_double_vulnerability)

        # 新增：更新双倍易伤触发次数
        if hasattr(self.referee, "double_vulnerability_count"):
            total_count = 2
            used_count = self.referee.double_vulnerability_count
            self.update_double_damage_count(used_count, total_count)
        
        # 新增：更新机器人标记状态
        self.update_robot_mark_status()

        if (
            hasattr(self, "main_event_loop")
            and self.main_event_loop is not None
            and hasattr(self.main_event_loop, "divisions_pos") and hasattr(self.main_event_loop, "faction") and hasattr(self.field_view, "update_vehicle_data")
        ):
            # print(self.main_event_loop.divisions_pos)
            self.field_view.update_vehicle_data(
                self.main_event_loop.divisions_pos, self.main_event_loop.faction
            )

    def get_camera_fps(self):
        self.fps_label.setText(f"相机帧率: {self.camera.get_fps():.2f} fps")
        
    def on_reset_referee(self):
        """重置裁判系统状态"""
        if hasattr(self.referee, "reset_double_trigger_state"):
            self.referee.reset_double_trigger_state()
            self.log_text.append("已重置裁判系统状态")
            print("Request count:", self.referee.request_count)
        else:
            QMessageBox.warning(self, "错误", "裁判系统不支持重置操作")

    def on_exposure_confirm(self):
        """确认设置曝光值"""
        try:
            exposure_value = self.exposure_input.value()

            # 设置相机曝光
            if hasattr(self.camera, "set_exposure"):
                success = self.camera.set_exposure(exposure_value)

                if success:
                    self.current_exposure_label.setText(f"当前: {exposure_value}μs")
                    self.current_exposure_label.setStyleSheet(
                        "color: green; font-size: 12px;"
                    )
                    self.logger.info(f"曝光值设置成功: {exposure_value}μs")

                    # 在系统日志中显示
                    self.log_text.append(
                        f"[{QDateTime.currentDateTime().toString('hh:mm:ss')}] 曝光值已设置为: {exposure_value}μs"
                    )
                    self.get_current_exposure()
                else:
                    self.current_exposure_label.setStyleSheet(
                        "color: red; font-size: 12px;"
                    )
                    self.logger.error(f"曝光值设置失败: {exposure_value}μs")

                    # 显示错误信息
                    QMessageBox.warning(
                        self,
                        "设置失败",
                        f"无法设置曝光值为 {exposure_value}μs\n请检查相机连接状态",
                    )
            else:
                self.logger.warning("相机不支持曝光设置")
                QMessageBox.information(
                    self, "功能不支持", "当前相机驱动不支持曝光设置功能"
                )

        except Exception as e:
            self.logger.error(f"设置曝光时发生错误: {e}")
            QMessageBox.critical(self, "错误", f"设置曝光时发生错误:\n{str(e)}")

    def on_exposure_value_changed(self, value):
        """曝光值改变时的回调"""
        # 实时显示即将设置的值
        self.current_exposure_label.setText(f"待设置: {value}μs")
        self.current_exposure_label.setStyleSheet("color: orange; font-size: 12px;")

    def on_gain_confirm(self):
        """确认设置增益值"""
        try:
            gain_value = self.gain_input.value()

            # 设置相机增益
            if hasattr(self.camera, "set_gain"):
                success = self.camera.set_gain(gain_value)

                if success:
                    self.current_gain_label.setText(f"当前: {gain_value}dB")
                    self.current_gain_label.setStyleSheet(
                        "color: green; font-size: 12px;"
                    )
                    self.logger.info(f"增益值设置成功: {gain_value}dB")

                    # 在系统日志中显示
                    self.log_text.append(
                        f"[{QDateTime.currentDateTime().toString('hh:mm:ss')}] 增益值已设置为: {gain_value}dB"
                    )
                    self.get_current_gain()
                else:
                    self.current_gain_label.setStyleSheet(
                        "color: red; font-size: 12px;"
                    )
                    self.logger.error(f"增益值设置失败: {gain_value}dB")

                    # 显示错误信息
                    QMessageBox.warning(
                        self,
                        "设置失败",
                        f"无法设置增益值为 {gain_value}dB\n请检查相机连接状态",
                    )
            else:
                self.logger.warning("相机不支持增益设置")
                QMessageBox.information(
                    self, "功能不支持", "当前相机驱动不支持增益设置功能"
                )

        except Exception as e:
            self.logger.error(f"设置增益时发生错误: {e}")
            QMessageBox.critical(self, "错误", f"设置增益时发生错误:\n{str(e)}")

    def on_gain_value_changed(self, value):
        """增益值改变时的回调"""
        # 实时显示即将设置的值
        self.current_gain_label.setText(f"待设置: {value}dB")
        self.current_gain_label.setStyleSheet("color: orange; font-size: 12px;")

    def open_keypoint_calibration(self):
        """打开关键点标定对话框"""
        try:
            # 获取当前视频帧作为背景
            current_image, _ = self.camera.get_image_latest("radar_station", timeout=1)

            if current_image is None:
                QMessageBox.warning(self, "错误", "无法获取当前视频帧，请检查相机连接")
                return

            # 确保图像是RGB格式
            if len(current_image.shape) == 3:
                background_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
            else:
                background_image = current_image

            # 创建并显示标定对话框
            calibration_dialog = KeypointCalibrationDialog(self, background_image)

            # if calibration_dialog.exec_() == QDialog.Accepted:
            # 获取标定结果
            # Wait for the calibration dialog to finish
            calibration_dialog.exec_()
            result = calibration_dialog.get_calibration_result()
            if result["success"]:
                # 在主界面显示校准结果
                self.display_calibration_result(result)
                self.log_text.append(
                    f"[{QDateTime.currentDateTime().toString('hh:mm:ss')}] "
                    f"外参校准成功，残差: {result['residual']:.4f}"
                )
            else:
                self.log_text.append(
                    f"[{QDateTime.currentDateTime().toString('hh:mm:ss')}] 外参校准失败"
                )

        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开标定界面时发生错误:\n{str(e)}")
            self.logger.error(f"Error opening calibration dialog: {e}")

    def display_calibration_result(self, result):
        """在主界面显示校准结果"""
        if result["success"]:
            # 更新校准状态
            self.calibration_status.setText("雷达标定状态: 已校准")
            self.calibration_status.setStyleSheet("color: green; font-weight: bold;")

            # 显示旋转矩阵和平移向量
            R = result["rotation_matrix"]
            t = result["translation_vector"]
            self.calibrated_R = R.copy()
            self.calibrated_T = t.copy()
            self.start_tracking_btn.setEnabled(True)

            residual = result["residual"]

            result_text = f"校准成功! 残差: {residual:.4f}\n"
            result_text += f"旋转矩阵 R:\n"
            result_text += f"[{R[0,0]:.4f} {R[0,1]:.4f} {R[0,2]:.4f}]\n"
            result_text += f"[{R[1,0]:.4f} {R[1,1]:.4f} {R[1,2]:.4f}]\n"
            result_text += f"[{R[2,0]:.4f} {R[2,1]:.4f} {R[2,2]:.4f}]\n\n"
            result_text += f"平移向量 t:\n[{t[0,0]:.4f} {t[1,0]:.4f} {t[2,0]:.4f}]"

            self.calibration_result.setText(result_text)
        else:
            self.calibration_status.setText("雷达标定状态: 校准失败")
            self.calibration_status.setStyleSheet("color: red; font-weight: bold;")
            self.calibration_result.setText("校准失败，请重新标定")

    def get_current_exposure(self):
        """获取当前相机曝光值"""
        try:
            if hasattr(self.camera, "get_exposure"):
                current_exposure = self.camera.get_exposure()
                if current_exposure is not None:
                    current_exposure = int(current_exposure)
                    self.exposure_input.setValue(current_exposure)
                    self.current_exposure_label.setText(f"当前: {current_exposure}μs")
                    self.current_exposure_label.setStyleSheet(
                        "color: green; font-size: 12px;"
                    )
                    return current_exposure
        except Exception as e:
            self.logger.error(f"获取当前曝光值失败: {e}")
        return None

    def get_current_gain(self):
        """获取当前相机增益值"""
        try:
            if hasattr(self.camera, "gain"):
                current_gain = self.camera.gain
                if current_gain is not None:
                    current_gain = int(current_gain)
                    self.gain_input.setValue(current_gain)
                    self.current_gain_label.setText(f"当前: {current_gain}dB")
                    self.current_gain_label.setStyleSheet(
                        "color: green; font-size: 12px;"
                    )
                    return current_gain
        except Exception as e:
            self.logger.error(f"获取当前增益值失败: {e}")
        return None


class FieldProjectionWidget(QWidget):
    """场地投影显示组件"""

    def __init__(self):
        super().__init__()
        self.field_image = cv2.imread("./field/field_image.png")
        if self.field_image is not None:
            self.field_image = cv2.rotate(
                self.field_image, cv2.ROTATE_90_COUNTERCLOCKWISE
            )
        self.setMinimumSize(500, 400)

        self.vehicles = []  # 车辆位置数据
        self.show_enemy = True
        self.show_ally = True

        self.faction = None

    def paintEvent(self, event):
        """绘制场地和车辆位置"""
        with QPainter(self) as painter:
            painter.setRenderHint(QPainter.Antialiasing)

            should_flip = self.should_flip_field()

            if should_flip:
                # 第一阶段：翻转绘制场地和车辆
                painter.translate(self.width() / 2, self.height() / 2)
                painter.rotate(180)
                painter.translate(-self.width() / 2, -self.height() / 2)

                # 绘制场地
                self.draw_field(painter)
                # 绘制车辆（不包含文本）
                self.draw_vehicles_without_text(painter)

                # 第二阶段：重置变换，绘制正向文本
                painter.resetTransform()
                painter.setRenderHint(QPainter.Antialiasing)

                # 绘制正向文本
                self.draw_vehicles_text_only(painter, flipped=True)
            else:
                # 正常绘制（红方视角）
                self.draw_field(painter)
                self.draw_vehicles(painter)

    def draw_vehicles_without_text(self, painter):
        """绘制车辆位置（不包含文本）"""
        if not hasattr(self, "vehicles") or not self.vehicles:
            return

        # 获取widget和场地图片的实际显示尺寸
        widget_width = self.width()
        widget_height = self.height()

        # 场地坐标系范围：右下角(0,0)，左上角(1500,2800)
        field_width_cm = 1500  # x轴方向(宽度)
        field_height_cm = 2800  # y轴方向(高度)

        # 计算显示区域参数
        if hasattr(self, "field_image") and self.field_image is not None:
            field_img_height, field_img_width = self.field_image.shape[:2]
            img_aspect = field_img_width / field_img_height
            widget_aspect = widget_width / widget_height

            if img_aspect > widget_aspect:
                display_width = widget_width
                display_height = widget_width / img_aspect
                offset_x = 0
                offset_y = (widget_height - display_height) // 2
            else:
                display_height = widget_height
                display_width = widget_height * img_aspect
                offset_x = (widget_width - display_width) // 2
                offset_y = 0
        else:
            display_width = widget_width
            display_height = widget_height
            offset_x = 0
            offset_y = 0

        for class_id, pos in enumerate(self.vehicles):
            if not pos.is_valid:
                continue

            # 检查显示控制
            if class_id < 5 and not self.show_ally:  # 蓝方
                continue
            if class_id >= 5 and not self.show_enemy:  # 红方
                continue

            # 坐标转换
            pos_x = max(0, min(field_height_cm, pos.x))
            pos_y = max(0, min(field_width_cm, pos.y))

            x_widget = int(
                offset_x + (field_width_cm - pos_y) / field_width_cm * display_width
            )
            y_widget = int(
                offset_y + (field_height_cm - pos_x) / field_height_cm * display_height
            )

            x_widget = max(int(offset_x), min(int(offset_x + display_width), x_widget))
            y_widget = max(int(offset_y), min(int(offset_y + display_height), y_widget))

            # 确定颜色
            if class_id < 5:  # 蓝方
                color = QColor(0, 100, 255)
            else:  # 红方
                color = QColor(255, 50, 50)

            # 绘制机器人位置
            painter.setBrush(color)
            painter.setPen(QPen(color, 2))

            # 绘制机器人圆点
            robot_radius = 8
            painter.drawEllipse(
                x_widget - robot_radius,
                y_widget - robot_radius,
                robot_radius * 2,
                robot_radius * 2,
            )

            # 绘制边框
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(
                x_widget - robot_radius,
                y_widget - robot_radius,
                robot_radius * 2,
                robot_radius * 2,
            )

    def draw_vehicles_text_only(self, painter, flipped=False):
        """仅绘制车辆文本（正向显示）"""
        if not hasattr(self, "vehicles") or not self.vehicles:
            return

        # 获取widget和场地图片的实际显示尺寸
        widget_width = self.width()
        widget_height = self.height()

        # 场地坐标系范围
        field_width_cm = 1500
        field_height_cm = 2800

        # 计算显示区域参数（与draw_vehicles_without_text保持一致）
        if hasattr(self, "field_image") and self.field_image is not None:
            field_img_height, field_img_width = self.field_image.shape[:2]
            img_aspect = field_img_width / field_img_height
            widget_aspect = widget_width / widget_height

            if img_aspect > widget_aspect:
                display_width = widget_width
                display_height = widget_width / img_aspect
                offset_x = 0
                offset_y = (widget_height - display_height) // 2
            else:
                display_height = widget_height
                display_width = widget_height * img_aspect
                offset_x = (widget_width - display_width) // 2
                offset_y = 0
        else:
            display_width = widget_width
            display_height = widget_height
            offset_x = 0
            offset_y = 0

        for class_id, pos in enumerate(self.vehicles):
            if not pos.is_valid:
                continue

            # 检查显示控制
            if class_id < 5 and not self.show_ally:
                continue
            if class_id >= 5 and not self.show_enemy:
                continue

            # 坐标转换
            pos_x = max(0, min(field_height_cm, pos.x))
            pos_y = max(0, min(field_width_cm, pos.y))

            if flipped:
                # 翻转后的坐标计算（因为场地已经翻转了180度）
                # 需要计算翻转前的原始显示位置，然后映射到翻转后的实际位置
                x_widget_original = int(
                    offset_x + (field_width_cm - pos_y) / field_width_cm * display_width
                )
                y_widget_original = int(
                    offset_y
                    + (field_height_cm - pos_x) / field_height_cm * display_height
                )

                # 翻转后的位置：相对于widget中心点对称
                x_widget = widget_width - x_widget_original
                y_widget = widget_height - y_widget_original
            else:
                x_widget = int(
                    offset_x + (field_width_cm - pos_y) / field_width_cm * display_width
                )
                y_widget = int(
                    offset_y
                    + (field_height_cm - pos_x) / field_height_cm * display_height
                )

            x_widget = max(int(offset_x), min(int(offset_x + display_width), x_widget))
            y_widget = max(int(offset_y), min(int(offset_y + display_height), y_widget))

            # 确定标签和颜色
            if class_id < 5:  # 蓝方
                text_color = QColor(0, 100, 255)
                robot_names = ["B1", "B2", "B3", "B4", "B7"]
                label = robot_names[class_id]
            else:  # 红方
                text_color = QColor(255, 50, 50)
                robot_names = ["R1", "R2", "R3", "R4", "R7"]
                label = robot_names[class_id - 5]

            # 准备显示文本
            coord_text = f"{label}\n({pos.x:.0f}, {pos.y:.0f})"

            # 设置字体
            font = painter.font()
            font.setPointSize(10)
            font.setBold(True)
            painter.setFont(font)

            # 获取文本尺寸
            metrics = painter.fontMetrics()
            text_lines = coord_text.split("\n")
            line_height = metrics.height()
            max_width = max(metrics.width(line) for line in text_lines)

            # 计算文本位置（机器人圆点右侧）
            robot_radius = 8
            text_x = x_widget + robot_radius + 5
            text_y = y_widget - robot_radius

            # 绘制背景（半透明白色）
            background_rect = QRect(
                text_x - 2,
                text_y - line_height,
                max_width + 4,
                line_height * len(text_lines) + 4,
            )
            painter.fillRect(background_rect, QColor(255, 255, 255, 180))

            # 绘制文本
            painter.setPen(QPen(text_color, 2))

            # 绘制每一行文本
            for i, line in enumerate(text_lines):
                painter.drawText(text_x, text_y + i * line_height, line)

    def update_vehicle_data(self, divisions_pos, faction):
        self.vehicles = divisions_pos
        self.faction = faction
        self.update()  # 触发重绘

    def should_flip_field(self):
        from driver.referee.referee_comm import FACTION

        if self.faction is None:
            return False
        elif self.faction == FACTION.BLUE:
            return True
        elif self.faction == FACTION.RED:
            return False
        elif self.faction == FACTION.UNKONWN:
            return False

    def draw_field(self, painter):
        if hasattr(self, "field_image") and self.field_image is not None:
            # 转换为QImage
            height, width, channel = self.field_image.shape
            bytes_per_line = channel * width
            q_image = QImage(
                self.field_image.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_BGR888,
            )

            # 转换为QPixmap
            pixmap = QPixmap.fromImage(q_image)

            # 按比例缩放以适应widget大小
            scaled_pixmap = pixmap.scaled(
                self.size(), Qt.IgnoreAspectRatio, Qt.FastTransformation
            )

            # 居中绘制
            x = (self.width() - scaled_pixmap.width()) // 2
            y = (self.height() - scaled_pixmap.height()) // 2
            painter.drawPixmap(x, y, scaled_pixmap)

    def draw_vehicles(self, painter):
        """绘制车辆位置"""

        def paint_text():
            for i, line in enumerate(text_lines):
                painter.drawText(text_x, text_y + i * line_height, line)

        if not hasattr(self, "vehicles") or not self.vehicles:
            return

        # 获取widget和场地图片的实际显示尺寸
        widget_width = self.width()
        widget_height = self.height()

        # 场地坐标系范围：右下角(0,0)，左上角(1500,2800)
        # pos.x 对应场地的高度方向(0-2800)，pos.y 对应场地的宽度方向(0-1500)
        field_width_cm = 1500  # x轴方向(宽度)
        field_height_cm = 2800  # y轴方向(高度)

        # 如果有场地图片，需要考虑图片的实际显示区域
        if hasattr(self, "field_image") and self.field_image is not None:
            # 获取场地图片的显示区域（居中显示后的实际区域）
            field_img_height, field_img_width = self.field_image.shape[:2]

            # 计算缩放后的图像尺寸（保持宽高比）
            img_aspect = field_img_width / field_img_height
            widget_aspect = widget_width / widget_height

            if img_aspect > widget_aspect:
                # 图像更宽，以widget宽度为准
                display_width = widget_width
                display_height = widget_width / img_aspect
                offset_x = 0
                offset_y = (widget_height - display_height) // 2
            else:
                # 图像更高，以widget高度为准
                display_height = widget_height
                display_width = widget_height * img_aspect
                offset_x = (widget_width - display_width) // 2
                offset_y = 0
        else:
            # 没有背景图片，使用整个widget区域
            display_width = widget_width
            display_height = widget_height
            offset_x = 0
            offset_y = 0

        for class_id, pos in enumerate(self.vehicles):
            if not pos.is_valid:
                continue

            # 检查显示控制
            if class_id < 5 and not self.show_ally:  # 蓝方
                continue
            if class_id >= 5 and not self.show_enemy:  # 红方
                continue

            # 坐标范围限制 (clamp)
            # pos.x 对应场地的高度方向(0-2800)，pos.y 对应场地的宽度方向(0-1500)
            pos_x = max(0, min(field_height_cm, pos.x))  # pos.x 对应场地的高度方向
            pos_y = max(0, min(field_width_cm, pos.y))  # pos.y 对应场地的宽度方向

            # 坐标系转换：场地坐标 -> widget坐标
            # 场地坐标系：右下角(0,0)，左上角(1500,2800)
            # widget坐标系：左上角(0,0)，右下角(width,height)
            # pos.x 对应图像的 y 轴，pos.y 对应图像的 x 轴

            x_widget = int(
                offset_x + (field_width_cm - pos_y) / field_width_cm * display_width
            )
            y_widget = int(
                offset_y + (field_height_cm - pos_x) / field_height_cm * display_height
            )

            # 确保绘制坐标在显示区域内
            x_widget = max(int(offset_x), min(int(offset_x + display_width), x_widget))
            y_widget = max(int(offset_y), min(int(offset_y + display_height), y_widget))

            # 确定颜色和标签
            if class_id < 5:  # 蓝方 B1-B4, B7
                color = QColor(0, 100, 255)  # 蓝色
                text_color = QColor(0, 100, 255)  # 蓝色文字
                robot_names = ["B1", "B2", "B3", "B4", "B7"]
                label = robot_names[class_id]
            else:  # 红方 R1-R4, R7
                color = QColor(255, 50, 50)  # 红色
                text_color = QColor(255, 50, 50)  # 红色文字
                robot_names = ["R1", "R2", "R3", "R4", "R7"]
                label = robot_names[class_id - 5]

            # 绘制机器人位置
            painter.setBrush(color)
            painter.setPen(QPen(color, 2))

            # 绘制机器人圆点
            robot_radius = 8
            painter.drawEllipse(
                x_widget - robot_radius,
                y_widget - robot_radius,
                robot_radius * 2,
                robot_radius * 2,
            )

            # 绘制边框
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(
                x_widget - robot_radius,
                y_widget - robot_radius,
                robot_radius * 2,
                robot_radius * 2,
            )

            # 准备显示文本：机器人名称 + 坐标
            coord_text = f"{label}\n({pos.x:.0f}, {pos.y:.0f})"

            # 设置字体
            font = painter.font()
            font.setPointSize(10)
            font.setBold(True)
            painter.setFont(font)

            # 获取文本尺寸以便合理布局
            metrics = painter.fontMetrics()
            text_lines = coord_text.split("\n")
            line_height = metrics.height()
            max_width = max(metrics.width(line) for line in text_lines)

            # 计算文本位置（机器人圆点右侧）

            text_x = x_widget + robot_radius + 5
            text_y = y_widget - robot_radius

            # 绘制背景（半透明白色，提高可读性）
            background_rect = QRect(
                text_x - 2,
                text_y - line_height,
                max_width + 4,
                line_height * len(text_lines) + 4,
            )
            painter.fillRect(background_rect, QColor(255, 255, 255, 180))

            # 绘制文本 - 使用阵营颜色

            painter.setPen(QPen(text_color, 2))

            # 绘制每一行文本

            paint_text()


class StatusIndicator(QWidget):
    """设备状态指示器"""

    def __init__(self, device_name):
        super().__init__()
        self.device_name = device_name
        self.connected = False
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.indicator = QLabel("●")
        self.indicator.setFixedSize(20, 20)
        self.label = QLabel(self.device_name)

        layout.addWidget(self.indicator)
        layout.addWidget(self.label)
        layout.addStretch()

        self.update_status(False)

    def update_status(self, connected):
        """更新连接状态"""
        self.connected = connected
        if connected:
            self.indicator.setStyleSheet("color: green; font-size: 16px;")
            self.label.setText(f"{self.device_name}: 已连接")
        else:
            self.indicator.setStyleSheet("color: red; font-size: 16px;")
            self.label.setText(f"{self.device_name}: 未连接")

    def update_display_settings(self):
        """更新显示设置"""
        self.field_view.show_enemy = self.show_enemy_cb.isChecked()
        self.field_view.show_ally = self.show_ally_cb.isChecked()
        self.field_view.update()


class ClickableImageLabel(QLabel):
    """可点击的图像标签"""

    left_clicked = pyqtSignal(int, int)
    right_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setCursor(Qt.ArrowCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.left_clicked.emit(event.x(), event.y())
        elif event.button() == Qt.RightButton:
            self.right_clicked.emit()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """鼠标移动事件，显示坐标"""
        super().mouseMoveEvent(event)
        # 可以在这里添加实时坐标显示功能

    def enterEvent(self, event):
        """鼠标进入事件"""
        self.setCursor(Qt.PointCursor)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """鼠标离开事件"""
        self.setCursor(Qt.ArrowCursor)
        super().leaveEvent(event)


class CalibrationGuideDialog(QDialog):
    """标定示意图对话框 - 支持缩放和拖拽"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("标定示意图")
        self.setModal(True)
        self.resize(1200, 900)

        self.setWindowFlags(
            Qt.Window  # 独立窗口
            | Qt.WindowTitleHint  # 显示标题栏
            | Qt.WindowSystemMenuHint  # 显示系统菜单
            | Qt.WindowMinMaxButtonsHint  # 显示最小化/最大化按钮
            | Qt.WindowCloseButtonHint  # 显示关闭按钮
        )

        self.init_ui()

    def init_ui(self):
        """初始化示意图界面"""
        layout = QVBoxLayout(self)

        # 标题
        title_label = QLabel("关键点标定示意图")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 15px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # 图像显示和控制区域
        image_controls_layout = QVBoxLayout()

        # 图像控制按钮
        controls_layout = QHBoxLayout()

        self.reset_btn = QPushButton("重置视图")
        self.reset_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 5px 10px; }"
        )
        self.reset_btn.clicked.connect(self.reset_image_view)

        self.fit_btn = QPushButton("适应窗口")
        self.fit_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 5px 10px; }"
        )
        self.fit_btn.clicked.connect(self.fit_image_to_window)

        # 缩放控制
        zoom_label = QLabel("缩放控制:")
        self.zoom_in_btn = QPushButton("放大 (+)")
        self.zoom_out_btn = QPushButton("缩小 (-)")

        self.zoom_in_btn.setStyleSheet("QPushButton { padding: 5px 10px; }")
        self.zoom_out_btn.setStyleSheet("QPushButton { padding: 5px 10px; }")

        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)

        controls_layout.addWidget(self.reset_btn)
        controls_layout.addWidget(self.fit_btn)
        controls_layout.addStretch()
        controls_layout.addWidget(zoom_label)
        controls_layout.addWidget(self.zoom_out_btn)
        controls_layout.addWidget(self.zoom_in_btn)

        image_controls_layout.addLayout(controls_layout)

        # 示意图显示区域 - 使用支持缩放拖拽的控件
        self.guide_image_label = ZoomableDraggableLabel()
        self.guide_image_label.setMinimumSize(800, 500)

        # 加载并显示示意图
        self.load_guide_image()

        image_controls_layout.addWidget(self.guide_image_label, 1)  # 占据主要空间
        layout.addLayout(image_controls_layout)

        # 操作提示
        tips_label = QLabel("操作提示：鼠标滚轮缩放，左键拖拽移动图像")
        tips_label.setStyleSheet("color: #666; font-size: 12px; margin: 5px;")
        tips_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(tips_label)

        # 说明文字
        instruction_text = QTextEdit()
        instruction_text.setMaximumHeight(100)
        instruction_text.setReadOnly(True)
        instruction_text.setStyleSheet(
            "background-color: #f0f0f0; border: 1px solid #ccc; padding: 10px;"
        )

        guide_text = """标定关键点说明：
1. 按照图中标记的顺序（1→2→3→4→5→6）依次标记关键点  2. 确保标记的点位置准确，这直接影响校准精度
3. 关键点应选择在场地边界或明显特征处  4. 避免选择在阴影或模糊区域的点  5. 标记完成后点击"开始校准"进行PnP求解"""

        instruction_text.setText(guide_text)
        layout.addWidget(instruction_text)

        # 关闭按钮
        button_layout = QHBoxLayout()

        self.close_btn = QPushButton("关闭")
        self.close_btn.setStyleSheet(
            "QPushButton { background-color: #FF5722; color: white; font-weight: bold; padding: 10px 20px; }"
        )
        self.close_btn.clicked.connect(self.close)

        button_layout.addStretch()
        button_layout.addWidget(self.close_btn)
        button_layout.addStretch()

        layout.addLayout(button_layout)

    def reset_image_view(self):
        """重置图像视图"""
        self.guide_image_label.reset_view()

    def fit_image_to_window(self):
        """适应窗口大小"""
        self.guide_image_label.fit_to_window()

    def zoom_in(self):
        """放大图像"""
        current_scale = self.guide_image_label.scale_factor
        new_scale = min(self.guide_image_label.max_scale, current_scale + 0.2)
        self.guide_image_label.scale_factor = new_scale
        self.guide_image_label.update_display()

    def zoom_out(self):
        """缩小图像"""
        current_scale = self.guide_image_label.scale_factor
        new_scale = max(self.guide_image_label.min_scale, current_scale - 0.2)
        self.guide_image_label.scale_factor = new_scale
        self.guide_image_label.update_display()

    def load_guide_image(self):
        """加载标定示意图"""
        try:
            # 尝试加载外部示意图文件
            guide_image_path = "interface/calib_tutorial.jpg"
            import os

            if os.path.exists(guide_image_path):
                # 如果存在外部图片文件，加载它
                pixmap = QPixmap(guide_image_path)
                self.guide_image_label.setPixmap(pixmap)
            else:
                # 如果没有外部文件，生成一个示例示意图
                self.generate_example_guide_image()

        except Exception as e:
            # 如果加载失败，生成示例图
            self.generate_example_guide_image()

    def generate_example_guide_image(self):
        """生成示例标定示意图"""
        try:
            # 创建一个示例图像 - 增大分辨率以便缩放观察
            width, height = 1600, 1000  # 增大分辨率
            image = np.ones((height, width, 3), dtype=np.uint8) * 240  # 浅灰色背景

            # 绘制场地轮廓（矩形）
            field_margin = 100
            cv2.rectangle(
                image,
                (field_margin, field_margin),
                (width - field_margin, height - field_margin),
                (100, 100, 100),
                6,  # 增加线条粗细
            )

            # 绘制中线
            cv2.line(
                image,
                (width // 2, field_margin),
                (width // 2, height - field_margin),
                (150, 150, 150),
                4,
            )

            # 定义关键点位置（示例）
            key_points = [
                (field_margin, field_margin),  # 左上角
                (width - field_margin, field_margin),  # 右上角
                (width - field_margin, height - field_margin),  # 右下角
                (field_margin, height - field_margin),  # 左下角
                (width // 2, field_margin),  # 中上
                (width // 2, height - field_margin),  # 中下
            ]

            # 绘制关键点和标号 - 增大尺寸
            colors = [
                (0, 255, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (255, 128, 0),
                (128, 0, 255),
            ]

            for i, (x, y) in enumerate(key_points):
                color = colors[i % len(colors)]

                # 绘制关键点 - 增大半径
                cv2.circle(image, (x, y), 20, color, -1)
                cv2.circle(image, (x, y), 25, (0, 0, 0), 4)

                # 绘制标号 - 增大字体
                cv2.putText(
                    image,
                    str(i + 1),
                    (x - 15, y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255, 255, 255),
                    4,
                )
                cv2.putText(
                    image,
                    str(i + 1),
                    (x - 15, y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 0),
                    2,
                )

            # 添加标题 - 增大字体
            cv2.putText(
                image,
                "Calibration Keypoints Guide",
                (width // 2 - 360, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (50, 50, 50),
                4,
            )

            # 添加坐标轴指示 - 增大尺寸
            arrow_size = 100
            cv2.arrowedLine(
                image,
                (width - 300, height - 150),
                (width - 300 + arrow_size, height - 150),
                (0, 0, 255),
                6,
            )
            cv2.putText(
                image,
                "X",
                (width - 180, height - 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
            )

            cv2.arrowedLine(
                image,
                (width - 300, height - 150),
                (width - 300, height - 150 - arrow_size),
                (0, 255, 0),
                6,
            )
            cv2.putText(
                image,
                "Y",
                (width - 330, height - 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
            )

            # 转换为QPixmap并显示
            height, width, channel = image.shape
            bytes_per_line = channel * width
            q_image = QImage(
                image.data, width, height, bytes_per_line, QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(q_image)

            self.guide_image_label.setPixmap(pixmap)

        except Exception as e:
            # 如果生成失败，显示文字说明
            error_text = (
                "无法生成标定示意图\n\n"
                "请按以下顺序标记关键点：\n"
                "1. 场地左上角\n"
                "2. 场地右上角\n"
                "3. 场地右下角\n"
                "4. 场地左下角\n"
                "5. 场地中线上端\n"
                "6. 场地中线下端"
            )
            self.guide_image_label.setText(error_text)
            self.guide_image_label.setStyleSheet(
                "border: 2px solid #333; background-color: #f9f9f9; "
                "font-size: 14px; color: #333; padding: 20px;"
            )

    def keyPressEvent(self, event):
        """键盘事件处理"""
        if event.key() == Qt.Key_R:
            self.reset_image_view()
        elif event.key() == Qt.Key_F:
            self.fit_image_to_window()
        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            self.zoom_in()
        elif event.key() == Qt.Key_Minus:
            self.zoom_out()
        else:
            super().keyPressEvent(event)


class ZoomableDraggableImageLabel(ZoomableDraggableLabel):
    """支持缩放、拖拽和关键点标记的图像标签"""
    
    left_clicked = pyqtSignal(float, float)  # 发送原图坐标
    right_clicked = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.keypoints = []  # 存储关键点在原图中的坐标
        self.click_threshold = 5  # 点击阈值（像素），小于此值认为是点击而非拖拽
        
        # 拖拽相关变量
        self.mouse_press_pos = None
        self.total_drag_distance = 0
        
    def set_keypoints(self, keypoints):
        """设置关键点列表"""
        self.keypoints = keypoints
        self.update_display()
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        self.mouse_press_pos = event.pos()
        self.total_drag_distance = 0
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.mouse_press_pos is not None:
            # 计算拖拽距离
            current_distance = (event.pos() - self.mouse_press_pos).manhattanLength()
            self.total_drag_distance = max(self.total_drag_distance, current_distance)
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if self.mouse_press_pos is not None:
            # 判断是点击还是拖拽
            if self.total_drag_distance < self.click_threshold:
                # 认为是点击操作
                if event.button() == Qt.LeftButton:
                    # 将屏幕坐标转换为原图坐标
                    original_coords = self.screen_to_image_coords(event.pos())
                    if original_coords is not None:
                        self.left_clicked.emit(original_coords[0], original_coords[1])
                elif event.button() == Qt.RightButton:
                    self.right_clicked.emit()
        
        self.mouse_press_pos = None
        self.total_drag_distance = 0
        super().mouseReleaseEvent(event)
    
    def screen_to_image_coords(self, screen_pos):
        """将屏幕坐标转换为原图坐标"""
        if self.original_pixmap is None:
            return None
        
        # 获取控件尺寸
        widget_size = self.size()
        
        # 计算缩放后的图像尺寸
        scaled_size = self.original_pixmap.size() * self.scale_factor
        
        # 计算图像在控件中的实际位置
        if scaled_size.width() < widget_size.width():
            image_left = (widget_size.width() - scaled_size.width()) // 2
        else:
            image_left = self.image_offset.x()
        
        if scaled_size.height() < widget_size.height():
            image_top = (widget_size.height() - scaled_size.height()) // 2
        else:
            image_top = self.image_offset.y()
        
        # 转换为图像内坐标
        image_x = screen_pos.x() - image_left
        image_y = screen_pos.y() - image_top
        
        # 检查是否在图像范围内
        if (image_x < 0 or image_y < 0 or 
            image_x >= scaled_size.width() or image_y >= scaled_size.height()):
            return None
        
        # 转换为原图坐标
        original_x = image_x / self.scale_factor
        original_y = image_y / self.scale_factor
        
        return (original_x, original_y)
    
    def image_to_screen_coords(self, image_x, image_y):
        """将原图坐标转换为屏幕坐标"""
        if self.original_pixmap is None:
            return None
        
        # 获取控件尺寸
        widget_size = self.size()
        
        # 计算缩放后的图像尺寸
        scaled_size = self.original_pixmap.size() * self.scale_factor
        
        # 计算缩放后的坐标
        scaled_x = image_x * self.scale_factor
        scaled_y = image_y * self.scale_factor
        
        # 计算图像在控件中的实际位置
        if scaled_size.width() < widget_size.width():
            image_left = (widget_size.width() - scaled_size.width()) // 2
        else:
            image_left = self.image_offset.x()
        
        if scaled_size.height() < widget_size.height():
            image_top = (widget_size.height() - scaled_size.height()) // 2
        else:
            image_top = self.image_offset.y()
        
        # 转换为屏幕坐标
        screen_x = scaled_x + image_left
        screen_y = scaled_y + image_top
        
        return (screen_x, screen_y)
    
    def create_display_image(self):
        """创建用于显示的图像（覆盖父类方法以添加关键点绘制）"""
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
        painter.setRenderHint(QPainter.Antialiasing)
        painter.drawPixmap(image_rect, self.scaled_pixmap)
        
        # 绘制关键点
        self.draw_keypoints_on_display(painter)

        # 绘制缩放信息
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.setFont(QFont("Arial", 12))
        scale_text = f"缩放: {self.scale_factor:.1f}x"
        painter.drawText(10, 25, scale_text)
        
        # 绘制操作提示
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.setFont(QFont("Arial", 10))
        help_text = "滚轮缩放 | 左键拖拽移动 | 单击标记关键点 | 右键撤销"
        painter.drawText(10, widget_size.height() - 10, help_text)

        painter.end()

        # 显示最终图像
        super(ZoomableDraggableLabel, self).setPixmap(display_image)
    
    def draw_keypoints_on_display(self, painter):
        """在显示图像上绘制关键点"""
        if not self.keypoints:
            return
        
        # 定义关键点颜色
        colors = [
            QColor(0, 255, 0),      # 绿色
            QColor(0, 255, 255),    # 青色
            QColor(255, 0, 255),    # 洋红
            QColor(255, 255, 0),    # 黄色
            QColor(255, 128, 0),    # 橙色
            QColor(128, 0, 255),    # 紫色
        ]
        
        for i, (x, y) in enumerate(self.keypoints):
            # 转换为屏幕坐标
            screen_coords = self.image_to_screen_coords(x, y)
            if screen_coords is None:
                continue
            
            screen_x, screen_y = screen_coords
            color = colors[i % len(colors)]
            
            # 根据缩放调整关键点大小
            point_radius = max(4, int(8 * self.scale_factor))
            border_radius = max(6, int(12 * self.scale_factor))
            
            # 绘制关键点
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color, 2))
            painter.drawEllipse(int(screen_x - point_radius), int(screen_y - point_radius), 
                              point_radius * 2, point_radius * 2)
            
            # 绘制边框
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(255, 255, 255), 3))
            painter.drawEllipse(int(screen_x - border_radius), int(screen_y - border_radius), 
                              border_radius * 2, border_radius * 2)
            
            # 绘制点号
            font_size = max(8, int(12 * self.scale_factor))
            font = QFont("Arial", font_size, QFont.Bold)
            painter.setFont(font)
            
            # 白色背景文字
            painter.setPen(QPen(QColor(255, 255, 255), 3))
            painter.drawText(int(screen_x + border_radius + 5), int(screen_y - border_radius), str(i + 1))
            
            # 黑色前景文字
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.drawText(int(screen_x + border_radius + 5), int(screen_y - border_radius), str(i + 1))


class KeypointCalibrationDialog(QDialog):
    """关键点标定对话框 - 支持缩放和拖拽"""

    def __init__(self, parent=None, background_image=None):
        super().__init__(parent)
        self.setWindowTitle("关键点标定")
        self.setModal(True)
        self.resize(1800, 1200)

        self.setWindowFlags(
            Qt.Window
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowMinMaxButtonsHint
            | Qt.WindowCloseButtonHint
        )

        # 背景图像和标定相关数据
        self.background_image = background_image
        self.original_image = (
            background_image.copy() if background_image is not None else None
        )
        self.keypoints = []  # 存储关键点坐标 [(x, y), ...]

        # 标定结果
        self.calibration_success = False
        self.rotation_matrix = None
        self.translation_vector = None
        self.residual = None

        self.init_ui()

    def init_ui(self):
        """初始化标定界面"""
        layout = QVBoxLayout(self)

        # 标题和说明
        title_label = QLabel("相机外参标定 - 关键点标记（支持缩放拖拽）")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        instruction_label = QLabel(
            "操作说明：\n"
            "• 鼠标滚轮缩放图像\n"
            "• 左键拖拽移动图像\n"
            "• 单击左键标记关键点（拖拽时不会标记）\n"
            "• 右键点击撤销最后一个关键点\n"
            "• 至少标记4个关键点才能进行校准"
        )
        instruction_label.setStyleSheet(
            "color: #666666; margin: 10px; padding: 10px; border: 1px solid #ddd;"
        )
        layout.addWidget(instruction_label)

        # 主要内容区域
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # 左侧：图像显示区域
        image_panel = self.create_image_panel()
        main_layout.addWidget(image_panel, 3)

        # 右侧：控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)

        layout.addWidget(main_widget)

        # 底部按钮
        button_layout = QHBoxLayout()

        self.clear_btn = QPushButton("清除所有点")
        self.clear_btn.setStyleSheet(
            "QPushButton { background-color: #FF5722; color: white; font-weight: bold; padding: 8px; }"
        )
        self.clear_btn.clicked.connect(self.clear_all_points)

        self.reset_view_btn = QPushButton("重置视图")
        self.reset_view_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 8px; }"
        )
        self.reset_view_btn.clicked.connect(self.reset_image_view)

        self.fit_view_btn = QPushButton("适应窗口")
        self.fit_view_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }"
        )
        self.fit_view_btn.clicked.connect(self.fit_image_to_window)

        self.show_guide_btn = QPushButton("标定示意图")
        self.show_guide_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 8px; }"
        )
        self.show_guide_btn.clicked.connect(self.show_calibration_guide)

        self.calibrate_btn = QPushButton("开始校准")
        self.calibrate_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }"
        )
        self.calibrate_btn.clicked.connect(self.start_calibration)
        self.calibrate_btn.setEnabled(False)

        self.close_btn = QPushButton("关闭")
        self.close_btn.setStyleSheet(
            "QPushButton { background-color: #9E9E9E; color: white; font-weight: bold; padding: 8px; }"
        )
        self.close_btn.clicked.connect(self.close)

        button_layout.addWidget(self.clear_btn)
        button_layout.addWidget(self.reset_view_btn)
        button_layout.addWidget(self.fit_view_btn)
        button_layout.addWidget(self.show_guide_btn)
        button_layout.addWidget(self.calibrate_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)

        # 设置初始图像
        if self.background_image is not None:
            self.set_background_image()

    def create_image_panel(self):
        """创建图像显示面板"""
        panel = QGroupBox("标定图像")
        layout = QVBoxLayout(panel)

        # 创建支持缩放拖拽的图像标签
        self.image_label = ZoomableDraggableImageLabel()
        self.image_label.setMinimumSize(1200, 800)
        self.image_label.setStyleSheet(
            "border: 2px solid #333; background-color: #f0f0f0;"
        )

        # 连接信号
        self.image_label.left_clicked.connect(self.add_keypoint)
        self.image_label.right_clicked.connect(self.remove_last_keypoint)

        layout.addWidget(self.image_label)

        return panel

    def create_control_panel(self):
        """创建控制面板"""
        panel = QGroupBox("标定控制")
        layout = QVBoxLayout(panel)

        # 关键点列表
        keypoint_label = QLabel("已标记的关键点:")
        keypoint_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(keypoint_label)

        self.keypoint_list = QListWidget()
        self.keypoint_list.setMaximumHeight(200)
        layout.addWidget(self.keypoint_list)

        # 状态信息
        status_label = QLabel("标定状态:")
        status_label.setStyleSheet("font-weight: bold; margin-top: 20px;")
        layout.addWidget(status_label)

        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(150)
        self.status_text.setPlaceholderText("等待标记关键点...")
        self.status_text.setReadOnly(True)
        layout.addWidget(self.status_text)

        # 校准结果显示
        result_label = QLabel("校准结果:")
        result_label.setStyleSheet("font-weight: bold; margin-top: 20px;")
        layout.addWidget(result_label)

        self.result_text = QTextEdit()
        self.result_text.setMaximumHeight(200)
        self.result_text.setPlaceholderText("校准结果将在此显示...")
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)

        return panel

    def set_background_image(self):
        """设置背景图像"""
        if self.background_image is not None:
            # 转换为QImage和QPixmap
            height, width, channel = self.background_image.shape
            bytes_per_line = channel * width
            q_image = QImage(
                self.background_image.data, width, height, bytes_per_line, QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(q_image)
            
            # 设置到图像标签
            self.image_label.setPixmap(pixmap)

    def reset_image_view(self):
        """重置图像视图"""
        self.image_label.reset_view()

    def fit_image_to_window(self):
        """适应窗口大小"""
        self.image_label.fit_to_window()

    def add_keypoint(self, x, y):
        """添加关键点"""
        # 添加到关键点列表
        self.keypoints.append((x, y))
        
        # 更新图像标签的关键点
        self.image_label.set_keypoints(self.keypoints)

        # 更新显示
        self.update_keypoint_list()
        self.update_status()

        self.status_text.append(
            f"添加关键点 {len(self.keypoints)}: ({x:.1f}, {y:.1f})"
        )

    def remove_last_keypoint(self):
        """移除最后一个关键点"""
        if self.keypoints:
            removed_point = self.keypoints.pop()
            
            # 更新图像标签的关键点
            self.image_label.set_keypoints(self.keypoints)
            
            self.update_keypoint_list()
            self.update_status()
            self.status_text.append(
                f"移除关键点: ({removed_point[0]:.1f}, {removed_point[1]:.1f})"
            )

    def clear_all_points(self):
        """清除所有关键点"""
        self.keypoints.clear()
        
        # 更新图像标签的关键点
        self.image_label.set_keypoints(self.keypoints)
        
        self.update_keypoint_list()
        self.update_status()
        self.status_text.append("清除所有关键点")

    def update_keypoint_list(self):
        """更新关键点列表显示"""
        self.keypoint_list.clear()
        for i, (x, y) in enumerate(self.keypoints):
            self.keypoint_list.addItem(f"点 {i+1}: ({x:.1f}, {y:.1f})")

    def update_status(self):
        """更新状态"""
        num_points = len(self.keypoints)
        if num_points >= 4:
            self.calibrate_btn.setEnabled(True)
            self.calibrate_btn.setText(f"开始校准 ({num_points}个点)")
        else:
            self.calibrate_btn.setEnabled(False)
            self.calibrate_btn.setText(f"需要至少4个点 (当前{num_points}个)")

    def show_calibration_guide(self):
        """显示标定示意图对话框"""
        guide_dialog = CalibrationGuideDialog(self)
        guide_dialog.exec_()

    def start_calibration(self):
        """开始PnP校准"""
        if len(self.keypoints) < 4:
            QMessageBox.warning(self, "错误", "至少需要4个关键点才能进行校准！")
            return

        try:
            # 准备图像点（2D坐标）
            image_points = np.array(self.keypoints, dtype=np.float32)

            # 加载3D物体点
            try:
                path = "transform/keypoint_6.txt"
                object_points = np.loadtxt(path, dtype=np.float32)
            except:
                QMessageBox.warning(
                    self, "警告", f"未找到3D关键点文件{path}，使用默认数据"
                )
                return

            # 确保3D点数量足够
            if len(object_points) < len(image_points):
                QMessageBox.warning(
                    self,
                    "错误",
                    f"3D关键点数量不足！需要至少{len(image_points)}个，当前只有{len(object_points)}个",
                )
                return

            # 创建PnP求解器
            from transform.solvepnp import PnPSolver

            pnp_solver = PnPSolver.from_config("config/params.yaml")

            # 执行校准
            success, R, tvec, residual = pnp_solver.solve(
                object_points[: len(image_points)], image_points
            )

            if success:
                self.calibration_success = True
                self.rotation_matrix = R
                self.translation_vector = tvec
                self.residual = residual

                # 显示结果
                result_text = f"校准成功！\n\n"
                result_text += f"旋转矩阵 R:\n{R}\n\n"
                result_text += f"平移向量 t:\n{tvec.flatten()}\n\n"
                result_text += f"投影残差: {residual:.4f} 像素"

                self.result_text.setText(result_text)
                self.status_text.append(f"校准成功！残差: {residual:.4f}")

                QMessageBox.information(
                    self,
                    "校准成功",
                    f"外参校准成功完成！\n"
                    f"投影残差: {residual:.4f} 像素\n"
                    f"点击确定查看详细结果",
                )

            else:
                self.calibration_success = False
                self.result_text.setText("校准失败！\n请检查关键点标记是否正确")
                self.status_text.append("校准失败")

                QMessageBox.warning(
                    self,
                    "校准失败",
                    "外参校准失败！\n"
                    "请检查关键点标记是否正确，\n"
                    "确保点的顺序与3D模型匹配",
                )

        except Exception as e:
            self.calibration_success = False
            error_msg = f"校准过程中发生错误: {str(e)}"
            self.result_text.setText(error_msg)
            self.status_text.append(error_msg)
            QMessageBox.critical(self, "错误", f"校准过程中发生错误:\n{str(e)}")

    def get_calibration_result(self):
        """获取校准结果"""
        if self.rotation_matrix is None or self.translation_vector is None:
            return {
                "success": False,
                "rotation_matrix": None,
                "translation_vector": None,
                "residual": None,
                "keypoints": [],
            }
        return {
            "success": self.calibration_success,
            "rotation_matrix": self.rotation_matrix.copy(),
            "translation_vector": self.translation_vector.copy(),
            "residual": self.residual,
            "keypoints": self.keypoints,
        }

    def keyPressEvent(self, event):
        """键盘快捷键"""
        if event.key() == Qt.Key_R:
            self.reset_image_view()
        elif event.key() == Qt.Key_F:
            self.fit_image_to_window()
        elif event.key() == Qt.Key_C:
            self.clear_all_points()
        elif event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

def launch():
    """启动雷达站主界面"""
    print(QCoreApplication.libraryPaths())
    QCoreApplication.setLibraryPaths(
        [
            "/home/fallengold/miniconda3/envs/radar/lib/python3.10/site-packages/PyQt5/Qt5/plugins"
        ]
    )
    import os

    os.environ["QT_PLUGIN_PATH"] = (
        "/home/fallengold/miniconda3/envs/radar/lib/python3.10/site-packages/PyQt5/Qt5/plugins"
    )
    os.environ["QT_LOGGING_RULES"] = "qt5ct.debug=false"
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 设置应用样式
    window = RadarStationMainWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    launch()
