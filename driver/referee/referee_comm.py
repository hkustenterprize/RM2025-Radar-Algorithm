# referee_comm.py
# 管理与裁判系统的通信
# 1. 管理消息发送的种类以及发送频率
# 2. 自动接收裁判系统的消息
# 3. 管理裁判系统与哨兵的连接状态
# -*- coding: utf-8 -*-
from .serial_comm import RefereeSerialManager
from .serial_protocol import *
import threading
import time
from enum import Enum


class FACTION(Enum):
    RED = 0
    BLUE = 1
    UNKONWN = 2


class RadarTriggerState(Enum):
    IDLE = 0
    TRIGGERING = 1


class RefereeCommManager(RefereeSerialManager):
    _instance = None

    def __init__(self, port="/dev/ttyUSB0", baudrate=115200):
        if self.__class__._instance is not None:
            return self.__class__._instance
        super().__init__(port, baudrate, auto_scan=True)
        self.bind(MsgID.ROBOT_DATA.value, self.status_message_decode_func)
        self.bind(MsgID.INTERACTIVE_DATA.value, self.interactive_message_decode_func)
        self.bind(MsgID.LAUNCHER_DATA.value, self.dart_status_message_decode_func)
        self.bind(MsgID.RADAR_MARK_PROGRESS.value, self.radar_mark_progress_message_decode_func)
        self.bind(MsgID.RADAR_DECISION_SYNC.value, self.radar_info_message_decode_func)

        # Status flag
        self.is_sentry_connected = False
        self.sentry_disconnect_counter = 0
        self.faction = FACTION.UNKONWN
        self.sentry_received_flag = False

        self.__class__._instance = self

        # TX
        # 1. radar2sentrymsg
        self.radar2sentry_msg = Radar2SentryMessage(
            is_blue=False,
            hero_x=-8888,
            hero_y=-8888,
            engineer_x=-8888,
            engineer_y=-8888,
            standard_3_x=-8888,
            standard_3_y=-8888,
            standard_4_x=-8888,
            standard_4_y=-8888,
            sentry_x=-8888,
            sentry_y=-8888,  # 40
            suggested_target=0,  # No target suggested
            flags=0,  # No flags
        )

        # 2. radar2client_msg
        self.radar2client_msg = Radar2ClientMessage(
            hero_x=0,
            hero_y=0,
            engineer_x=0,
            engineer_y=0,
            standard_3_x=0,
            standard_3_y=0,
            standard_4_x=0,
            standard_4_y=0,
            standard_5_x=0,
            standard_5_y=0,
            sentry_x=0,
            sentry_y=0,  # 40
        )

        # 3. 哨兵坐标反馈
        self.sentry2radar_msg = Sentry2RadarMessage(is_blue=False)

        # 4. 飞镖请求易伤指令
        self.dart_info = DartStatusMessage(
            dart_remaining_time=0,
            recent_hit_target=0,
            accumulated_hit_count=0,
            selected_target=0,
            reserve=0,
        )

        self.target = 0
        self.last_target = 0
        self.target_3_counter = 0
        self.target_3_fixed = False

        # 5. 标记进度反馈
        self.radar_mark_progress_msg = RadarMarkMessage()

        # 6. 雷达双倍易伤标记
        self.radar_info_msg = RadarInfoMessage()
        self.double_vulnerability_count = 0
        self.is_double_vulnerability = 0
        self.request_count = 0
        self.trigger_state = RadarTriggerState.IDLE

    def pack_radar_decision_message(self) -> RadarDecisionMessage:
        """打包雷达决策信息"""
        return RadarDecisionMessage(
            is_blue=self.faction == FACTION.BLUE, radar_cmd=self.request_count
        )
    
    def reset_double_trigger_state(self):
        print("Before resetting: ", self.request_count)
        self.request_count = 0
        print("After resetting: ", self.request_count)
        self.trigger_state = RadarTriggerState.IDLE
        self.target = 0
        self.last_target = 0
        self.target_3_counter = 0
        self.target_3_fixed = False


    def get_faction(self):
        return self.faction

    def status_message_decode_func(self, cmd_id, data):
        if cmd_id == MsgID.ROBOT_DATA.value:
            message = RobotStatusMessage.from_bytes(data)
            self_id = message.robot_id
            if self_id < 100:
                self.faction = FACTION.RED
            else:
                self.faction = FACTION.BLUE
            # print(f"[STATUS] Robot Status: {message}")

    def dart_status_message_decode_func(self, cmd_id, data):
        if cmd_id == MsgID.LAUNCHER_DATA.value:
            self.dart_info = DartStatusMessage.from_bytes(data)
            self.target = self.dart_info.selected_target

            # print(f"[DART] Dart Status: {self.dart_info}")

    def radar_mark_progress_message_decode_func(self, cmd_id, data):
        if cmd_id == MsgID.RADAR_MARK_PROGRESS.value:
            self.radar_mark_progress_msg = RadarMarkMessage.from_bytes(data)
            # print(
            #     f"[RADAR MARK PROGRESS] Radar Mark Progress: {self.radar_mark_progress_msg}"
            # )

    def radar_info_message_decode_func(self, cmd_id, data):
        if cmd_id == MsgID.RADAR_DECISION_SYNC.value:
            self.radar_info_msg = RadarInfoMessage.from_bytes(data)
            self.is_double_vulnerability = self.radar_info_msg.is_double_vulnerability
            self.double_vulnerability_count = (
                self.radar_info_msg.double_vulnerability_count
            )
            # print(f"[RADAR INFO] Radar Info: {self.radar_info_msg}")

    def interactive_message_decode_func(self, cmd_id, data):
        if cmd_id == MsgID.INTERACTIVE_DATA.value:
            import struct

            sub_cmd_id = struct.unpack("<H", data[0:2])[0]
            match sub_cmd_id:
                case SubCmdID.SENTRY_2_RADAR.value:
                    # Handle Sentry to Radar message
                    self.is_sentry_connected = True
                    self.sentry_disconnect_counter = 0
                    self.sentry2radar_msg = Sentry2RadarMessage.from_bytes(data)
                    self.sentry_received_flag = bool(self.sentry2radar_msg.flag)
                    # print(f"[SENTRY2RADAR] Sentry to Radar Message: {message}")

    def start(self):
        """启动裁判系统通信管理器"""
        status = super().start()
        if not status:
            self.get_logger().warning(
                "Failed to start RefereeComm Main Handling Logic."
            )
            return False
        self.message_daemon_stop_event = threading.Event()
        self.message_daemon_thread = threading.Thread(
            target=self.message_daemon, daemon=True
        )
        self.message_daemon_thread.start()
        return True

    def close(self):
        """停止裁判系统通信管理器"""
        self.message_daemon_stop_event.set()
        super().close()

    def message_daemon(self):
        """消息处理守护线程"""
        tick = 0
        radar2xx_counter = 0
        last, now = time.time(), time.time()
        while not self.message_daemon_stop_event.is_set():
            self.sentry_disconnect_counter += 1
            # 检测哨兵的连接状态
            if self.sentry_disconnect_counter > 3:
                self.is_sentry_connected = False
                self.sentry_disconnect_counter = 0
                # self.get_logger().warning(
                #     "[RefereeCommLogic] Sentry disconnected, resetting connection status."
                # )

            # 发送雷达标记信息
            # 1 HZ -> Triggering double 
            # 1 HZ -> Sentry
            # 8 HZ -> Referee
            # if tick % 10 == 0:
            radar2xx_counter += 1
            if radar2xx_counter % 10 == 0:
                self.tx(self.radar2sentry_msg.pack())

            elif radar2xx_counter % 10 == 5:
                 # 状态机逻辑：只有从0/1/2到3才能触发一次
                if self.trigger_state == RadarTriggerState.IDLE:
                    # 检测是否由0/1/2变为0
                    if self.double_vulnerability_count > 0 and self.is_double_vulnerability == 0:
                        if (self.last_target in [0, 1, 2]) and (self.target == 3):
                            self.target_3_counter += 1
                            self.target_3_fixed = True
                        elif self.target in [0, 1, 2]:
                            # 状态回到0/1/2，允许下一次触发
                            self.target_3_counter = 0
                            self.target_3_fixed = False

                        # 满足条件，触发双倍易伤
                        if (
                            self.target_3_fixed
                            and self.target_3_counter >= 1  # 只允许一次
                        ):
                            self.get_logger().warning("[RefereeCommLogic] Triggering double vulnerability.")
                            self.request_count += 1
                            self.trigger_state = RadarTriggerState.TRIGGERING
                
                else:  # TRIGGERING 状态
                    self.tx(self.pack_radar_decision_message().pack())
                    if self.is_double_vulnerability == 1 or self.double_vulnerability_count == 0:
                        self.trigger_state = RadarTriggerState.IDLE
                        self.get_logger().warning("[RefereeCommLogic] Double vulnerability triggered, resetting trigger state.")
                        self.target_3_counter = 0
                        self.target_3_fixed = False

                # 每次循环都更新 last_target
                self.last_target = self.target
            else:
                self.tx(self.radar2client_msg.pack())

            time.sleep(0.08)


if __name__ == "__main__":
    # Example usage
    import rclpy

    port = "/dev/ttyUSB0"  # Replace with your actual port
    baudrate = 115200

    rclpy.init()

    referee_manager = RefereeCommManager(port, baudrate)
    if referee_manager.start():
        print("RefereeCommManager started successfully.")
    else:
        print("Failed to start RefereeCommManager.")

    try:
        while True:
            referee_manager.summarize()
            time.sleep(1)  # Keep the script running
    except KeyboardInterrupt:
        referee_manager.close()
        print("RefereeCommManager stopped.")
