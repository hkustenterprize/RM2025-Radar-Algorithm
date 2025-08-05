import struct
import serial
from enum import Enum
from driver.referee.crc import Crc
import numpy as np
from abc import ABC, abstractmethod
import ctypes


# Command protocal
# ======================
# Start Bytes：uint8_t
# payloadLength : uint8_t
# id: uint8_t
# CRC16: uint16_t
# data field: x
# CRC16: uint16_t
# ======================
# Total size： 7 + x


class MsgID(Enum):
    """
    Message ID definition for RoboMaster protocol
    """

    # 比赛状态数据，固定以1Hz发送
    GAME_STATUS = 0x0001
    # 比赛结果数据，比赛结束后触发
    GAME_RESULT = 0x0002
    # 机器人血量数据，固定以3Hz发送
    ROBOT_HP = 0x0003

    # 场地事件数据，固定以1Hz发送
    FIELD_EVENT = 0x0101
    # 裁判系统警告数据，己方判负/判罚时触发发送，其余时间以1Hz频率发送
    REFEREE_WARNING = 0x0104
    # 飞镖发射相关数据，固定以1Hz发送
    LAUNCHER_DATA = 0x0105

    # 机器人性能体系数据，固定以10Hz频率发送
    ROBOT_DATA = 0x0201
    # 实时底盘缓冲能量和射击热量数据，固定以10Hz频率发送
    ROBOT_POWER = 0x0202
    # 机器人位置数据，固定以1Hz频率发送
    SELF_ROBOT_POS = 0x0203
    # 机器人增益和底盘能量数据，固定以3Hz频率发送
    CHASSIS_AND_GAIN = 0x0204
    # 伤害状态数据，伤害发生后发送
    HURT_DATA = 0x0206

    # 实时射击数据，弹丸发射后发送
    SHOOT_DATA = 0x0207
    # 允许发弹量，固定以10Hz频率发送
    BULLET_ALLOWED = 0x0208
    # 机器人RFID模块状态，固定以3Hz频率发送
    RFID_STATUS = 0x0209

    # 飞镖选手端指令数据，固定以3Hz频率发送
    LAUNCHER_CMD = 0x020A
    # 地面机器人位置数据，固定以1Hz频率发送
    SENTRY_POS_DATA = 0x020B
    # 雷达标记进度数据，固定以1Hz频率发送
    RADAR_MARK_PROGRESS = 0x020C
    # 哨兵自主决策信息同步，固定以1Hz频率发送
    SENTRY_DECISION_SYNC = 0x020D

    # 雷达自主决策信息同步，固定以1Hz频率发送
    RADAR_DECISION_SYNC = 0x020E

    # 机器人交互数据，发送方触发发送，频率上限为30Hz
    INTERACTIVE_DATA = 0x0301
    # 自定义控制器与机器人交互数据，发送方触发发送，频率上限为30Hz
    CLIENT_CUSTOMIZED_CONTROLLER_DATA = 0x0302

    # 选手端小地图交互数据，选手端触发发送
    CLIENT_INTERACTIVE_DATA = 0x0303
    # 键鼠遥控数据，固定30Hz频率发送
    KEYBAORD_MOUSE_DATA = 0x0304

    # 选手端小地图接收雷达数据，频率上限为5Hz
    CLIENT_RADAR_DATA = 0x0305
    # 自定义控制器与选手端交互数据，发送方触发发送，频率上限为30Hz
    CUSTOMIZED_CONTROLLER_DATA = 0x0306

    # 选手端小地图接收路径数据，频率上限为1Hz
    CLIENT_PATH_DATA = 0x0307
    # 选手端小地图接收机器人数据，频率上限为3Hz
    CLIENT_ROBOT_DATA = 0x0308
    # 自定义控制器接收机器人数据，频率上限为10Hz
    CUSTOMIZED_CONTROLLER_ROBOT_DATA = 0x0309


class SubCmdID(Enum):
    DELETE_LAYER = 0x0100
    ADD_1_PATTERN = 0x0101
    ADD_2_PATTERN = 0x0102
    ADD_5_PATTERN = 0x0103
    ADD_7_PATTERN = 0x0104
    ADD_TEXT = 0x0110
    SENTRY_DECISION = 0x0120
    RADAR_DECISION = 0x0121

    RADAR_2_SENTRY = 0x0233
    SENTRY_2_RADAR = 0x0222


class OBJECT_ID(Enum):
    R_HERO = 1
    R_ENGINEER = 2
    R_INFANTRY_3 = 3
    R_INFANTRY_4 = 4
    R_INFANTRY_5 = 5
    R_DRONE = 6
    R_SENTRY = 7
    R_LAUNCHER = 8
    R_RADAR = 9
    R_OUTPOST = 10
    R_BASE = 11

    B_HERO = 101
    B_ENGINEER = 102
    B_INFANTRY_3 = 103
    B_INFANTRY_4 = 104
    B_INFANTRY_5 = 105
    B_DRONE = 106
    B_SENTRY = 107
    B_LAUNCHER = 108
    B_RADAR = 109
    B_OUTPOST = 110
    B_BASE = 111

    R_HERO_CLIENT = 0x0101
    R_ENGINEER_CLIENT = 0x0102
    R_INFANTRY_3_CLIENT = 0x0103
    R_INFANTRY_4_CLIENT = 0x0104
    R_INFANTRY_5_CLIENT = 0x0105
    R_DRONE_CLIENT = 0x0106

    B_HERO_CLIENT = 0x0165
    B_ENGINEER_CLIENT = 0x0166
    B_INFANTRY_3_CLIENT = 0x0167
    B_INFANTRY_4_CLIENT = 0x0168
    B_INFANTRY_5_CLIENT = 0x0169
    B_DRONE_CLIENT = 0x016A

    SERVER = 0x8080


class BaseMsg(ABC):
    def __init__(self):
        self.packed_buffer = b""

    @abstractmethod
    def pack(self):
        """Pack the message into the binary format"""
        return self.packed_buffer

    def get_packed_buffer(self):
        return self.packed_buffer


class RefereeGenericMessage(BaseMsg):
    """
    RoboMaster protocol message format:
    Frame structure: frame_header(5-byte) + cmd_id(2-byte) + data(n-byte) + frame_tail(2-byte CRC16)
    Frame header: SOF(1-byte) + data_length(2-byte) + seq(1-byte) + CRC8(1-byte)
    """

    format_string = "BBBH B s H"
    SOF = 0xA5

    def __init__(self, command_id: np.uint16, *data_fields):
        super().__init__()
        assert isinstance(command_id, np.uint16), "command_id must be np.uint16"
        self.command_id = command_id
        self.seq: int = 0
        self.data_fields = data_fields

    def pack(self):
        # 1. Prepare packed payload data
        packed_data = b"".join(
            self._pack_one_field(field) for field in self.data_fields
        )

        # 2. Calculate data_length (cmd_id + data)
        data_length = len(packed_data)  # 2 bytes for cmd_id + data length

        # 3. Pack frame header (without CRC8)
        # header_without_crc = (
        #     struct.pack("B", self.SOF)
        #     + struct.pack("H", data_length)
        #     + struct.pack("B", self.seq)
        # )
        header_without_crc = struct.pack("<BBH", self.SOF, data_length, self.seq)

        # 4. Calculate CRC8 for header
        crc8_header = Crc.get_crc8_check_sum(header_without_crc)

        # 5. Complete frame header
        frame_header = header_without_crc + struct.pack("B", crc8_header)

        # 6. Pack cmd_id
        cmd_id_packed = struct.pack("H", int(self.command_id))

        # 7. Combine header + cmd_id + data
        message_without_tail = frame_header + cmd_id_packed + packed_data

        # 8. Calculate CRC16 for entire message
        crc16_tail = Crc.get_crc16_check_sum(message_without_tail)

        # 9. Pack final message
        self.packed_buffer = message_without_tail + struct.pack("H", crc16_tail)

        return self.packed_buffer

    def _pack_one_field(self, data):
        """
        Pack data fields based on their type.
        This method can be extended to handle more types as needed. Numpy types are recommended.
        """
        if isinstance(data, bytes):
            return data
        elif isinstance(data, np.uint8):  # uint8
            return struct.pack("B", int(data))
        elif isinstance(data, np.uint16):  # uint16
            return struct.pack("H", int(data))
        elif isinstance(data, np.uint32):  # uint32
            return struct.pack("I", int(data))
        elif isinstance(data, np.int8):  # int8
            return struct.pack("b", int(data))
        elif isinstance(data, np.int16):  # int16
            return struct.pack("h", int(data))
        elif isinstance(data, np.int32):  # int32
            return struct.pack("i", int(data))
        elif isinstance(data, np.float32):  # float32
            return struct.pack("f", float(data))
        else:
            raise TypeError(
                f"Unsupported data type: {type(data)}. Supported types are: uint8, uint16, uint32, int8, int16, int32, float32."
            )


class InteractiveMessage(RefereeGenericMessage):
    """
    Data field -> sub_cmd_id(2-byte) + sender_id(2-byte) + receiver_id(2-byte) + data(n-byte)
    Notice that n < 112 bytes
    """

    def __init__(
        self,
        sub_cmd_id: np.uint16,
        sender_id: np.uint16,
        receiver_id: np.uint16,
        *data_fields,
    ):
        assert isinstance(sub_cmd_id, np.uint16), "sub_cmd_id must be np.uint16"
        assert isinstance(sender_id, np.uint16), "sender_id must be np.uint16"
        assert isinstance(receiver_id, np.uint16), "receiver_id must be np.uint16"

        super().__init__(
            np.uint16(MsgID.INTERACTIVE_DATA.value),
            np.uint16(sub_cmd_id),
            np.uint16(sender_id),
            np.uint16(receiver_id),
            *data_fields,
        )
        self.sub_cmd_id = sub_cmd_id
        self.sender_id = sender_id
        self.receiver_id = receiver_id


class RadarSelfDecisionCmd(InteractiveMessage):

    def __init__(self, radar_cmd: np.uint8, is_blue: bool = True):
        assert isinstance(radar_cmd, np.uint8), "radar_cmd must be np.uint8"
        super().__init__(
            np.uint16(SubCmdID.RADAR_DECISION.value),
            np.uint16(OBJECT_ID.B_RADAR.value if is_blue else OBJECT_ID.R_RADAR.value),
            np.uint16(OBJECT_ID.SERVER.value),
            np.uint8(radar_cmd),
        )


class InterRobotMessage(InteractiveMessage):
    """
    Inter-robot communication message format.
    This class can be used to send messages to different robot, such as sentry or hero
    """

    def __init__(
        self, sub_cmd_id: np.uint16, receiver_id: np.uint16, is_blue=True, *data_fields
    ):
        assert isinstance(sub_cmd_id, np.uint16), "sub_cmd_id must be np.uint16"
        assert isinstance(receiver_id, np.uint16), "receiver_id must be np.uint16"

        super().__init__(
            sub_cmd_id,
            np.uint16(
                OBJECT_ID.B_RADAR.value if is_blue else OBJECT_ID.B_RADAR.value
            ),  # Sender is always the server
            receiver_id,
            *data_fields,
        )
        self.sub_cmd_id = sub_cmd_id
        self.receiver_id = receiver_id


class Sentry2RadarData(ctypes.Structure):
    """
    Sentry to Radar data structure.
    This structure is used to send data from sentry to radar.
    """

    _pack_ = 1
    _fields_ = [
        ("hero_x", ctypes.c_float),  # Friend Hero X position
        ("hero_y", ctypes.c_float),  # Friend Hero Y position
        ("engineer_x", ctypes.c_float),  # Friend Engineer X position
        ("engineer_y", ctypes.c_float),  # Friend Engineer Y position
        ("standard_3_x", ctypes.c_float),  # Friend Standard 3 X position
        ("standard_3_y", ctypes.c_float),  # Friend Standard 3 Y position
        ("standard_4_x", ctypes.c_float),  # Friend Standard 4 X position
        ("standard_4_y", ctypes.c_float),  # Friend Standard 4 Y position
        ("sentry_x", ctypes.c_float),  # Friend Sentry X position
        ("sentry_y", ctypes.c_float),  # Friend Sentry Y position
        ("flag", ctypes.c_uint8),  # Flags for received
    ]


class Radar2SentryData(ctypes.Structure):
    """
    Radar to Sentry data structure.
    This structure is used to send data from radar to sentry.
    """

    _pack_ = 1
    _fields_ = [
        ("hero_x", ctypes.c_float),  # Enemy Hero X position
        ("hero_y", ctypes.c_float),  # Enemy Hero Y position
        ("engineer_x", ctypes.c_float),  # Enemy Engineer X position
        ("engineer_y", ctypes.c_float),  # Enemy Engineer Y position
        ("standard_3_x", ctypes.c_float),  # Enemy Standard 3 X position
        ("standard_3_y", ctypes.c_float),  # Enemy Standard 3 Y position
        ("standard_4_x", ctypes.c_float),  # Enemy Standard 4 X position
        ("standard_4_y", ctypes.c_float),  # Enemy Standard 4 Y position
        ("sentry_x", ctypes.c_float),  # Enemy Sentry X position
        ("sentry_y", ctypes.c_float),  # Enemy Sentry Y position
        ("suggested_target", ctypes.c_uint8),
        ("flags", ctypes.c_uint16),  # Flags for target selection
    ]


class DartStatData(ctypes.Structure):
    """
    Dart status data structure.
    This structure is used to receive launcher status
    """

    _pack_ = 1
    _fields_ = [
        ("dart_remaining_time", ctypes.c_uint8),  # Remaining time for the dart
        ("recent_hit_target", ctypes.c_uint8, 3),  # Recent hit target (2 bits)
        ("accumulated_hit_count", ctypes.c_uint8, 3),  # Accumulated hit count (2 bits)
        (
            "selected_target",
            ctypes.c_uint8,
            2,
        ),  # Selected target (2 bits). Use to trigger the double vulnerability
        ("reserve", ctypes.c_uint8, 8),  # Reserved for future use, 8 bits
    ]


class RadarMarkProgressData(ctypes.Structure):
    """
    Mark progress data structure.
    """

    _pack_ = 1
    _fields_ = [
        ("enemy_hero", ctypes.c_uint8, 1),  # Enemy hero marked (1 bit)
        ("enemy_engineer", ctypes.c_uint8, 1),  # Enemy engineer marked
        ("enemy_standard_3", ctypes.c_uint8, 1),  # Enemy standard
        ("enemy_standard_4", ctypes.c_uint8, 1),  # Enemy standard 4 marked
        ("enemy_sentry", ctypes.c_uint8, 1),  # Enemy sentry marked
        ("reserve", ctypes.c_uint8, 3),  # Reserved bits for future use
    ]


class RadarInfoData(ctypes.Structure):
    """
    Radar Info data structure
    """

    _pack_ = 1
    _fields_ = [
        (
            "double_vulnerability_count",
            ctypes.c_uint8,
            2,
        ),  # Double vulnerability status (1 bit)
        (
            "is_double_vulnerability",
            ctypes.c_uint8,
            1,
        ),  # Is double vulnerability (1 bit)
        ("reserve", ctypes.c_uint8, 5),  # Reserved bits for future use
    ]


class RobotStatusData(ctypes.Structure):
    """
    Robot status data structure. This status is to retrieve the faction of the robot.
    """

    _pack_ = 1
    _fields_ = [
        ("robot_id", ctypes.c_uint8),  # Robot ID (1 byte)
        ("robot_level", ctypes.c_uint8),  # Robot level (1 byte)
        ("current_hp", ctypes.c_uint16),  # Current HP (2 byte)
        ("max_hp", ctypes.c_uint16),  # Max HP (2 byte)
        (
            "shooter_barrel_cooling_value",
            ctypes.c_uint16,
        ),  # Shooter barrel cooling value (2 byte)
        (
            "shooter_barrel_heat_limit",
            ctypes.c_uint16,
        ),  # Shooter barrel heat limit (2 byte)
        ("chassis_power_limit", ctypes.c_uint16),  # Chassis power limit (1 bit)
        ("power_management_gimbal_output", ctypes.c_uint8, 1),
        (
            "power_management_chassis_output",
            ctypes.c_uint8,
            1,
        ),  # Power management chassis output (1 bit)
        ("power_management_shooter_output", ctypes.c_uint8, 1),  # Power
        ("reserve", ctypes.c_uint8, 5),  # Reserved bits for future use
    ]


class Radar2ClientData(ctypes.Structure):
    """
    Radar to client data structure
    """

    _pack_ = 1
    _fields_ = [
        ("hero_x", ctypes.c_uint16),  # Enemy Hero X position
        ("hero_y", ctypes.c_uint16),  # Enemy Hero Y position
        ("engineer_x", ctypes.c_uint16),  # Enemy Engineer X position
        ("engineer_y", ctypes.c_uint16),  # Enemy Engineer Y position
        ("standard_3_x", ctypes.c_uint16),  # Enemy Standard 3 X position
        ("standard_3_y", ctypes.c_uint16),  # Enemy Standard 3 Y position
        ("standard_4_x", ctypes.c_uint16),  # Enemy Standard 4 X position
        ("standard_4_y", ctypes.c_uint16),  # Enemy Standard 4 Y position
        ("standard_5_x", ctypes.c_uint16),  # Enemy Standard 5 X position
        ("standard_5_y", ctypes.c_uint16),  # Enemy Standard 5 Y position
        ("sentry_x", ctypes.c_uint16),  # Enemy Sentry X position
        ("sentry_y", ctypes.c_uint16),  # Enemy Sentry Y position
    ]


class RadarDecisionData(ctypes.Structure):
    """
    Radar double vulnerability decision data structure
    """

    _pack_ = 1
    _fields_ = [
        ("radar_cmd", ctypes.c_uint8),  # Radar command (1 byte)
    ]


class StructureMessage(BaseMsg):
    STRUCT_CLASS = None

    def __init__(self, msg_id, **kwargs):
        if self.STRUCT_CLASS is None:
            raise NotImplementedError("struct class must be defined in subclass")

        super().__init__()
        self.struct_data = self.STRUCT_CLASS()
        self.msg_id = np.uint16(msg_id)

        for field, value in kwargs.items():
            if hasattr(self.struct_data, field):
                setattr(self.struct_data, field, value)
            else:
                raise ValueError(
                    f"Field {field} not found in {self.STRUCT_CLASS.__name__}"
                )

    def pack(self):
        msg = RefereeGenericMessage(self.msg_id, bytes(self.struct_data))
        msg.pack()
        self.packed_buffer = msg.get_packed_buffer()
        return self.packed_buffer

    def __getattr__(self, item):
        if hasattr(self.struct_data, item):
            return getattr(self.struct_data, item)
        raise AttributeError(f"{item} not found in {self.STRUCT_CLASS.__name__}")

    def __setattr__(self, name, value):
        if name in ["struct_data", "msg_id", "packed_buffer"]:
            super().__setattr__(name, value)
        elif hasattr(self, "struct_data") and hasattr(self.struct_data, name):
            setattr(self.struct_data, name, value)
        else:
            super().__setattr__(name, value)

    def __str__(self):
        text = ""
        if hasattr(self, "struct_data"):
            for field in self.STRUCT_CLASS._fields_:
                name = field[0]
                value = getattr(self.struct_data, name)
                text += f"{field}: {value}\n"
        return text

    @classmethod
    def from_bytes(cls, data: bytes):
        if cls.STRUCT_CLASS is None:
            raise NotImplementedError("struct class must be defined in subclass")

        if len(data) < ctypes.sizeof(cls.STRUCT_CLASS):
            raise ValueError(
                f"Data length {len(data)} is less than struct size {ctypes.sizeof(cls.STRUCT_CLASS)}"
            )

        instance = cls.__new__(cls)
        BaseMsg.__init__(instance)
        instance.struct_data = cls.STRUCT_CLASS.from_buffer_copy(data)
        instance.msg_id = cls._get_msg_id()
        return instance

    @classmethod
    def _get_msg_id(cls):
        """子类可以重写此方法返回对应的消息ID"""
        raise NotImplementedError("子类必须实现 _get_msg_id 方法")


class InteractiveStructMessage(BaseMsg):
    STRUCT_CLASS = None  # 子类需要定义
    SUB_CMD_ID = None  # 子类需要定义

    def __init__(self, sender_id, receiver_id, **kwargs):
        if self.STRUCT_CLASS is None or self.SUB_CMD_ID is None:
            raise NotImplementedError("子类必须定义 STRUCT_CLASS 和 SUB_CMD_ID")

        super().__init__()
        self.sub_cmd_id = self.SUB_CMD_ID
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.data = self.STRUCT_CLASS()

        for key, value in kwargs.items():
            if hasattr(self.data, key):
                setattr(self.data, key, value)

    def pack(self):
        msg = InteractiveMessage(
            np.uint16(self.sub_cmd_id),
            np.uint16(self.sender_id),
            np.uint16(self.receiver_id),
            bytes(self.data),
        )

        return msg.pack()

    def __getattr__(self, name):
        if hasattr(self.data, name):
            return getattr(self.data, name)
        else:
            return super().__getattr__(name)

    def __setattr__(self, name, value):
        if name in ["sub_cmd_id", "sender_id", "receiver_id", "data", "packed_buffer"]:
            super().__setattr__(name, value)
        elif hasattr(self, "data") and hasattr(self.data, name):
            setattr(self.data, name, value)
        else:
            super().__setattr__(name, value)

    @classmethod
    def from_bytes(cls, data_bytes: bytes):
        """从交互数据的字节创建消息，data_bytes应该是去掉前7字节(cmd_id + crc)的结构体数据"""
        if cls.STRUCT_CLASS is None:
            raise NotImplementedError("子类必须定义 STRUCT_CLASS")

        if len(data_bytes) < ctypes.sizeof(cls.STRUCT_CLASS):
            raise ValueError(
                f"Data length {len(data_bytes)} is less than struct size {ctypes.sizeof(cls.STRUCT_CLASS)}"
            )

        # 创建实例但不调用__init__
        instance = cls.__new__(cls)
        BaseMsg.__init__(instance)
        instance.sub_cmd_id = cls.SUB_CMD_ID
        instance.sender_id = struct.unpack("<H", data_bytes[2:4])[0]
        instance.receiver_id = struct.unpack("<H", data_bytes[4:6])[0]
        instance.data = cls.STRUCT_CLASS.from_buffer_copy(data_bytes[6:])
        return instance

    def __str__(self):
        text = ""
        if hasattr(self, "data"):
            for field in self.STRUCT_CLASS._fields_:
                name = field[0]
                value = getattr(self.data, name)
                text += f"{field}: {value}\n"
        return text


# 具体的消息类
class RobotStatusMessage(StructureMessage):
    STRUCT_CLASS = RobotStatusData

    def __init__(self, **kwargs):
        super().__init__(MsgID.ROBOT_DATA.value, **kwargs)

    @classmethod
    def _get_msg_id(cls):
        return np.uint16(MsgID.ROBOT_DATA.value)


class DartStatusMessage(StructureMessage):
    STRUCT_CLASS = DartStatData

    def __init__(self, **kwargs):
        super().__init__(MsgID.LAUNCHER_DATA.value, **kwargs)

    @classmethod
    def _get_msg_id(cls):
        return np.uint16(MsgID.LAUNCHER_DATA.value)


class RadarMarkMessage(StructureMessage):
    STRUCT_CLASS = RadarMarkProgressData

    def __init__(self, **kwargs):
        super().__init__(MsgID.RADAR_MARK_PROGRESS.value, **kwargs)

    @classmethod
    def _get_msg_id(cls):
        return np.uint16(MsgID.RADAR_MARK_PROGRESS.value)


class RadarInfoMessage(StructureMessage):
    STRUCT_CLASS = RadarInfoData

    def __init__(self, **kwargs):
        super().__init__(MsgID.RADAR_DECISION_SYNC.value, **kwargs)

    @classmethod
    def _get_msg_id(cls):
        return np.uint16(MsgID.RADAR_DECISION_SYNC.value)


class Radar2ClientMessage(StructureMessage):
    STRUCT_CLASS = Radar2ClientData

    def __init__(self, **kwargs):
        super().__init__(MsgID.CLIENT_RADAR_DATA.value, **kwargs)

    @classmethod
    def _get_msg_id(cls):
        return np.uint16(MsgID.CLIENT_RADAR_DATA.value)


class Sentry2RadarMessage(InteractiveStructMessage):
    STRUCT_CLASS = Sentry2RadarData
    SUB_CMD_ID = SubCmdID.SENTRY_DECISION.value

    def __init__(self, is_blue=True, **kwargs):
        sender_id = OBJECT_ID.B_SENTRY.value if is_blue else OBJECT_ID.R_SENTRY.value
        receiver_id = OBJECT_ID.B_RADAR.value if is_blue else OBJECT_ID.R_RADAR.value
        super().__init__(sender_id, receiver_id, **kwargs)


class Radar2SentryMessage(InteractiveStructMessage):
    STRUCT_CLASS = Radar2SentryData
    SUB_CMD_ID = SubCmdID.RADAR_2_SENTRY.value

    def __init__(self, is_blue=True, **kwargs):
        sender_id = OBJECT_ID.B_RADAR.value if is_blue else OBJECT_ID.R_RADAR.value
        receiver_id = OBJECT_ID.B_SENTRY.value if is_blue else OBJECT_ID.R_SENTRY.value
        super().__init__(sender_id, receiver_id, **kwargs)


class RadarDecisionMessage(InteractiveStructMessage):
    STRUCT_CLASS = RadarDecisionData
    SUB_CMD_ID = SubCmdID.RADAR_DECISION.value

    def __init__(self, is_blue=True, **kwargs):
        sender_id = OBJECT_ID.B_RADAR.value if is_blue else OBJECT_ID.R_RADAR.value
        receiver_id = OBJECT_ID.SERVER.value
        super().__init__(sender_id, receiver_id, **kwargs)


if __name__ == "__main__":

    sentry_msg = Sentry2RadarMessage(
        is_blue=True,
        hero_x=1.0,
        hero_y=2.0,
        engineer_x=3.0,
        engineer_y=4.0,
        standard_3_x=5.0,
        standard_3_y=6.0,
        standard_4_x=7.0,
        standard_4_y=8.0,
        sentry_x=9.0,
        sentry_y=10.0,  # 40
    )
    print(f"Hero position: ({sentry_msg.hero_x}, {sentry_msg.hero_y})")
    packed_data = sentry_msg.pack()
    print(f"Packed data length: {len(packed_data)} bytes")

    print(sentry_msg.sender_id)
    print(sentry_msg.receiver_id)
    print(sentry_msg.hero_x)
    print(sentry_msg.hero_y)
    print(sentry_msg.engineer_x)
    print(sentry_msg.engineer_y)
    print(sentry_msg.standard_3_x)
    print(sentry_msg.standard_3_y)
    print(sentry_msg.standard_4_x)
    print(sentry_msg.standard_4_y)
    print(sentry_msg.sentry_x)
    print(sentry_msg.sentry_y)

    # print(
    #     f"Pack data hex: {sentry_msg.pack()[7:].hex()}, length: {len(sentry_msg.pack())} bytes"
    # )

    sentry_msg_rx = Sentry2RadarMessage.from_bytes(packed_data[7:])
    print(sentry_msg_rx.sender_id)
    print(sentry_msg_rx.receiver_id)
    print(sentry_msg_rx.hero_x)
    print(sentry_msg_rx.hero_y)
    print(sentry_msg_rx.engineer_x)
    print(sentry_msg_rx.engineer_y)
    print(sentry_msg_rx.standard_3_x)
    print(sentry_msg_rx.standard_3_y)
    print(sentry_msg_rx.standard_4_x)
    print(sentry_msg_rx.standard_4_y)
    print(sentry_msg_rx.sentry_x)
    print(sentry_msg_rx.sentry_y)
    # print(
    #     f"Pack data rx hex: {sentry_msg_rx.pack()[7:].hex()}, length: {len(sentry_msg_rx.pack())} bytes"
    # )
    sentry_msg_rx.pack()

    robot_status_msg = RobotStatusMessage(
        robot_id=OBJECT_ID.R_HERO.value,
        robot_level=1,
        current_hp=1000,
        max_hp=2000,
        shooter_barrel_cooling_value=500,
        shooter_barrel_heat_limit=1000,
        chassis_power_limit=1500,
        power_management_gimbal_output=1,
        power_management_chassis_output=1,
        power_management_shooter_output=1,
        reserve=0,
    )
    packed_data = robot_status_msg.pack()
    print(
        f"Robot Status Packed data length: {len(packed_data)} bytes, data: {packed_data}"
    )

    robot_status_msg_rx = RobotStatusMessage.from_bytes(packed_data[7:])
    print(robot_status_msg_rx.struct_data.robot_id)  # 打印机器人ID
    print(robot_status_msg_rx.struct_data.robot_level)  # 打印机器人等级
    print(robot_status_msg_rx.struct_data.current_hp)  # 打印当前HP
    print(robot_status_msg_rx.struct_data.max_hp)  # 打印最大HP
    print(
        robot_status_msg_rx.struct_data.shooter_barrel_cooling_value
    )  # 打印射手炮管冷却值
    print(robot_status_msg_rx.struct_data.shooter_barrel_heat_limit)  # 打印

    packed_data = robot_status_msg_rx.pack()
    print(
        f"Robot Status RX Packed data length: {len(packed_data)} bytes, data: {packed_data}"
    )
