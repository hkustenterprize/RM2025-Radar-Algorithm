from .serial_comm import RefereeSerialManager
from .serial_protocol import Sentry2RadarMessage, RobotStatusMessage
from .serial_protocol import Radar2SentryMessage
from .serial_protocol import MsgID, SubCmdID


def status_message_decode_func(cmd_id, data):

    if cmd_id == MsgID.ROBOT_DATA.value:
        message = RobotStatusMessage.from_bytes(data)
        print(f"[STATUS] Robot Status: {message}")


def sentry2radar_message_decode_func(cmd_id, data):
    if cmd_id == MsgID.INTERACTIVE_DATA.value:
        message = Sentry2RadarMessage.from_bytes(data)
        print(f"[SENTRY2RADAR] Sentry to Radar Message: {message}")


if __name__ == "__main__":
    serial_manager = RefereeSerialManager(port="/dev/ttyUSB0", baudrate=115200)
    serial_manager.bind(MsgID.ROBOT_DATA.value, status_message_decode_func)
    serial_manager.bind(MsgID.INTERACTIVE_DATA.value, sentry2radar_message_decode_func)
    serial_manager.start()
    import time

    while True:
        # serial_manager.summarize()
        sentry_msg = Radar2SentryMessage(
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
            suggested_target=1,
            flags=2,
        )
        serial_manager.summarize()
        serial_manager.tx(sentry_msg.pack())
        time.sleep(1.0)
