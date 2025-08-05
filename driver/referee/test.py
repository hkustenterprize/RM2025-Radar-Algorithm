import time
import threading
import numpy as np
from driver.referee.serial_comm import RefereeSerialManager, msg_cb_func
from driver.referee.serial_protocol import RefereeGenericMessage, MsgID
import traceback


# 测试回调函数
def test_callback(cmd_id, data):
    print(f"[TEST] Received - cmd_id: 0x{cmd_id:04x}, data_length: {len(data)}")
    print(f"[TEST] Data hex: {data.hex() if data else 'empty'}")

    # 根据不同的消息ID解析数据
    if cmd_id == MsgID.INTERACTIVE_DATA.value:
        if len(data) >= 12:  # 假设我们发送了3个uint32
            import struct

            values = struct.unpack("<III", data[:12])
            print(f"[TEST] Parsed values: {values}")


def test_message_packing():
    """测试消息打包功能"""
    print("\n=== Testing Message Packing ===")

    # 测试不同类型的数据
    test_msg = RefereeGenericMessage(
        np.uint16(MsgID.INTERACTIVE_DATA.value),
        np.uint32(12345),  # 整数
        np.float32(3.14159),  # 浮点数
        np.uint16(0xABCD),  # 16位整数
    )

    test_msg.pack()
    packed_data = test_msg.get_packed_buffer()

    print(f"Packed message length: {len(packed_data)} bytes")
    print(f"Packed message hex: {packed_data.hex()}")

    # 手动验证帧结构
    print("\n--- Frame Structure Analysis ---")
    print(f"SOF: 0x{packed_data[0]:02x} (should be 0xA5)")

    import struct

    data_length = struct.unpack("<H", packed_data[1:3])[0]
    print(f"Data length: {data_length}")

    seq = packed_data[3]
    print(f"Sequence: {seq}")

    crc8 = packed_data[4]
    print(f"Header CRC8: 0x{crc8:02x}")

    cmd_id = struct.unpack("<H", packed_data[5:7])[0]
    print(f"Command ID: 0x{cmd_id:04x}")

    return packed_data


def run_loopback_test(port1, port2):
    """运行回环测试"""
    print(f"\n=== Starting Loopback Test ===")
    print(f"TX Port: {port1}")
    print(f"RX Port: {port2}")

    # 创建发送和接收管理器
    tx_manager = RefereeSerialManager(port1, 115200)
    rx_manager = RefereeSerialManager(port2, 115200)

    # 绑定接收回调
    rx_manager.bind(MsgID.INTERACTIVE_DATA, test_callback)
    rx_manager.bind(MsgID.GAME_STATUS, test_callback)

    try:
        # 启动接收端
        rx_manager.start()
        time.sleep(0.5)  # 等待启动
        print("Here")

        # 启动发送端
        tx_manager.start()
        time.sleep(0.5)  # 等待启动

        print("\n=== Sending Test Messages ===")

        # 测试1: 发送基本消息
        print("\n[Test 1] Sending INTERACTIVE_DATA message...")
        tx_manager.tx(
            np.uint16(MsgID.INTERACTIVE_DATA.value),
            np.uint32(123456),
            np.float32(2.718),
            np.uint16(0x1234),
        )
        time.sleep(0.1)

        # 测试2: 发送游戏状态消息
        print("\n[Test 2] Sending GAME_STATUS message...")
        tx_manager.tx(
            np.uint16(MsgID.GAME_STATUS.value),
            np.uint8(1),  # game_type
            np.uint8(4),  # game_progress
            np.uint16(600),  # stage_remain_time
        )
        time.sleep(0.1)

        # 测试3: 连续发送多条消息
        print("\n[Test 3] Sending multiple messages...")
        for i in range(5):
            tx_manager.tx(
                np.uint16(MsgID.INTERACTIVE_DATA.value),
                np.uint32(i),
                np.float32(i * 1.5),
                np.uint16(0x1000 + i),
            )
            # time.sleep(0.05)

        # 等待处理完成
        time.sleep(1)

        # 显示统计信息
        print("\n=== RX Statistics ===")
        rx_manager.summarize()

    except Exception as e:
        print(f"Test error: {traceback.format_exc()}")
    finally:
        print("\n=== Cleaning Up ===")
        tx_manager.close()
        rx_manager.close()
        time.sleep(0.5)


def main():
    print("=== RoboMaster Serial Protocol Loopback Test ===")

    # 首先测试消息打包
    test_message_packing()

    # 提示用户创建虚拟串口
    print("\n" + "=" * 50)
    print("Please create virtual serial ports first:")
    print("sudo socat -d -d pty,raw,echo=0 pty,raw,echo=0")
    print("Then enter the two PTY paths below:")
    print("=" * 50)

    # 获取用户输入的串口路径
    port1 = input("Enter TX port (e.g., /dev/pts/1): ").strip()
    port2 = input("Enter RX port (e.g., /dev/pts/2): ").strip()

    if not port1 or not port2:
        print("Using default ports for testing...")
        port1 = "/dev/pts/1"
        port2 = "/dev/pts/2"

    run_loopback_test(port1, port2)


if __name__ == "__main__":
    main()
