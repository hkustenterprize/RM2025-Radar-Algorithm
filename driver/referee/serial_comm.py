import serial
import struct
import collections
import time
import os
import threading
from driver.referee.crc import Crc
from driver.referee.serial_protocol import RefereeGenericMessage
from driver.referee.serial_protocol import MsgID
from driver.referee.usb import USBDeviceManager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from enum import Enum
import rclpy
from rclpy.node import Node

# Constants
RX_BUF_SIZE = 1024
MAX_PACKAGE_SIZE = 128
START_BYTE = 0xA5
CRC16_SIZE = 2
FRAME_HEADER_SIZE = 5

class SerialState(Enum):
    CLOSED = 0
    SCANNING = 3
    OPENNED = 1
    RECOVERING = 2

class PackageStatus:
    HEADER_INCOMPLETE = 1
    HEADER_CRC_ERROR = 2
    PAYLOAD_INCOMPLETE = 3
    CRC_ERROR = 4
    PACKAGE_COMPLETE = 5

class CircularBuffer:
    def __init__(self, size):
        self.buffer = collections.deque(maxlen=size)

    def enqueue_rear(self, data):
        self.buffer.extend(data)

    def enqueue_front(self, data):
        self.buffer.extendleft(reversed(data))

    def dequeue_front(self, count):
        data = bytearray()
        for _ in range(count):
            if self.buffer:
                data.append(self.buffer.popleft())
            else:
                break
        return data

    def available(self):
        return len(self.buffer)

class RefereeSerialManager(Node):
    def __init__(self, port=None, baudrate=115200, auto_scan=True):
        super().__init__("referee_serial_manager")
        self.initial_port = port
        self.baudrate = baudrate
        self.auto_scan = auto_scan
        self.current_port = None
        self.current_port_info = None

        self.state = SerialState.CLOSED
        self.error_counter = 0
        self.max_error_count = 10
        self.reconnect_interval = 2.0
        self.scan_interval = 2.0
        self.last_scan_time = 0

        self.rx_buffer = CircularBuffer(RX_BUF_SIZE)
        self.rx_frame_buffer = bytearray(FRAME_HEADER_SIZE + MAX_PACKAGE_SIZE + CRC16_SIZE)
        self.rx_count = {
            "Package_Received": 0,
            "Bytes_Received": 0,
            "Header_Incomplete": 0,
            "Header_CRC_Error": 0,
            "Payload_Incomplete": 0,
            "CRC_Error": 0,
            "Package_Complete": 0,
        }
        self.rx_rate = {"CRC_Error_Rate": 0}
        self.package_rx_count = {}
        self.cb_funcs = {}

        # Logging
        self.recording_save_root_dir = "/home/siyu/rm/data/mnt/ssd/referee_logs"
        self.tx_frame_count = 0
        self.rx_frame_count = 0
        self.lock = threading.Lock()
        self.log_file = None
        timestamp = self.get_formatted_time()
        self.log_file_path = os.path.join(self.recording_save_root_dir, f"referee_logs_{timestamp}.txt")
        Path(self.recording_save_root_dir).mkdir(parents=True, exist_ok=True)
        self.open_log_file()
        self.rx_task = None

    def get_formatted_time(self):
        """获取UTC+8时间戳"""
        tz = timezone(timedelta(hours=8))
        return datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    def open_log_file(self):
        """打开日志文件"""
        try:
            self.log_file = open(self.log_file_path, 'a')
            print(f"Opened log file: {self.log_file_path}")
        except Exception as e:
            print(f"Error opening log file: {e}")

    def close_log_file(self):
        """关闭日志文件"""
        try:
            if self.log_file:
                self.log_file.close()
                print(f"Closed log file: {self.log_file_path}")
        except Exception as e:
            print(f"Error closing log file: {e}")
        finally:
            self.log_file = None

    def _find_available_port(self):
        """查找可用串口"""
        if self.initial_port and os.path.exists(self.initial_port):
            if USBDeviceManager.test_serial_port(self.initial_port, self.baudrate):
                return {
                    "device": self.initial_port,
                    "description": "Specified Port",
                    "manufacturer": "Unknown",
                    "vid": None,
                    "pid": None,
                    "serial_number": None,
                }

        if self.auto_scan:
            print("Scanning for USB serial ports...")
            usb_ports = USBDeviceManager.scan_usb_serial_ports()
            print(f"Found {len(usb_ports)} USB serial ports:")
            for port_info in usb_ports:
                print(f"  - {port_info['device']}: {port_info['description']}")

            for port_info in usb_ports:
                port = port_info["device"]
                print(f"Testing port: {port}")
                if USBDeviceManager.test_serial_port(port, self.baudrate):
                    print(f"Port {port} is available")
                    return port_info
                else:
                    print(f"Port {port} test failed")

        return None

    def _open_port(self):
        """打开串口"""
        try:
            port_info = self._find_available_port()
            if not port_info:
                print("No available USB serial ports found")
                return False

            port = port_info["device"]
            self.ser = serial.Serial(port, self.baudrate, timeout=1)
            self.current_port = port
            self.current_port_info = port_info
            print(f"Port {port} opened successfully")
            return True

        except serial.SerialException as e:
            print(f"Failed to open port: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error opening port: {e}")
            return False

    def _close_port(self):
        """关闭串口"""
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
                print(f"Port {self.current_port} closed")
        except Exception as e:
            print(f"Error closing port: {e}")
        finally:
            self.ser = None
            self.current_port = None

    def is_connected(self):
        """检查连接状态"""
        return self.state == SerialState.OPENNED and self.ser and self.ser.is_open

    def start(self):
        """启动串口管理器"""
        if self.rx_task and self.rx_task.is_alive():
            print("Serial manager is already running")
            return False

        self.rx_task_stop_event, self.rx_task = self._create_thread()
        self.rx_task.daemon = True
        self.rx_task.start()
        print("Serial manager started")
        return True

    def close(self):
        """关闭串口管理器"""
        if self.rx_task_stop_event:
            self.rx_task_stop_event.set()

        if self.rx_task and self.rx_task.is_alive():
            self.rx_task.join(timeout=2.0)

        self._close_port()
        self.close_log_file()
        print("Serial manager stopped")

    def _create_thread(self):
        return threading.Event(), threading.Thread(target=self.rx_task_func)

    def rx_task_func(self):
        """接收任务主循环"""
        while not self.rx_task_stop_event.is_set():
            current_time = time.time()

            if self.state == SerialState.CLOSED or self.state == SerialState.SCANNING:
                if current_time - self.last_scan_time > self.scan_interval:
                    self.state = SerialState.SCANNING
                    print("Attempting to connect to serial device...")
                    if self._open_port():
                        self.state = SerialState.OPENNED
                        self.error_counter = 0
                        print("Successfully connected")
                    else:
                        print("Failed to connect, retrying...")
                        time.sleep(self.reconnect_interval)
                    self.last_scan_time = current_time
                else:
                    time.sleep(0.1)

            elif self.state == SerialState.OPENNED:
                try:
                    if self.ser.in_waiting > 0:
                        data = self.ser.read(self.ser.in_waiting)
                        self.rx_buffer.enqueue_rear(data)

                    status = PackageStatus.PACKAGE_COMPLETE
                    while self.rx_buffer.available() and (
                        status != PackageStatus.HEADER_INCOMPLETE
                        and status != PackageStatus.PAYLOAD_INCOMPLETE
                    ):
                        status = self.update_and_get_next_frame()
                        if status == PackageStatus.PACKAGE_COMPLETE:
                            data_length = struct.unpack("<H", self.rx_frame_buffer[1:3])[0]
                            cmd_id, data = self._get_cmd_id_and_data_from_buffer(data_length)
                            # 直接记录rx消息
                            with self.lock:
                                timestamp = self.get_formatted_time()
                                data_hex = data.hex()
                                log_entry = f"rx {timestamp} frame_{self.rx_frame_count:06d} cmd_id=0x{cmd_id:04x} data={data_hex}\n"
                                if self.log_file:
                                    self.log_file.write(log_entry)
                                    self.log_file.flush()
                                self.rx_frame_count += 1
                                # print(f"Logged rx message: frame_{self.rx_frame_count-1:06d}")

                            key_id = f"0x{cmd_id:04x}"
                            if key_id in self.cb_funcs:
                                for cb_func in self.cb_funcs[key_id]:
                                    try:
                                        cb_func(cmd_id, data)
                                    except Exception as e:
                                        print(f"Error in callback for cmd_id {key_id}: {e}")

                except serial.SerialException as e:
                    print(f"Serial read error: {e}")
                    self.error_counter += 1
                    self._handle_connection_error()

                except Exception as e:
                    print(f"Unexpected error in rx_task: {e}")
                    self.error_counter += 1
                    self._handle_connection_error()

            elif self.state == SerialState.RECOVERING:
                self._close_port()
                time.sleep(self.reconnect_interval)
                if self._open_port():
                    self.state = SerialState.OPENNED
                    self.error_counter = 0
                    print("Connection recovered")
                else:
                    print("Recovery failed, entering scan mode...")
                    self.state = SerialState.SCANNING
                    self.last_scan_time = 0

            time.sleep(0.001)

        self._close_port()
        self.close_log_file()
        self.state = SerialState.CLOSED
        print("RX task terminated")

    def _handle_connection_error(self):
        """处理连接错误"""
        if self.error_counter >= self.max_error_count:
            print(f"Too many errors ({self.error_counter}), entering recovery mode...")
            self.state = SerialState.RECOVERING
        else:
            time.sleep(0.1)

    def _log_rx_package(self, cmd_id):
        cmd_id_str = f"0x{cmd_id:04x}"
        if cmd_id_str not in self.package_rx_count:
            self.package_rx_count[cmd_id_str] = 1
        else:
            self.package_rx_count[cmd_id_str] += 1

    def update_and_get_next_frame(self):
        size = self.rx_buffer.available()

        for _ in range(size):
            if self.rx_buffer.buffer[0] == START_BYTE:
                break
            self.rx_buffer.dequeue_front(1)
            size -= 1

        if size < FRAME_HEADER_SIZE:
            self.rx_count["Header_Incomplete"] += 1
            return PackageStatus.HEADER_INCOMPLETE

        self.rx_frame_buffer[:FRAME_HEADER_SIZE] = self.rx_buffer.dequeue_front(FRAME_HEADER_SIZE)
        header_bytes = self.rx_frame_buffer[:FRAME_HEADER_SIZE]
        data_length = struct.unpack("<H", header_bytes[1:3])[0]
        crc8_received = header_bytes[4]

        header_for_crc = header_bytes[:4]
        crc8_calculated = Crc.get_crc8_check_sum(header_for_crc)
        if crc8_received != crc8_calculated:
            self.rx_count["Header_CRC_Error"] += 1
            return PackageStatus.HEADER_CRC_ERROR

        if data_length > MAX_PACKAGE_SIZE:
            self.rx_count["Header_CRC_Error"] += 1
            return PackageStatus.HEADER_CRC_ERROR

        total_needed = FRAME_HEADER_SIZE + 2 + data_length + CRC16_SIZE
        if size < total_needed:
            self.rx_buffer.enqueue_front(header_bytes)
            self.rx_count["Payload_Incomplete"] += 1
            return PackageStatus.PAYLOAD_INCOMPLETE

        remaining_bytes = self.rx_buffer.dequeue_front(2 + data_length + CRC16_SIZE)
        self.rx_frame_buffer[FRAME_HEADER_SIZE : FRAME_HEADER_SIZE + data_length + CRC16_SIZE + 2] = remaining_bytes

        message_without_tail_crc = self.rx_frame_buffer[: FRAME_HEADER_SIZE + data_length + 2]
        crc16_received = struct.unpack("<H", self.rx_frame_buffer[FRAME_HEADER_SIZE + 2 + data_length : FRAME_HEADER_SIZE + 2 + data_length + CRC16_SIZE])[0]
        crc16_calculated = Crc.get_crc16_check_sum(message_without_tail_crc)

        if crc16_received != crc16_calculated:
            self.rx_count["CRC_Error"] += 1
            return PackageStatus.CRC_ERROR

        self.rx_count["Package_Complete"] += 1
        self.rx_count["Bytes_Received"] += total_needed
        cmd_id = struct.unpack("<H", self.rx_frame_buffer[FRAME_HEADER_SIZE : FRAME_HEADER_SIZE + 2])[0]
        self._log_rx_package(cmd_id)
        return PackageStatus.PACKAGE_COMPLETE

    def _get_cmd_id_and_data_from_buffer(self, data_length):
        """提取cmd_id和数据"""
        cmd_id = struct.unpack("<H", self.rx_frame_buffer[FRAME_HEADER_SIZE : FRAME_HEADER_SIZE + 2])[0]
        data = self.rx_frame_buffer[FRAME_HEADER_SIZE + 2 : FRAME_HEADER_SIZE + data_length + 2]
        return cmd_id, data

    def bind(self, msg_id, cb_func):
        if isinstance(msg_id, MsgID):
            msg_id = msg_id.value
        key_id = f"0x{msg_id:04x}"
        if key_id not in self.cb_funcs:
            self.cb_funcs[key_id] = []
        self.cb_funcs[key_id].append(cb_func)

    def unbind(self, msg_id, cb_func):
        if isinstance(msg_id, MsgID):
            msg_id = msg_id.value
        key_id = f"0x{msg_id:04x}"
        if key_id not in self.cb_funcs:
            print(f"Message ID {key_id} not registered")
            return
        if cb_func not in self.cb_funcs[key_id]:
            print(f"Callback function not found for ID {key_id}")
            return
        self.cb_funcs[key_id].remove(cb_func)

    def tx(self, data: bytes):
        """发送消息并记录"""
        if self.state != SerialState.OPENNED:
            return False
        if not isinstance(data, bytes):
            print("Data must be of type bytes")
            return False

        try:
            if len(data) >= FRAME_HEADER_SIZE + 2:
                cmd_id = struct.unpack("<H", data[FRAME_HEADER_SIZE : FRAME_HEADER_SIZE + 2])[0]
                with self.lock:
                    timestamp = self.get_formatted_time()
                    data_hex = data.hex()
                    log_entry = f"tx {timestamp} frame_{self.tx_frame_count:06d} cmd_id=0x{cmd_id:04x} data={data_hex}\n"
                    if self.log_file:
                        self.log_file.write(log_entry)
                        self.log_file.flush()
                    self.tx_frame_count += 1
                    # print(f"Logged tx message: frame_{self.tx_frame_count-1:06d}")
            self.ser.write(data)
            return True
        except Exception as e:
            print(f"Error sending data: {e}")
            self.error_counter += 1
            self._handle_connection_error()
            return False

    def summarize(self):
        total_packages = (
            self.rx_count["Header_CRC_Error"]
            + self.rx_count["CRC_Error"]
            + self.rx_count["Package_Complete"]
        )
        if total_packages > 0:
            self.rx_rate["CRC_Error_Rate"] = (
                self.rx_count["Header_CRC_Error"] + self.rx_count["CRC_Error"]
            ) / total_packages

        print(f"Current port: {self.current_port}")
        print(f"Connection state: {self.state}")
        print(f"Error count: {self.error_counter}")
        print(f"Rx Count: {self.rx_count}")
        print(f"Rx Rate: {self.rx_rate}")
        print(f"Package count: {self.package_rx_count}")
        print(f"Tx frames logged: {self.tx_frame_count}")
        print(f"Rx frames logged: {self.rx_frame_count}")


def main(port, baudrate):

    def msg_cb_func(cmd_id, data):
        print("In Callback")
        print("cmd_id: 0x{:04x}; data length: {}".format(cmd_id, len(data)))

    referee_manager = RefereeSerialManager(port, baudrate, auto_scan=True)

    # Bind some test callbacks
    referee_manager.bind(MsgID.GAME_STATUS, msg_cb_func)
    referee_manager.bind(MsgID.ROBOT_HP, msg_cb_func)

    referee_manager.start()
    count = 0

    try:
        while True:
            count += 1
            if count % 100 == 0:
                referee_manager.summarize()

            # Send test message (you can customize this)
            from .serial_protocol import Radar2SentryMessage, Radar2ClientMessage
            radar2client_msg = Radar2ClientMessage(
                hero_x=100,
                hero_y=200,
                engineer_x=300,
                engineer_y=400,
                standard_3_x=500,
                standard_3_y=600,
                standard_4_x=700,
                standard_4_y=800,
                sentry_x=900,
                sentry_y=1000,

            )
            referee_manager.tx(radar2client_msg.pack())
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        referee_manager.close()


if __name__ == "__main__":
    SERIAL_PORT = "/dev/ttyUSB0"  # Change to your serial port
    BAUD_RATE = 115200  # RoboMaster referee system baud rate

    main(SERIAL_PORT, BAUD_RATE)
