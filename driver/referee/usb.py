import serial
import serial.tools.list_ports
import glob
import os


class USBDeviceManager:

    @staticmethod
    def scan_tty_devices_direct():
        """直接扫描/dev/目录下的串口设备"""
        patterns = [
            '/dev/ttyUSB*',
            '/dev/ttyACM*', 
            '/dev/ttyCH341USB*',
            '/dev/ttyS*'
        ]
        
        devices = []
        for pattern in patterns:
            devices.extend(glob.glob(pattern))
        
        return sorted(devices)

    @staticmethod
    def scan_usb_serial_ports():
        """扫描所有可用的USB串口设备，结合pyserial和直接扫描"""
        usb_ports = []
        found_devices = set()

        # 方法1: 使用pyserial扫描
        try:
            ports = serial.tools.list_ports.comports()
            for port in ports:
                # print(f"PySerial found: {port.device}")
                found_devices.add(port.device)
                # 过滤USB设备
                if (
                    "USB" in port.description.upper()
                    or "ttyUSB" in port.device
                    or "ttyACM" in port.device
                    or "ttyCH341USB" in port.device
                ):
                    usb_ports.append(
                        {
                            "device": port.device,
                            "description": port.description,
                            "manufacturer": getattr(port, "manufacturer", "Unknown"),
                            "vid": getattr(port, "vid", None),
                            "pid": getattr(port, "pid", None),
                            "serial_number": getattr(port, "serial_number", None),
                            "source": "pyserial"
                        }
                    )
        except Exception as e:
            print(f"Error scanning with pyserial: {e}")

        # 方法2: 直接扫描/dev/目录
        try:
            direct_devices = USBDeviceManager.scan_tty_devices_direct()
            for device in direct_devices:
                # print(f"Direct scan found: {device}")
                if device not in found_devices:
                    # 检查设备是否存在且可访问
                    if os.path.exists(device):
                        usb_ports.append(
                            {
                                "device": device,
                                "description": "Direct scan device",
                                "manufacturer": "Unknown",
                                "vid": None,
                                "pid": None,
                                "serial_number": None,
                                "source": "direct_scan"
                            }
                        )
                        found_devices.add(device)
        except Exception as e:
            print(f"Error scanning with direct method: {e}")

        return usb_ports

    @staticmethod
    def test_serial_port(port, baudrate=115200, timeout=1):
        """测试指定的串口是否可用"""

        try:
            ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
            if ser.is_open:
                ser.close()
                return True
        except serial.SerialException as e:
            print(f"Serial port {port} is not available: {e}")
        return False

    @staticmethod
    def find_referee_system_port(baudrate=115200):
        """查找裁判系统串口，优先使用pyserial检测到的设备"""
        usb_ports = USBDeviceManager.scan_usb_serial_ports()
        
        # 先尝试pyserial检测到的设备
        pyserial_ports = [port for port in usb_ports if port.get("source") == "pyserial"]
        for port_info in pyserial_ports:
            port = port_info["device"]
            if USBDeviceManager.test_serial_port(port, baudrate):
                print(f"Found referee system at {port} (via pyserial)")
                return port
        
        # 如果pyserial检测到的设备都不可用，再尝试直接扫描到的设备
        direct_ports = [port for port in usb_ports if port.get("source") == "direct_scan"]
        for port_info in direct_ports:
            port = port_info["device"]
            if USBDeviceManager.test_serial_port(port, baudrate):
                print(f"Found referee system at {port} (via direct scan)")
                return port

        return None


if __name__ == "__main__":
    usb_manager = USBDeviceManager()
    ports = usb_manager.scan_usb_serial_ports()
    print("Available USB Serial Ports:")
    for port in ports:
        print(
            f"Device: {port['device']}, Description: {port['description']}, "
            f"Manufacturer: {port['manufacturer']}, Source: {port.get('source', 'unknown')}"
        )

    print(f"\nTotal found: {len(ports)} devices")
    
    referee_port = usb_manager.find_referee_system_port()
    if referee_port:
        print(f"Referee system found at: {referee_port}")
    else:
        print("No referee system found.")
