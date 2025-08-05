# -*- coding: utf-8 -*-
"""
Hikvision Camera Driver for RM2025 Radar Algorithm System
Copyright (c) 2025 香港科技大学ENTERPRIZE战队（HKUST ENTERPRIZE Team）

Licensed under the MIT License. See LICENSE file in the project root for license information.
"""

import concurrent.futures
import os

os.environ["MVCAM_COMMON_RUNENV"] = "/opt/MVS/lib"
from .MvImport.MvCameraControl_class import *
import ctypes
from ctypes import *
import cv2
import numpy as np
from threading import Thread, Lock, Event, Condition
import time
import rclpy
from rclpy.node import Node
from enum import Enum
from utils.config import get_parser
from utils.average_meter import AverageMeter
from pathlib import Path
from queue import Queue
from datetime import datetime
from datetime import timezone, timedelta
tz = timezone(timedelta(hours=8))
import threading, queue
import concurrent
class HikState(Enum):
    DISABLED = 0
    WORKING = 1
    RECONNECTING = 2


class FrameData:
    def __init__(self, img_w, img_h):
        self.time_stamp = -1
        self.frame_buffer = np.zeros((img_h, img_w, 3), dtype=np.uint8)


class RingBuffer:
    def __init__(self, num, img_w, img_h):
        self.num = num
        self.frames = [FrameData(img_w, img_h) for _ in range(num)]
        self.write_pos = -1
        self.lock = Lock()
        self.group_conditions = {}
        self.group_last_read = {}

    def register_group(self, group_id):
        with self.lock:
            if group_id not in self.group_conditions:
                self.group_conditions[group_id] = Condition(self.lock)
                self.group_last_read[group_id] = -1

    def put(self, data, time_stamp):
        with self.lock:
            self.write_pos = (self.write_pos + 1) % self.num
            self.frames[self.write_pos].frame_buffer = data
            self.frames[self.write_pos].time_stamp = time_stamp
            for group_id, condition in self.group_conditions.items():
                condition.notify(n=1)

    def get_latest(self, group_id, timeout=1.0):
        with self.lock:
            if group_id not in self.group_conditions:
                raise ValueError(f"Group {group_id} not registered")
            condition = self.group_conditions[group_id]
            start_time = time.time()
            while True:
                if self.write_pos > self.group_last_read[group_id] or (
                    self.write_pos < self.group_last_read[group_id]
                    and self.write_pos + self.num > self.group_last_read[group_id]
                ):
                    self.group_last_read[group_id] = self.write_pos
                    return self.frames[self.write_pos]

                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return None

                remaining_timeout = timeout - elapsed
                condition.wait(remaining_timeout)

    def get_buffer(self, index):
        return self.frames[index]


class SimpleHikCamera(Node):
    _instance = None

    def __init__(self, args):
        if SimpleHikCamera._instance is not None:
            raise RuntimeError("SimpleHikCamera instance already exists")
        super().__init__("camera_driver")
        self.cam = MvCamera()

        self.stop_streaming_signal = None
        self.streaming_thread = None

        # save thread variables
        self.save_thread = None
        self.stop_save_signal = None
        self.save_queue = Queue(maxsize=100)  # 限制队列大小防止内存溢出
        self.enable_save = False
        self.save_directory = "saved_images"
        self.frame_counter = 0
        # fps calculation
        self.clock = self.get_clock()
        self.now = self.clock.now().nanoseconds / 1e9
        self.last = self.clock.now().nanoseconds / 1e9
        self.fps = 0.0

        self.status = HikState.DISABLED
        self.error_counter = 0

        self.width = args.width
        self.height = args.height
        self.args = args
        self.exposure = args.exposure_time
        self.gain = args.gain
        self._init_buffer()
        self.__class__._instance = self
        
        self.recording_workers_num = args.recording_workers_num
        self.recording_save_root_dir = args.recording_save_root_dir

    def _init_sdk(self):
        ret = MvCamera.MV_CC_Initialize()
        if ret != 0:
            self.get_logger().error(f"SDK initialize failed! Error code: {hex(ret)}")
        return ret

    def _init_device(self):
        device_list = MV_CC_DEVICE_INFO_LIST()
        ret = self.cam.MV_CC_EnumDevices(MV_USB_DEVICE, device_list)
        if ret != 0:
            self.get_logger().error(f"Enum devices failed! Error code: {hex(ret)}")
            return ret

        if device_list.nDeviceNum < 1:
            self.get_logger().error(f"No available cameras found")
            ret = 0xFF
            return ret

        self._print_device_info(device_list)
        stDeviceInfo = cast(
            device_list.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)
        ).contents
        ret = self.cam.MV_CC_CreateHandle(stDeviceInfo)
        if ret != 0:
            self.get_logger().error(f"Create handle failed! Error code: {hex(ret)}")
            return ret
        ret = self.cam.MV_CC_OpenDevice()
        if ret != 0:
            self.get_logger().error(f"Open device failed! Error code: {hex(ret)}")
        return ret

    def _init_buffer(self):

        self.stFrameInfo = MV_FRAME_OUT_INFO_EX()
        self.ring_buffer = RingBuffer(num=6, img_h=self.height, img_w=self.width)
        self.nPayloadSize = self.width * self.height
        self.data_buf = (ctypes.c_ubyte * self.width * self.height)()

    def _print_device_info(self, device_list):
        self.get_logger().info(f"Found {device_list.nDeviceNum} devices:")
        for i in range(device_list.nDeviceNum):
            dev_info = cast(
                device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)
            ).contents
            if dev_info.nTLayerType == MV_USB_DEVICE:
                usb_info = dev_info.SpecialInfo.stUsb3VInfo

                # Device name
                name_bytes = bytearray(usb_info.chUserDefinedName)
                device_name = name_bytes.split(b"\x00")[0].decode(
                    "ascii", errors="ignore"
                )

                # Serial Number
                serial_bytes = bytearray(usb_info.chSerialNumber)
                serial_num = serial_bytes.split(b"\x00")[0].decode(
                    "ascii", errors="ignore"
                )

                self.get_logger().info(f"Device {i}: {device_name}")
                self.get_logger().info(f"Serial: {serial_num}")

    def _get_int_value(self, name):
        stInt = MVCC_INTVALUE()
        ret = self.cam.MV_CC_GetIntValue(name, stInt)
        if ret != 0:
            self.get_logger().error(f"Get {name} failed! Error code: {hex(ret)}")
        return stInt.nCurValue

    def _get_enum_value(self, name):
        stEnum = MVCC_ENUMVALUE()
        ret = self.cam.MV_CC_GetEnumValue(name, stEnum)
        if ret != 0:
            self.get_logger().error(f"Get {name} failed! Error code: {hex(ret)}")
        return stEnum.nCurValue

    def _get_float_value(self, name):
        stFloat = MVCC_FLOATVALUE()
        ret = self.cam.MV_CC_GetFloatValue(name, stFloat)
        if ret != 0:
            self.get_logger().error(f"Get {name} failed! Error code: {hex(ret)}")
        return stFloat.fCurValue

    def _configure_basic(self):
        params = [
            ("PixelFormat", PixelType_Gvsp_BayerGB8, "enum"),
            ("AcquisitionMode", MV_ACQ_MODE_CONTINUOUS, "enum"),
            ("TriggerMode", MV_TRIGGER_MODE_OFF, "enum"),
            ("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF, "enum"),
            ("GainAuto", MV_GAIN_MODE_OFF, "enum"),
            ("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS, "enum"),
            ("GammaSelector", MV_GAMMA_SELECTOR_USER, "enum"),
            ("AcquisitionFrameRate", self.args.acquisition_rate, "float"),
            ("ExposureTime", self.args.exposure_time, "float"),
            ("Gain", self.args.gain, "float"),
            ("Width", self.args.width, "int"),
            ("Height", self.args.height, "int"),
        ]

        for name, value, value_type in params:
            if value_type == "float":
                self.cam.MV_CC_SetFloatValue(name, value)
            elif value_type == "enum":
                self.cam.MV_CC_SetEnumValue(name, value)
            elif value_type == "int":
                self.cam.MV_CC_SetIntValue(name, value)

        time.sleep(0.05)
        ## Check params
        width = self._get_int_value("Width")
        if width != self.width:
            self.get_logger().error(
                f"Camera width {width} does not match the configuration width {self.width}"
            )
            return 1

        height = self._get_int_value("Height")
        if height != self.height:
            self.get_logger().error(
                f"Camera height {height} does not match the configuration width {self.height}"
            )
            return 1

        pixel_format = self._get_enum_value("PixelFormat")

        if pixel_format != PixelType_Gvsp_BayerGB8:
            self.get_logger().error(
                f"Camera pixel format {pixel_format} does not match BayerGB8"
            )
            return 1

        return 0

    def set_exposure(self, exposure: float):
        """
        Set the exposure time of the camera.
        Args:
            exposure (float): Exposure time in seconds.
        """
        ret = self.cam.MV_CC_SetFloatValue("ExposureTime", exposure)
        if ret != 0:
            self.get_logger().error(f"Set exposure failed! Error code: {hex(ret)}")
            return False
        else:
            self.get_logger().info(f"Exposure set to {exposure} seconds")
        self.exposure = exposure
        return True 
    
    def set_gain(self, gain: float):
        """
        Set the gain of the camera
        Args:
            exposure (float): Gain value.
        """
        ret = self.cam.MV_CC_SetFloatValue("Gain", gain)
        if ret != 0:
            self.get_logger().error(f"Set gain failed! Error code: {hex(ret)}")
            return False
        else:
            self.get_logger().info(f"Gain set to {gain}")
        self.gain = gain
        return True
    
    def _get_formatted_time(self):
        return datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    def get_exposure(self):
        """
        Get the current exposure time of the camera.
        Returns:
            float: Current exposure time in seconds.
        """
        return self.exposure

    def is_connected(self):
        return self.status == HikState.WORKING

    def capture_one_frame(self):
        ret = self.cam.MV_CC_GetOneFrameTimeout(
            self.data_buf, self.nPayloadSize, self.stFrameInfo, 1000
        )
        # self.cam.MV_CC_StopGrabbing()

        if ret != 0:
            self.get_logger().error(f"Get frame failed! Error code: {hex(ret)}")

        else:
            self.get_logger().info("Get one frame success")

        width = self.stFrameInfo.nWidth
        height = self.stFrameInfo.nHeight
        pixel_format = self.stFrameInfo.enPixelType
        buf_view = memoryview(self.data_buf).cast("B")
        if pixel_format != PixelType_Gvsp_BayerGB8:
            self.get_logger().error(f"Unsupported format: {pixel_format}")

        return {
            "data": buf_view,
            "width": width,
            "height": height,
            "pixel_format": pixel_format,
            "frame_info": self.stFrameInfo,
        }

    def start_streaming(self):
        self.stop_streaming_signal = Event()
        self.streaming_thread = Thread(target=self.streaming_thread_impl)
        self.status = HikState.RECONNECTING
        self.streaming_thread.start()
        self.get_logger().info("Started streaming")

    def stop_streaming(self):
        if self.stop_streaming_signal is None or self.streaming_thread is None:
            self.get_logger().warning(
                "Need to start streaming before calling stopping, exiting"
            )
            return
        elif not self.streaming_thread.is_alive():
            self.get_logger().warning("Thread is not alive. Exiting")
            return
        self.cam.MV_CC_StopGrabbing()
        self.get_logger().info("Stopped streaming")

    def _create_save_directory(self):
        """创建保存图像的目录"""
        timestamp = self._get_formatted_time()
        self.save_directory = f"{self.recording_save_root_dir}/saved_images_{timestamp}"
        Path(self.save_directory).mkdir(parents=True, exist_ok=True)
        self.get_logger().info(f"Created save directory: {self.save_directory}")
        
    def save_thread_impl(self, image_data, thread_id):
        """单次图像保存任务"""
        try:
            timestamp, rgb_image = image_data
            filename = f"frame_{self.frame_counter:06d}_{timestamp}.jpg"
            filepath = os.path.join(self.save_directory, filename)
            
            # 转换为BGR格式保存
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(filepath, bgr_image)
            
            if success:
                with self.lock:  # 确保frame_counter线程安全
                    self.frame_counter += 1
                    if self.frame_counter % 100 == 0:
                        self.get_logger().info(f"Saved {self.frame_counter} frames")
            else:
                self.get_logger().error(f"Failed to save image: {filepath}")
        
        except Exception as e:
            self.get_logger().error(f"Error in save thread {thread_id}: {str(e)}")

    # def save_thread_impl(self):
    #     """保存图像的线程实现"""
    #     self.get_logger().info("Image save thread started")
    #     last = time.time()
    #     while not self.stop_save_signal.is_set():
    #         try:
    #             # 从队列中获取图像数据，超时时间1秒
    #             if not self.save_queue.empty():
    #                 image_data = self.save_queue.get(timeout=1.0)
    #                 timestamp, rgb_image = image_data
                    
    #                 # 生成文件名
    #                 filename = f"frame_{self.frame_counter:06d}_{timestamp}.jpg"
    #                 filepath = os.path.join(self.save_directory, filename)
                    
    #                 # 转换为BGR格式保存
    #                 bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                    
    #                 # 保存图像
    #                 success = cv2.imwrite(filepath, bgr_image)
    #                 if success:
    #                     self.frame_counter += 1
    #                     if self.frame_counter % 100 == 0:  # 每100帧打印一次日志
    #                         self.get_logger().info(f"Saved {self.frame_counter} frames")
    #                 else:
    #                     self.get_logger().error(f"Failed to save image: {filepath}")
                        
    #                 self.save_queue.task_done()
    #             else:
    #                 time.sleep(0.01)  # 队列为空时短暂休眠
                
    #             self.get_logger().info(
    #                 f"Image save fps: {1.0 / (time.time() - last):.2f}"
    #             )
    #             last = time.time()
                
                    
    #         except Exception as e:
    #             self.get_logger().error(f"Error in save thread: {str(e)}")
    #             time.sleep(0.1)
        
    #     self.get_logger().info("Image save thread stopped")

    # def start_saving_images(self):
    #     """开始保存图像"""
    #     if self.save_thread is not None and self.save_thread.is_alive():
    #         self.get_logger().warning("Save thread is already running")
    #         return
        
    #     self._create_save_directory()
    #     self.enable_save = True
    #     self.frame_counter = 0
    #     self.stop_save_signal = Event()
    #     self.save_thread = Thread(target=self.save_thread_impl)
    #     self.save_thread.start()
    #     self.get_logger().info("Started saving images")
    def start_saving_threads(self):
        """使用线程池启动保存任务"""
        if self.save_thread is not None and self.save_thread.is_alive():
            self.get_logger().warning("Save thread is already running")
            return
        self._create_save_directory()
        self.enable_save = True
        self.frame_counter = 0
        self.save_queue = queue.Queue()
        self.stop_save_signal = threading.Event()
        self.lock = threading.Lock()  # 用于线程安全计数
        num_threads = self.recording_workers_num
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
        self.get_logger().info(f"Started thread pool with {num_threads} workers")
        
        def process_queue():
            thread_id = threading.get_ident()  # 获取线程ID
            while not self.stop_save_signal.is_set():
                try:
                    if not self.save_queue.empty():
                        image_data = self.save_queue.get(timeout=1.0)
                        self.thread_pool.submit(self.save_thread_impl, image_data, thread_id)
                        self.save_queue.task_done()
                    else:
                        time.sleep(0.01)
                    
                except Exception as e:
                    self.get_logger().error(f"Error in queue processing: {str(e)}")
                    time.sleep(0.1)
    
        self.queue_thread = threading.Thread(target=process_queue, daemon=False)
        self.queue_thread.start()
    

    def stop_saving_images(self):
        """停止保存图像"""
        if self.save_thread is None or not self.save_thread.is_alive():
            self.get_logger().warning("Save thread is not running")
            return
        
        self.enable_save = False
        self.stop_save_signal.set()
        
        # 等待队列中的任务完成
        self.queue_thread.join()
        self.thread_pool.shutdown(wait=True)
        self.save_thread = None
        self.stop_save_signal = None
        self.get_logger().info(f"Stopped saving images. Total saved: {self.frame_counter}")

    def streaming_thread_impl(self):
        time.sleep(0.1)
        while not self.stop_streaming_signal.is_set():
            if self.status == HikState.WORKING:
                if self.error_counter > 5:
                    self.status = HikState.RECONNECTING
                    self._close_device()
                    self.fps = 0.0
                    time.sleep(0.1)

            elif self.status == HikState.RECONNECTING:
                if (
                    self._init_sdk() == 0
                    and self._init_device() == 0
                    and self._configure_basic() == 0
                    and self.cam.MV_CC_StartGrabbing() == 0
                ):
                    self.error_counter = 0
                    self.status = HikState.WORKING
                    time.sleep(1)
            else:
                raise ValueError("The hik camera should not be disabled when running")

            if self.status == HikState.WORKING:
                ret = self.cam.MV_CC_GetOneFrameTimeout(
                    self.data_buf,
                    self.nPayloadSize,
                    self.stFrameInfo,
                    1000,
                )
                if ret != 0:
                    self.get_logger().error(f"Get frame failed! Error code: {hex(ret)}")
                    self.error_counter += 1
                    time.sleep(0.05)
                    continue
                else:
                    self.error_counter = 0
                time_stamp = self.get_clock().now().nanoseconds / 1e9
                recorded_time_stamp = self._get_formatted_time()
                np_data_buf = np.frombuffer(
                    self.data_buf,
                    dtype=np.uint8,
                ).reshape(self.height, self.width)

                rgb = cv2.cvtColor(np_data_buf, cv2.COLOR_BAYER_GR2RGB)
                self.ring_buffer.put(rgb, time_stamp)

                # sace the frames
                if self.enable_save and not self.save_queue.full():
                    try:
                        rgb_view = rgb.view()
                        rgb_view.flags.writeable = False
                        self.save_queue.put(
                            (recorded_time_stamp, rgb_view), block=False
                        )
                    except:
                        pass

                self.now = self.clock.now().nanoseconds / 1e9
                self.fps = 0.8 * self.fps + 0.2 / (self.now - self.last + 1e-8)

                if self.args.display_fps:
                    self.get_logger().info("FPS {}".format(self.fps))
                self.last = self.now
                time.sleep(0.01)

                # self.cam.MV_CC_ClearImageBuffer()

                # self.get_logger().info(
                #     "Capture 1 frame. Timestamp: {:.4f}".format(
                #         self.get_clock().now().nanoseconds / 1e9,
                #     )
                # )
            elif self.status == HikState.RECONNECTING:
                time.sleep(0.2)

        print("Streaming thread stop by signal")

    def register_group(self, group_id: str):
        self.ring_buffer.register_group(group_id)

    def get_image_latest(
        self, group_id: str = "ptda", timeout: float = 1.0
    ) -> tuple[np.ndarray, float]:
        """
        Args:
            timeout: float, maximum time to wait for an image in seconds
        Returns:
            tuple: (image: np.ndarray, time_stamp: float)
        Raises:
            TimeoutError: not raised explicitly, but logs error on timeout
        """

        frame_data = self.ring_buffer.get_latest(group_id=group_id, timeout=timeout)
        if frame_data is None:
            return None, None

        img_data = frame_data.frame_buffer
        img_data.flags.writeable = False
        time_stamp = frame_data.time_stamp
        return img_data, time_stamp

    def get_time(self):
        return self.get_clock().now().nanoseconds / 1e9

    def get_fps(self):
        return self.fps

    def _close_device(self):
        self.cam.MV_CC_CloseDevice()
        self.cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()

    def close(self):
        if self.save_thread and self.save_thread.is_alive():
            self.stop_saving_images()
            self.get_logger().info("Save thread killed")
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.stop_streaming_signal.set()
            self.streaming_thread.join()
            self.streaming_thread = None
            self.stop_streaming_signal = None
            self.get_logger().info("Streaming thread killed")
        if self.cam:
            self._close_device()

        self.get_logger().info("Camera is closed gracefully")
        self.status = HikState.DISABLED

    def __del__(self):
        self.close()


if __name__ == "__main__":
    args = get_parser()
    camera = None
    rclpy.init()
    try:
        # camera = SimpleHikCamera(display=True)
        # camera.start_streaming()
        # while True:
        # time.sleep(0.1)

        camera = SimpleHikCamera(args)
        camera.start_streaming()
        camera.register_group("test")
        camera.start_saving_threads()
        exposure = 10000
        while True:
            img_rgb, timestamp = camera.get_image_latest("test", timeout=1)
            print(timestamp)
            print(camera.status)
            if img_rgb is not None:
                display_bgr = cv2.resize(
                    img_rgb, dsize=(1280, 840), interpolation=cv2.INTER_LINEAR
                )
                display_bgr = cv2.cvtColor(display_bgr, code=cv2.COLOR_RGB2BGR)

            # exposure += 100
            # camera.set_exposure(exposure)
            # print("Exposure: ", exposure)

    except KeyboardInterrupt:
        print("Stream stopped by user")
    except Exception as e:
        import traceback
        print(f"Error: {traceback.format_exc()}")
    finally:
        if camera:
            camera.close()
