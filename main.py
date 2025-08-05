# -*- coding: utf-8 -*-
"""
Main entry point for RM2025 Radar Algorithm System
Copyright (c) 2025 香港科技大学ENTERPRIZE战队（HKUST ENTERPRIZE Team）

Licensed under the MIT License. See LICENSE file in the project root for license information.
"""

from driver.referee.referee_comm import RefereeCommManager
from utils.config import (
    load_cfg_from_cfg_file,
    merge_cfg_from_list,
)
from interface.core import launch
import argparse
import rclpy

from main_event_loop import MainEventLoop


def get_parser():
    parser = argparse.ArgumentParser(
        description="Pytorch Referring Expression Segmentation"
    )
    parser.add_argument("--gpu", default="0,1")
    parser.add_argument(
        "--config", default="path to xxx.yaml", type=str, help="config file"
    )
    parser.add_argument(
        "--device_config",
        default="path to device.yaml",
        type=str,
        help="camera and lidar setting",
    )
    parser.add_argument(
        "--opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="override some settings in the config.",
    )

    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.device_config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg

def load_yaml_config(path: str):
    import yaml
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    args = get_parser()
    yaml_config = load_yaml_config("config/params.yaml")
    rclpy.init()

    if yaml_config["debug"]["inference_video"]:
        from driver.hik_camera.mock_hik import SimpleHikCamera
        camera = SimpleHikCamera(
            video_source=yaml_config["debug"]["video_source"]
        )
    else:
        from driver.hik_camera.hik import SimpleHikCamera
        camera = SimpleHikCamera(args)

    referee = RefereeCommManager(port=yaml_config["referee"]["port"], 
                                baudrate=yaml_config["referee"]["baudrate"])   
    referee.start()

    event_loop = MainEventLoop(
        camera = camera, referee=referee
    )
    camera.start_streaming()
    
    
    if not yaml_config["debug"]["inference_video"] and yaml_config["debug"]["streaming_video"]:
        camera.start_saving_threads()

    launch()
    import time
    
    ## do not terminate the main eventloop
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, stopping...")
        referee.close()
        camera.stop_streaming()
        camera.stop_saving_images()
        camera.close()
        event_loop.stop()
        
        
