# -*- coding: utf-8 -*-
"""
Main Event Loop for RM2025 Radar Algorithm System
Copyright (c) 2025 香港科技大学ENTERPRIZE战队（HKUST ENTERPRIZE Team）

Licensed under the MIT License. See LICENSE file in the project root for license information.
"""
from driver.referee.serial_protocol import *
from driver.referee.referee_comm import RefereeCommManager, FACTION
import yaml
from transform.ray_renderer import PixelToWorld
from threading import Thread, Event
import logging
import time
from dataclasses import dataclass


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model", "pfa_yolov5"))


@dataclass
class RefereeDivision2dPosition:
    x: int
    y: int
    is_valid: bool


class MainEventLoop:
    
    _instance = None

    def __init__(self, camera, referee: RefereeCommManager):
        # print(f"In initializer: {calibrated_R}, {calibrated_T}")
        self.config = yaml.safe_load(open("config/params.yaml", "r"))
        self.camera = camera
        self.camera.register_group("tracker")
        self.referee = referee
        ## Display status
        self.inference_fps = 0
        self.imgsize = self.config["car_detector"]["img_size"]

        os.environ["OPENBLAS_FPE"] = "0"

        # 0 - 4 for blue, 5 - 9 for red
        self.divisions_pos = [
            RefereeDivision2dPosition(-1, -1, False) for _ in range(10)
        ]
        
        # Load weight to the cuda during the initialization
        self.dummy_pixel_world_transform = PixelToWorld.build_from_config(
            self.config
        )
        if self.config["tracker"]["type"] == "CascadeMatchTracker":
            from tracker.CascadeMatchTracker.tracker import CascadeMatchTracker
            self.tracker = CascadeMatchTracker(
                self.config, pixel_world_transform=self.dummy_pixel_world_transform, visualize=True
        )
        elif self.config["tracker"]["type"] == "BotIdMatchTracker":
            from tracker.CascadeMatchTracker.botidtracker import BotIdMatchTracker
            self.tracker = BotIdMatchTracker(
                self.config, pixel_world_transform=self.dummy_pixel_world_transform, visualize=True
            )
        else:
            raise ValueError(f"Unknown tracker type: {self.config['tracker']['type']}")
        
        # warm up
        self.tracker.warmup(warmup_num=20)
        
        self.__class__._instance = self

        ## Update the reference of the visualization image
    
    def set_calibrated_param(self, calibrated_R, calibrated_T):
        """Set the calibrated parameters for the pixel to world transformation."""
        self.pixel_world_transform = PixelToWorld.build_from_config_and_extrinsics(
            self.config, calibrated_R, calibrated_T
        )
        self.calibrated_R = calibrated_R
        self.calibrated_T = calibrated_T
        self.tracker.pixel_world_transform = self.pixel_world_transform


    def run(self):
        """Run the main event loop for the tracker."""
        self.thread = Thread(target=self.main_loop_thread, daemon=False)
        self.set_event = Event()
        self.thread.start()

    def stop(self):
        """Stop the main event loop."""
        if getattr(self, "set_event", None) is None:
            logger.warning("Main event loop is not running, nothing to stop.")
            return
        elif getattr(self, "thread", None) is None:
            logger.warning("Main event loop thread is not running, nothing to stop.")
            return
        self.set_event.set()
        self.thread.join()

    def pack_radar2sentrymsg(self) -> Radar2SentryMessage:
        if self.faction == FACTION.RED or self.faction == FACTION.UNKONWN:  # Sending red target if blue
            is_blue = False
            hero_x = (
                self.divisions_pos[0].x / 100.0
                if self.divisions_pos[0].is_valid
                else -8888
            )
            hero_y = (
                self.divisions_pos[0].y / 100.0
                if self.divisions_pos[0].is_valid
                else -8888
            )
            engineer_x = (
                self.divisions_pos[1].x / 100.0
                if self.divisions_pos[1].is_valid
                else -8888
            )
            engineer_y = (
                self.divisions_pos[1].y / 100.0
                if self.divisions_pos[1].is_valid
                else -8888
            )
            standard_3_x = (
                self.divisions_pos[2].x / 100.0
                if self.divisions_pos[2].is_valid
                else -8888
            )
            standard_3_y = (
                self.divisions_pos[2].y / 100.0
                if self.divisions_pos[2].is_valid
                else -8888
            )
            standard_4_x = (
                self.divisions_pos[3].x / 100.0
                if self.divisions_pos[3].is_valid
                else -8888
            )
            standard_4_y = (
                self.divisions_pos[3].y / 100.0
                if self.divisions_pos[3].is_valid
                else -8888
            )
            sentry_x = (
                self.divisions_pos[4].x / 100.0
                if self.divisions_pos[4].is_valid
                else -8888
            )
            sentry_y = (
                self.divisions_pos[4].y / 100.0
                if self.divisions_pos[4].is_valid
                else -8888
            )
        else:
            is_blue = True
            hero_x = (
                self.divisions_pos[5].x / 100.0
                if self.divisions_pos[5].is_valid
                else -8888
            )
            hero_y = (
                self.divisions_pos[5].y / 100.0
                if self.divisions_pos[5].is_valid
                else -8888
            )
            engineer_x = (
                self.divisions_pos[6].x / 100.0
                if self.divisions_pos[6].is_valid
                else -8888
            )
            engineer_y = (
                self.divisions_pos[6].y / 100.0
                if self.divisions_pos[6].is_valid
                else -8888
            )
            standard_3_x = (
                self.divisions_pos[7].x / 100.0
                if self.divisions_pos[7].is_valid
                else -8888
            )
            standard_3_y = (
                self.divisions_pos[7].y / 100.0
                if self.divisions_pos[7].is_valid
                else -8888
            )
            standard_4_x = (
                self.divisions_pos[8].x / 100.0
                if self.divisions_pos[8].is_valid
                else -8888
            )
            standard_4_y = (
                self.divisions_pos[8].y / 100.0
                if self.divisions_pos[8].is_valid
                else -8888
            )
            sentry_x = (
                self.divisions_pos[9].x / 100.0
                if self.divisions_pos[9].is_valid
                else -8888
            )
            sentry_y = (
                self.divisions_pos[9].y / 100.0
                if self.divisions_pos[9].is_valid
                else -8888
            )
        return Radar2SentryMessage(
            is_blue=is_blue,
            hero_x=hero_x,
            hero_y=hero_y,
            engineer_x=engineer_x,
            engineer_y=engineer_y,
            standard_3_x=standard_3_x,
            standard_3_y=standard_3_y,
            standard_4_x=standard_4_x,
            standard_4_y=standard_4_y,
            sentry_x=sentry_x,
            sentry_y=sentry_y,  # 40
            suggested_target=0,  # No target suggested
            flags=0,  # No flags set
        )

    def pack_radar2clientmsg(self) -> Radar2ClientMessage:

        if self.faction == FACTION.RED or self.faction == FACTION.UNKONWN:
            hero_x = (
                int(self.divisions_pos[0].x) if self.divisions_pos[0].is_valid else 0
            )
            hero_y = (
                int(self.divisions_pos[0].y) if self.divisions_pos[0].is_valid else 0
            )
            engineer_x = (
                int(self.divisions_pos[1].x) if self.divisions_pos[1].is_valid else 0
            )
            engineer_y = (
                int(self.divisions_pos[1].y) if self.divisions_pos[1].is_valid else 0
            )
            standard_3_x = (
                int(self.divisions_pos[2].x) if self.divisions_pos[2].is_valid else 0
            )
            standard_3_y = (
                int(self.divisions_pos[2].y) if self.divisions_pos[2].is_valid else 0
            )
            standard_4_x = (
                int(self.divisions_pos[3].x) if self.divisions_pos[3].is_valid else 0
            )
            standard_4_y = (
                int(self.divisions_pos[3].y) if self.divisions_pos[3].is_valid else 0
            )
            sentry_x = (
                int(self.divisions_pos[4].x) if self.divisions_pos[4].is_valid else 0
            )
            sentry_y = (
                int(self.divisions_pos[4].y) if self.divisions_pos[4].is_valid else 0
            )
        else:
            hero_x = (
                int(self.divisions_pos[5].x) if self.divisions_pos[5].is_valid else 0
            )
            hero_y = (
                int(self.divisions_pos[5].y) if self.divisions_pos[5].is_valid else 0
            )
            engineer_x = (
                int(self.divisions_pos[6].x) if self.divisions_pos[6].is_valid else 0
            )
            engineer_y = (
                int(self.divisions_pos[6].y) if self.divisions_pos[6].is_valid else 0
            )
            standard_3_x = (
                int(self.divisions_pos[7].x) if self.divisions_pos[7].is_valid else 0
            )
            standard_3_y = (
                int(self.divisions_pos[7].y) if self.divisions_pos[7].is_valid else 0
            )
            standard_4_x = (
                int(self.divisions_pos[8].x) if self.divisions_pos[8].is_valid else 0
            )
            standard_4_y = (
                int(self.divisions_pos[8].y) if self.divisions_pos[8].is_valid else 0
            )
            sentry_x = (
                int(self.divisions_pos[9].x) if self.divisions_pos[9].is_valid else 0
            )
            sentry_y = (
                int(self.divisions_pos[9].y) if self.divisions_pos[9].is_valid else 0
            )
        return Radar2ClientMessage(
            hero_x=hero_x,
            hero_y=hero_y,
            engineer_x=engineer_x,
            engineer_y=engineer_y,
            standard_3_x=standard_3_x,
            standard_3_y=standard_3_y,
            standard_4_x=standard_4_x,
            standard_4_y=standard_4_y,
            standard_5_x=0,  # No standard 5 position
            standard_5_y=0,  # No standard 5 position
            sentry_x=sentry_x,
            sentry_y=sentry_y,  # 40
        )

    def main_loop_thread(self):

        logger.info(
            "Tracker initialized and the model weights are loaded, starting main loop."
        )
        time.sleep(2)  # Allow some time for the tracker to initialize
        while not self.set_event.is_set():
            time.sleep(0.01)
            start = time.time()
            # Update faction
            self.faction = self.referee.get_faction()

            try:
                # Get the latest frame from the camera
                frame, time_stamp = self.camera.get_image_latest("tracker", timeout=0.1)
                if frame is None:
                    # logger.warning("No frame received from camera, skipping iteration.")
                    continue
                # Process the frame with the tracker
                self.tracks, self.detect_vis_img, self.track_vis_img = (
                    self.tracker.track(frame, faction=self.faction)
                )

                # Get faction of the image
                for track in self.tracks:
                    if track.is_active:                      
                        self.divisions_pos[track.class_id] = RefereeDivision2dPosition(
                            x = int(track.pos_2d_uwb[0] * 100),
                            y = int(track.pos_2d_uwb[1] * 100),
                            is_valid=True
                        )
                    
                    elif track.is_start_guess:
                        self.divisions_pos[track.class_id] = RefereeDivision2dPosition(
                            x=int(track.guess_point[0] * 100),
                            y=int(track.guess_point[1] * 100),
                            is_valid=True
                        )

                    else:
                        self.divisions_pos[track.class_id] = RefereeDivision2dPosition(
                            x=-1, y=-1, is_valid=False
                        )
                ## send the message to the sentry and the referee
                self.referee.radar2sentry_msg = self.pack_radar2sentrymsg()
                self.referee.radar2client_msg = self.pack_radar2clientmsg()
            
            except Exception as e:
                import traceback

                logger.error(f"Error in main loop: {traceback.format_exc()}")
            finally:
                end = time.time()
                self.inference_fps = 1 / (end - start) if end - start > 0 else 0
                logger.info(f"Inference FPS: {self.inference_fps:.2f}")
