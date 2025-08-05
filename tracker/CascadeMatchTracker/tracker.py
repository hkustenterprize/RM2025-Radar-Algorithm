from tracker.CascadeMatchTracker.guess_pts import PointGuesser
from tracker.CascadeMatchTracker.type import TrackingState, SingleDetectionResult, State
from tracker.CascadeMatchTracker.detector import BaseDetector
from tracker.CascadeMatchTracker.utils import (
    compute_iou,
    xyxy2xywh,
    xywh2xyxy,
)
from typing import List, Optional, Dict
from scipy.optimize import linear_sum_assignment
import numpy as np
import cv2
import time
from .type import BotIdTrack
from driver.referee.referee_comm import FACTION
from transform.solidwork2uwb import solidwork2uwb


class CascadeMatchTracker(BaseDetector):

    W1 = 5.0  ## Weight of the class_id confidence
    W2 = 1.0  ## Weight of the IOU
    W3 = 1.0  ## Weight of the unknown armor detection
    W4 = 0.4  ## Weight of the center distance
    W5 = 0.5  ## Weight of the speed diff

    COST_THRESHOLD = -0.5  ## Cost threshold for matching
    INACTIVE_COST_THRESHOLD = -1.0
    HIT_COUNT_THRESHOLD = 2  ## Hit count threshold to confirm a track
    MISS_COUNT_THRESHOLD = 5  ## Miss count threshold to mark a track as lost

    def __init__(self, config: Dict, pixel_world_transform, visualize: bool = False):
        super().__init__(config, pixel_world_transform, visualize)
        self.max_lost_frames = 15
        self.frame_count = 0  # 帧计数器
        self.fs = 30.0
        self.real_dt = 0.1
        self.last = time.time()
        self.bot_id_trajectories = {}  
        self.point_guesser = PointGuesser(config["tracker"]["point_guesser_config_path"])
        self.faction = FACTION.UNKONWN  # 默认阵营为红方

    def nms(
        self, detections: List[SingleDetectionResult], iou_threshold: float = 0.5
    ) -> List[SingleDetectionResult]:
        """
        Non-Maximum Suppression (NMS) to filter overlapping detections.
        Args:
            detection: List of SingleDetectionResult objects.
            iou_threshold: IoU threshold for suppression.
        Returns:
            Filtered list of detections after applying NMS.
        """
        if not detections:
            return detections

        def get_confidence(det: SingleDetectionResult) -> float:
            """
            Get the confidence score for a detection.
            """
            return (
                det.car_conf * det.class_conf
                if det.class_id != -1
                else det.car_conf * 0.1
            )

        # Sort detections by confidence score
        detections = sorted(detections, key=lambda x: get_confidence(x), reverse=True)
        keep = []
        for det in detections:
            # Skip if current detection overlaps with a kept detection of same or undefined class_id
            if any(
                compute_iou(det.car_box, kept_det.car_box) > iou_threshold
                and (
                    det.class_id == kept_det.class_id
                    or det.class_id == -1
                    or kept_det.class_id == -1
                )
                for kept_det in keep
            ):
                continue
            keep.append(det)
        return keep
    
    def _get_pos3d_diff(self, pos3d1: List[float], pos3d2: List[float]) -> float:
        """
        计算两个3D位置之间的差异
        Args:
            pos3d1: 第一个3D位置
            pos3d2: 第二个3D位置
        Returns:
            位置差异的欧氏距离, penalize more on the y axis
        """
        
        return np.linalg.norm(np.array(pos3d1) - np.array(pos3d2))
    
    def _get_pos2d_diff(self, pos2d1: List[float], pos2d2: List[float]) -> float:
        """
        计算两个2D位置之间的差异
        Args:
            pos2d1: 第一个2D位置
            pos2d2: 第二个2D位置
        Returns:
            位置差异的欧氏距离
        """
        return np.linalg.norm(np.array(pos2d1) - np.array(pos2d2))
    
    
    def _compute_score(
        self, track: TrackingState, det: SingleDetectionResult
    ) -> float:
        score = 0.0
        # box iou score
        # Do not compute iou for the INACTIVE and TENTATIVE tracks
        bot_id_trajectory: BotIdTrack = self.bot_id_trajectories[det.bot_id]
        if track.state in [State.CONFIRMED, State.TENTATIVE, State.LOST]:
            if track.car_box is not None:
                iou = compute_iou(det.car_box, track.car_box)
                score += iou * self.W2
            else:
                score += 0.0
                ## pos3d score
            if track.pos_3d is not None and det.pos_3d is not None:
                pos3d_diff = self._get_pos3d_diff(track.pos_3d, det.pos_3d)
                score += max(-1.0, 1.0 - pos3d_diff * self.W4)
            else:
                score += 0.0
                
            # bot id score
            if track.bot_id is not None:
                if track.bot_id == det.bot_id:
                    score += 1.0
                else:
                    score += 0.0
            else: 
                score += 0.0
        
        else:
            score += 0.5

        
        class_id_confidence, class_id_distribution = bot_id_trajectory.get_class_id_exponent_confidence(
            history_length=10, tau=0.5
        )
        # score += class_id_distribution[track.class_id] * self.W1
        score += class_id_confidence[track.class_id] * self.W1
        
        return score
        
            

    def track(
        self, img: np.ndarray, radar_data: Optional[np.ndarray] = None, faction = FACTION.RED, 
    ) -> tuple[List[dict], Optional[np.ndarray]]:
        self.faction = faction  # 更新阵营信息
        self.frame_count += 1
        now = time.time()
        if self.frame_count > 1:
            self.real_dt = now - self.last
        self.last = now
        ## step1: run detection and initialize botid trajectory

        detections, detect_vis_img = self.detect(img)
        
        for det in detections:
            if det.bot_id not in self.bot_id_trajectories:
                # self.bot_id_trajectories[det.bot_id] = BotIdTrack(
                #     bot_id=det.bot_id, class_id=det.class_id, class_conf = det.class_conf
                # )
                self.bot_id_trajectories[det.bot_id] = BotIdTrack(
                    bot_id=det.bot_id, class_id=det.class_id, class_conf=1.0
                )
                # self.bot_id_trajectories[det.bot_id].update(det.class_id)
            else:
                if det.class_id != -1:
                    # self.bot_id_trajectories[det.bot_id].update(det.class_id, det.class_conf)
                    self.bot_id_trajectories[det.bot_id].update(det.class_id, 1.0)  # Update with 1.0 confidence
                else:
                    self.bot_id_trajectories[det.bot_id].update(-1, 1.0)  # Update with -1 if not updated
            self.bot_id_trajectories[det.bot_id].updated = True
        
        # 收集需删除的bot_id
        to_delete = []
        for bot_id_trajectory in self.bot_id_trajectories.values():
            if not bot_id_trajectory.updated:
                bot_id_trajectory.update(-1, 1.0)
                bot_id_trajectory.lost_counter += 1
                if bot_id_trajectory.lost_counter >= 30:
                    to_delete.append(bot_id_trajectory.bot_id)
                    print(f"[CascadeMatchTracker]: Bot ID {bot_id_trajectory.bot_id} trajectory deleted due to timeout.")

        # 删除超时的轨迹
        for bot_id in to_delete:
            del self.bot_id_trajectories[bot_id]

        # 重置updated标志
        for bot_id_trajectory in self.bot_id_trajectories.values():
            bot_id_trajectory.updated = False
                
        # detections = self.nms(detections, iou_threshold=0.5)

        ## step2: Kalman filter state interpolation for tracks that is TENTATIVE or CONFIRMED
        for track in self.tracks:
            if track.state in [State.LOST, State.CONFIRMED]:
                track.pos_2d_uwb = track.kalman_filter_2d.predict(dt = self.real_dt)[0]
                xywh = track.kalman_filter.predict()
                xyxy = xywh2xyxy(xywh)
                track.box_trajectories.append(xyxy)
                track.car_box = xyxy
                track.pos_3d = self.xyxy2pos3d(xyxy)
                track.pos_2d_uwb_det = solidwork2uwb(track.pos_3d, faction)
                track.kalman_filter_2d.update(track.pos_2d_uwb_det)
                

        ## step3: Match detections with existing tracks

        ## i): Build the cost matrix
        N, M = len(detections), len(self.tracks)
        cost_matrix = np.zeros((M, N), dtype=np.float32)

        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                score = self._compute_score(track, det)
                cost_matrix[i, j] = -1.0 * score  # 取负值，因为我们要最小化成本
        ## ii): Hungarian matching
        matches_temp = linear_sum_assignment(cost_matrix)
        ## Filter out the high-cost matches with state-specific thresholds
        matches = []
        for i, j in zip(*matches_temp):
            track = self.tracks[i]
            threshold = self.INACTIVE_COST_THRESHOLD if (track.state == State.INACTIVE or track.state == State.TENTATIVE) else self.COST_THRESHOLD
            if cost_matrix[i, j] < threshold:
                matches.append((i, j))
        
        ## Find the matched track and the unmatched track
        matched_track_det = [(self.tracks[i], detections[j]) for (i, j) in matches]
        unmatched_tracks = [
            track
            for track in self.tracks
            if track not in [mt[0] for mt in matched_track_det]
        ]

        ## step4: Update the track based on the matching
        ## Matched tracks
        for track, det in matched_track_det:
            track.is_dead = True if det.class_id >= 10 else False
            match track.state:
                case State.INACTIVE:
                    track.state = State.TENTATIVE
                    track.hit_count = 0
                    track.inacitve_count = 0
                case State.TENTATIVE:
                    track.hit_count += 1
                    if track.hit_count >= self.HIT_COUNT_THRESHOLD:
                        track.is_active = True
                        track.state = State.CONFIRMED
                        initial_xywh = xyxy2xywh(det.car_box)
                        initial_pos_3d = self.xyxy2pos3d(det.car_box)
                        initial_pos_2d_uwb = solidwork2uwb(initial_pos_3d, faction)
                        track.kalman_filter.reset(initial_xywh)
                        track.kalman_filter_2d.reset(initial_pos_2d_uwb)
                case State.CONFIRMED:
                    ## Associate the bot id with the track
                    track.is_active = True
                    
                    ## Jump condition: The bot id is not the same with the track's bot id
                    if track.bot_id != det.bot_id:
                        # Reset the Kalman filter with the new detection
                        initial_xywh = det.car_box
                        initial_pos_3d = self.xyxy2pos3d(det.car_box)
                        initial_pos_2d_uwb = solidwork2uwb(initial_pos_3d, faction)
                        track.kalman_filter.reset(xyxy2xywh(det.car_box))
                        track.kalman_filter_2d.reset(initial_pos_2d_uwb)
                    else:
                        track.kalman_filter.update(xyxy2xywh(det.car_box))
                        ## Outlier filtering
                        norm = np.linalg.norm(
                            np.array(track.kalman_filter_2d.get_position()) - np.array(track.pos_2d_uwb_det)
                        )
                        OUTLIER_THRESHOLD = 1.5
                        if norm < OUTLIER_THRESHOLD:
                            track.kalman_filter_2d.update(track.pos_2d_uwb_det)

                    track.bot_id = det.bot_id
                case State.LOST:
                    ## Associate the bot id with the track
                    track.is_active = True

                    ## Jump condition: The bot id is not the same with the track's bot id
                    if track.bot_id != det.bot_id:
                        # Reset the Kalman filter with the new detection
                        initial_xywh = det.car_box
                        initial_pos_3d = self.xyxy2pos3d(det.car_box)
                        initial_pos_2d_uwb = solidwork2uwb(initial_pos_3d, faction)
                        track.kalman_filter.reset(xyxy2xywh(det.car_box))
                        track.kalman_filter_2d.reset(initial_pos_2d_uwb)
                        ## Debug: print in red
                    else:
                        track.kalman_filter.update(xyxy2xywh(det.car_box))
                        track.kalman_filter_2d.update(track.pos_2d_uwb_det)
        
                    track.miss_count = 0
                    track.state = State.CONFIRMED
                    track.bot_id = det.bot_id

        for track in unmatched_tracks:
            match track.state:
                case State.INACTIVE:
                    track.is_active = False
                    track.inactive_count += 1
                    continue
                case State.TENTATIVE:
                    track.is_active = False
                    track.hit_count -= 1
                    if track.hit_count <= 0:
                        track.state = State.INACTIVE
                case State.CONFIRMED:
                    track.state = State.LOST
                    track.miss_count = 0
                case State.LOST:
                    track.miss_count += 1
                    if track.miss_count >= self.MISS_COUNT_THRESHOLD:
                        track.guess_point = self.point_guesser.predict_points(track, ref_color = "blue" if self.faction == FACTION.BLUE else "red")
                        track.is_start_guess = True
                        print(f"[CascadeMatchTracker]: Track lost, guess point set {track.guess_point} for robot:", track.class_id)
                        
                        # track.box_trajectories.clear()
                        track.bot_id = -1 # Reset bot_id assotiation when track is inactive
                        track.kalman_filter.reset()
                        track.kalman_filter_2d.reset()
                        track.is_active = False
                        track.state = State.INACTIVE
                        
                        ## Add guess point logic here

        track_vis_img = self.visualize_results(
            img,
            detections,
            {track.class_id: track for track in self.tracks},
        )

        return (
            self.tracks,
            detect_vis_img,
            track_vis_img,
        )
    
    def warmup(self, warmup_num = 20, img_path = "./tracker/warmup_image.jpg") -> None:
        """预热函数，执行一定数量的检测以初始化模型"""
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        for num in range(warmup_num):
            detections, _ = self.detect(img)
            print(f"[tracker]: Doing warm up {num}", )

    def visualize_results(
        self,
        img: np.ndarray,
        detections: List[SingleDetectionResult],
        tracking_states: Dict[int, TrackingState],
    ) -> np.ndarray:
        """可视化检测和追踪结果，适配新逻辑"""
        # 首先将输入图像resize到1024x768
        vis_img = cv2.resize(img, (1024, 768), interpolation=cv2.INTER_LINEAR)

        # 计算缩放比例，用于调整边界框坐标
        original_height, original_width = img.shape[:2]
        scale_x = 1024 / original_width
        scale_y = 768 / original_height

        # 后续代码保持不变，但需要调整边界框坐标
        height, width = vis_img.shape[:2]  # 现在是768, 1024
        panel_width = 0

        # 绘制追踪结果
        for class_id, track in tracking_states.items():
            if not track.is_active or track.car_box is None:
                continue
            color = (0, 0, 255) if class_id < 5 else (255, 0, 0)
            x1, y1, x2, y2 = map(int, track.car_box)
            x1 = int(track.car_box[0] * scale_x)
            y1 = int(track.car_box[1] * scale_y)
            x2 = int(track.car_box[2] * scale_x)
            y2 = int(track.car_box[3] * scale_y)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            label = f"{track.name} BOT ID:{track.bot_id} State:{track.state.value}"
            cv2.putText(
                vis_img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255) if class_id < 5 else (255, 0, 0),
                2,
            )

        # 绘制信息面板
        y_offset = 30
        cv2.putText(
            vis_img,
            "Tracking Status",
            (width + 10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
        y_offset += 30
        for class_id, track in sorted(tracking_states.items()):
            state_color = (0, 255, 0) if track.is_active else (128, 128, 128)
            status = f"{track.name}: {'Active' if track.is_active else 'Inactive'}, Conf: {track.confidence:.2f}, Dead: {track.is_dead}, ID: {track.bot_id}"
            cv2.putText(
                vis_img,
                status,
                (width + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                state_color,
                1,
            )
            y_offset += 20

        return vis_img

    def run(self, video_source: str = 0):
        """
        运行追踪器，处理视频流并显示可视化结果

        Args:
            video_source: 视频源（文件路径或摄像头索引，默认为 0）
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Cannot open video source {video_source}")
            return

        self.visualize = True  # 强制启用可视化
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 追踪并获取可视化结果
            track_results, detect_vis_img, track_vis_img = self.track(frame)
            if detect_vis_img is not None:
                detect_vis_img = cv2.cvtColor(detect_vis_img, cv2.COLOR_RGB2BGR)
                track_vis_img = cv2.cvtColor(track_vis_img, cv2.COLOR_RGB2BGR)
                detect_vis_img = cv2.resize(
                    detect_vis_img, (2048, 1024), interpolation=cv2.INTER_LINEAR
                )
                track_vis_img = cv2.resize(
                    track_vis_img, (2048, 1024), interpolation=cv2.INTER_LINEAR
                )

                cv2.imshow("CascadeMatchTracker-Detect", detect_vis_img)
                cv2.imshow("CascadeMatchTracker-Track", track_vis_img)

            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # 按's' 暂停
            if cv2.waitKey(1) & 0xFF == ord("s"):
                print("Paused. Press 'r' to resume.")
                while True:
                    if cv2.waitKey(1) & 0xFF == ord("r"):
                        print("Resumed.")
                        break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import yaml
    from transform.ray_renderer import PixelToWorld
    import numpy as np

    config = yaml.safe_load(open("config/params.yaml", "r"))
    pixel_world_transform = PixelToWorld.build_from_config(config)
    tracker = CascadeMatchTracker(config, pixel_world_transform, visualize=True)

    video_source = "test_assets/video/20250515_101351-2-3R2.avi"
    tracker.run(video_source)
