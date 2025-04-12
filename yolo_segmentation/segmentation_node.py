#!/usr/bin/env python3
import os
import cv2
import numpy as np
from datetime import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO
import torch

# ─────────────── 설정 ───────────────
VIDEO_SAVE_DIR = '/home/hm/yolo_segmentation/predicted_output'  # 저장할 영상 디렉토리
MODEL_PATH = '/home/hm/sam2/runs/segment/train/weights/best.pt'  # 학습된 YOLO11n-seg 모델 경로 (.pt)
CONF_THRESHOLD = 0.5        # 예측 confidence 임계값
IMG_SIZE = 640              # 모델 추론시 사용할 입력 크기
SIDEWALK_CLASS_ID = 0       # dataset.yaml에서 설정한 sidewalk 클래스의 ID
ALPHA = 0.5                 # 오버레이 blending 계수 (0~1)
MASK_COLOR = (0, 0, 255)    # (B, G, R) – 빨간색 (마스크 내부 채우기)
CONTOUR_COLOR = (0, 255, 0) # (B, G, R) – 초록색 (윤곽선)

# ─────────────── YOLO‑segmentation 관련 ───────────────
# YOLO 모델을 segmentation 작업모드로 로드합니다.
# task="segment" 옵션을 반드시 명시해야 세그멘테이션 결과를 얻을 수 있습니다.
def load_yolo_model(model_path, conf_threshold, img_size, class_id):
    model = YOLO(model_path, task="segment")
    # (필요시 추가 파라미터 설정 가능)
    return model

# ─────────────── ROS2 노드 클래스 ───────────────
class YoloSegmentationNode(Node):
    def __init__(self):
        super().__init__('yolo_segmentation_node')
        self.get_logger().info("YOLO Segmentation Node 시작")
        
        # 파라미터 선언 (필요에 따라 launch file 등으로 오버라이드 가능)
        self.declare_parameter('model_path', MODEL_PATH)
        self.declare_parameter('conf_threshold', CONF_THRESHOLD)
        self.declare_parameter('img_size', IMG_SIZE)
        self.declare_parameter('sidewalk_class_id', SIDEWALK_CLASS_ID)
        self.declare_parameter('video_save_dir', VIDEO_SAVE_DIR)
        self.declare_parameter('input_topic', '/camera/image_raw')
        
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        img_size = int(self.get_parameter('img_size').get_parameter_value().integer_value)
        sidewalk_class_id = int(self.get_parameter('sidewalk_class_id').get_parameter_value().integer_value)
        self.video_save_dir = self.get_parameter('video_save_dir').get_parameter_value().string_value
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value

        # 모델 로드
        self.model = load_yolo_model(model_path, conf_threshold, img_size, sidewalk_class_id)
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.sidewalk_class_id = sidewalk_class_id

        # cv_bridge 초기화
        self.bridge = CvBridge()

        # 입력 이미지 토픽 구독
        self.subscription = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            10
        )
        
        self.get_logger().info(f"Subscribed to {input_topic}")
        
        # 영상 저장을 위한 VideoWriter (초기화는 첫 이미지 수신 시)
        self.video_writer = None
        self.frame_size = None  # (width, height)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge 변환 오류: {e}")
            return

        if self.frame_size is None:
            h, w = frame.shape[:2]
            self.frame_size = (w, h)
            fps = 20  # 기본 FPS (입력 토픽에 FPS 정보가 없으면 설정)
            now = datetime.now().strftime('%Y%m%d_%H%M%S')
            video_save_path = os.path.join(self.video_save_dir, f'segmented_output_{now}.mp4')
            os.makedirs(self.video_save_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, self.frame_size)
            self.get_logger().info(f"비디오 저장 파일: {video_save_path}")

        # YOLO 모델 추론 (입력 프레임은 원본 해상도; imgsz 파라미터는 내부 리사이즈에 사용)
        results = self.model.predict(frame, conf=self.conf_threshold, imgsz=self.img_size,
                                      classes=[self.sidewalk_class_id], stream=True)
        result = results[0]

        # 원본 복사본 생성
        annotated_frame = frame.copy()

        # segmentation 마스크 처리
        if result.masks is not None:
            # 모델 출력 마스크 텐서 (예: shape: (n, H_mask, W_mask))
            mask_tensor = result.masks.data
            # 모델 출력 마스크를 numpy 배열로 변환
            masks_np = mask_tensor.cpu().numpy()
            for mask in masks_np:
                # (0~1) 범위를 0~255로 변환 후 uint8 변환
                mask_uint8 = (mask * 255).astype(np.uint8)
                # 원본 영상 크기로 리사이즈 (cv2.resize 인자는 (width, height) 순)
                mask_resized = cv2.resize(mask_uint8, self.frame_size, interpolation=cv2.INTER_NEAREST)
                # 이진화 (threshold=127)
                _, binary_mask = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
                # colored overlay: MASK_COLOR 채우기
                overlay = np.zeros_like(frame)
                overlay[binary_mask == 255] = MASK_COLOR
                # alpha blending 적용
                annotated_frame = cv2.addWeighted(annotated_frame, 1-ALPHA, overlay, ALPHA, 0)
                # 윤곽선 그리기 (CONTOUR_COLOR, 두께 2)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated_frame, contours, -1, CONTOUR_COLOR, 2)

        # 결과 영상 디스플레이 및 저장
        cv2.imshow("Sidewalk Segmentation", annotated_frame)
        cv2.waitKey(1)
        if self.video_writer is not None:
            self.video_writer.write(annotated_frame)

    def destroy_node(self):
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = YoloSegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
