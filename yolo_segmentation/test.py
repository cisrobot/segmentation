import cv2
import numpy as np
import os
from glob import glob
from datetime import datetime
from ultralytics import YOLO
from skimage.morphology import skeletonize  # 스켈레톤 추출을 위해 추가

# ========= 설정 =========
VIDEO_PATH = '/home/hm/yolo_segmentation/original/record2_003.mp4'
MODEL_PATH = '/home/hm/yolo_segmentation/models/best.pt'
OUTPUT_PATH = '/home/hm/yolo_segmentation/segmented_output.avi'

SIDEWALK_CLASS_ID = 0  # dataset.yaml 기준 'sidewalk'
CONF_THRESHOLD = 0.5
IMG_SIZE = 640
ALPHA = 0.5             # 오버레이 투명도 (0~1)
MASK_COLOR = (0, 0, 255)  # 빨간색 (BGR)
CONTOUR_COLOR = (0, 255, 0) # 초록색 (BGR)
SKELETON_COLOR = (255, 0, 0) # 파란색 (BGR)

# ========= 모델 로딩 =========
model = YOLO(MODEL_PATH, task="segment")  # segmentation 모드로 로드

# ========= 비디오 설정 =========
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Failed to open video: {VIDEO_PATH}")
    exit(1)

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))

print("===== 시작: Sidewalk Segmentation =====")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 예측 (원본 해상도를 그대로 사용하며 내부에서 이미지 리사이즈됨)
    results = model.predict(
        source=frame,
        conf=CONF_THRESHOLD,
        stream=True,
        classes=[SIDEWALK_CLASS_ID],
        imgsz=IMG_SIZE
    )

    # results는 iterable이며, 첫 번째 결과만 사용 (보통 한 프레임에 대해 하나의 결과)
    for r in results:
        # segmentation 결과가 없는 경우 건너뛰기
        if r.masks is None:
            continue

        # r.masks.data: 모델이 반환한 마스크 텐서, 보통 shape=(n, H_mask, W_mask) (값은 0~1 범위)
        masks = r.masks.data.cpu().numpy()  # (n, H_mask, W_mask)
        # segmentation이 여러 객체에 대해 있을 수 있으므로, 각 객체마다 처리
        for mask in masks:
            # 임계값 0.7로 이진화 (0: 배경, 1: 객체)
            binary_mask = (mask > 0.7).astype(np.uint8)

            # 만약 mask 해상도가 원본 프레임과 다르면, 원본 해상도로 리사이즈 (cv2.resize 인자는 (width, height))
            if binary_mask.shape != (frame_h, frame_w):
                binary_mask = cv2.resize(binary_mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

            # 3채널 마스크 생성 (객체가 있는 부분은 1)
            mask_3ch = np.stack([binary_mask] * 3, axis=-1)

            # 색상 레이어: MASK_COLOR로 채운 영상
            color_layer = np.full_like(frame, MASK_COLOR, dtype=np.uint8)

            # 투명 오버레이: binary_mask가 1인 부분만 MASK_COLOR 오버레이 적용
            frame = np.where(mask_3ch == 1,
                             ((1 - ALPHA) * frame + ALPHA * color_layer).astype(np.uint8),
                             frame)

            # 윤곽선 그리기 (CONTOUR_COLOR, 두께 2)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, CONTOUR_COLOR, thickness=2)

            # ** 중심선(스켈레톤) 그리기 **
            # 스켈레톤은 이진 마스크의 중심선으로, skimage의 skeletonize() 함수를 사용
            skel = skeletonize(binary_mask.astype(bool))
            # skel은 boolean 배열. 해당 픽셀을 SKELETON_COLOR로 표시합니다.
            # 먼저, 복사본을 만듭니다.
            skel_overlay = frame.copy()
            skel_overlay[skel] = SKELETON_COLOR
            # 원본 frame에 스켈레톤 오버레이를 강하게 덮어쓰도록 하거나, alpha blending을 적용할 수 있습니다.
            # 여기서는 alpha blending을 적용합니다.
            frame = cv2.addWeighted(frame, 0.8, skel_overlay, 0.2, 0)

    writer.write(frame)
    cv2.imshow("Segmented (Sidewalk Only)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC로 종료
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"✅ 완료! 세그멘테이션 결과는 {OUTPUT_PATH}에 저장되었습니다.")
