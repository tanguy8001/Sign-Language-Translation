from mmpose.apis import init_pose_model, inference_top_down_pose_model, vis_pose_result
import cv2

scanner = cv2.VideoCapture(0)
ret, frame = scanner.read()

model = init_pose_model("mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark.py", "model_paths/pose_check.pth",device="cuda")
pose_results, _ = inference_top_down_pose_model(
    model,
    frame,
    bbox_thr=None,
    format='xyxy',
    dataset=None,
    dataset_info=None,
    return_heatmap=False,
    outputs=None)

img = vis_pose_result(
    model,
    frame,
    pose_results
)

cv2.imshow("image", img)
cv2.waitKey(0)