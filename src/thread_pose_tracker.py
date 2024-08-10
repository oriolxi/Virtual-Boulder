'''
cff-version: 1.3.1
message: "If you use this software, please cite it as below."
authors:
  - name: "MMPose Contributors"
title: "OpenMMLab Pose Estimation Toolbox and Benchmark"
date-released: 2020-08-31
url: "https://github.com/open-mmlab/mmpose"
license: Apache-2.0
model checkpoints MMDET - https://github.com/open-mmlab/mmdetection/tree/main/configs
model checkpoints MMPOSE - https://mmpose.readthedocs.io/en/latest/model_zoo/body_2d_keypoint.html
'''

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from mmengine import DefaultScope
from mmengine.structures import InstanceData
from mmpose.apis import init_model, inference_topdown
from mmpose.visualization import FastVisualizer
from mmdet.apis import init_detector, inference_detector

class PoseTracker(QThread):
    signal_preview = pyqtSignal(np.ndarray)
    signal_detection = pyqtSignal(np.ndarray)
    signal_data = pyqtSignal(list)

    render_preview = True
    render_reprojection = False

    margin = 200 # safty margin arround RoI bounding box
    person_label = 0 # label used in the detection dataset for 'person'

    def __init__(self):
        super().__init__()
        # create mmdetect model (human segmentations)
        with DefaultScope.overwrite_default_scope('mmdet'):
            self.mmdet_model = init_detector(
                config='models/configs/mmdet/yolox_tiny_8xb8-300e_coco.py',
                checkpoint='models/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth',
                device='cpu')
        
        # create mmpose model and visualizer (pose estimation)
        with DefaultScope.overwrite_default_scope('mmpose'):
            self.mmpose_model = init_model(
                config='models/configs/mmpose/rtmo-t_8xb32-600e_body7-416x416.py',
                checkpoint='models/rtmo-t_8xb32-600e_body7-416x416-f48f75cb_20231219.pth',
                device='cpu')
            '''
            self.mmpose_model = init_model(
                config='models/configs/mmpose/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py',
                checkpoint='models/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth',
                device='cpu')
                '''
            self.mmpose_visualizer = FastVisualizer(metainfo=self.mmpose_model.dataset_meta, radius=10, line_width=6, kpt_thr=0.5)

    def setRenderPreview(self, b):
        self.render_preview = b

    def setRenderReprojection(self, b):
        self.render_reprojection = b

    def detect(self, frame):
        # detect human bounding box
        with DefaultScope.overwrite_default_scope('mmdet'):
            result = inference_detector(self.mmdet_model, frame)
            pred_instances = result.pred_instances
            indexes = np.where(pred_instances.labels.numpy() == self.person_label)[0]
            if len(indexes) > 0:
                bbox = pred_instances.bboxes[int(indexes[0])]
                bbox_int = bbox.numpy().astype(int) + [ - self.margin, - self.margin, + self.margin, + self.margin]
                bbox_int = np.clip(bbox_int, 0, None) # ensure bounding box has positive values
                cropped_frame = frame[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]]
            else: 
                cropped_frame = frame
        
        # perform pose estimation
        with DefaultScope.overwrite_default_scope('mmpose'):
            result = inference_topdown(self.mmpose_model, cropped_frame)
            pred_instances = result[0].pred_instances
            
            keypoints = {"detection":False}
            if len(pred_instances.keypoints) > 0:
                bb_translate = [0, 0]
                if len(indexes) > 0: bb_translate = [bbox_int[0], bbox_int[1]]
                keypoints = {"detection":True,

                            "shoulder_R":pred_instances.keypoints[0][6] + bb_translate,
                            "shoulder_L":pred_instances.keypoints[0][5] + bb_translate,
                            "elbow_R":pred_instances.keypoints[0][8] + bb_translate,
                            "elbow_L":pred_instances.keypoints[0][7] + bb_translate,
                            "wrist_R":pred_instances.keypoints[0][10] + bb_translate,
                            "wrist_L":pred_instances.keypoints[0][9] + bb_translate,

                            "shoulder_R_score":pred_instances.keypoint_scores[0][6],
                            "shoulder_L_score":pred_instances.keypoint_scores[0][5],
                            "elbow_R_score":pred_instances.keypoint_scores[0][8],
                            "elbow_L_score":pred_instances.keypoint_scores[0][7],
                            "wrist_R_score":pred_instances.keypoint_scores[0][10],
                            "wrist_L_score":pred_instances.keypoint_scores[0][9]}

        with DefaultScope.overwrite_default_scope('mmpose'):
            preview = None
            if self.render_preview:
                preview = frame.copy()

                if len(indexes) > 0: 
                    cropped_preview = preview[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]]
                else: 
                    cropped_preview = preview
                if len(indexes) > 0: 
                    cv2.rectangle(preview, (bbox_int[0], bbox_int[1]), (bbox_int[2], bbox_int[3]), (255,255,255), 2)
                
                if keypoints["detection"]: 
                    draw_instances = InstanceData()
                    draw_instances.keypoints = [pred_instances.keypoints[0]]
                    draw_instances.keypoint_scores = [pred_instances.keypoint_scores[0]]
                    draw_instances.keypoints_visible = [pred_instances.keypoints_visible[0]]
                    self.mmpose_visualizer.draw_pose(cropped_preview, draw_instances)
                self.signal_preview.emit(preview)

                if self.render_reprojection:
                    if keypoints["detection"]:
                        img = np.zeros_like(frame)
                        if len(indexes) > 0: 
                            cropped_img = img[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]]
                        else: 
                            cropped_img = img
                        self.mmpose_visualizer.draw_pose(cropped_img, draw_instances)
                        self.signal_detection.emit(img)

        self.signal_data.emit([keypoints, frame, preview])
        return [keypoints, frame, preview]
