'''
cff-version: 1.3.1
message: "If you use this software, please cite it as below."
authors:
  - name: "MMPose Contributors"
title: "OpenMMLab Pose Estimation Toolbox and Benchmark"
date-released: 2020-08-31
url: "https://github.com/open-mmlab/mmpose"
license: Apache-2.0
'''

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from mmpose.apis import init_model, inference_bottomup, inference_topdown
from mmpose.visualization import FastVisualizer
from mmengine.structures import InstanceData

class MMposeTracker(QThread):
    signal_preview = pyqtSignal(np.ndarray)
    signal_detection = pyqtSignal(np.ndarray)
    signal_data = pyqtSignal(list)

    render_preview = True

    def __init__(self):
        super().__init__()
        self.mmpose_model = init_model(
            config='models/configs/rtmo-s_8xb32-600e_body7-640x640.py',
            checkpoint='models/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.pth',
            device='cpu')
        self.mmpose_visualizer = FastVisualizer(metainfo=self.mmpose_model.dataset_meta, radius=10, line_width=6, kpt_thr=0.5)

    def setRenderPreview(self, b):
        self.render_preview = b

    def detect(self, frame):
        results = inference_topdown(self.mmpose_model, frame)
        pred_instances = results[0].pred_instances

        keypoints = {"detection":False}
        if len(pred_instances.keypoints) > 0:
            keypoints = {"detection":True,

                        "shoulder_R":pred_instances.keypoints[0][6],
                        "shoulder_L":pred_instances.keypoints[0][5],
                        "elbow_R":pred_instances.keypoints[0][8],
                        "elbow_L":pred_instances.keypoints[0][7],
                        "wrist_R":pred_instances.keypoints[0][10],
                        "wrist_L":pred_instances.keypoints[0][9], 
                        "knee_R":pred_instances.keypoints[0][14],
                        "knee_L":pred_instances.keypoints[0][13],
                        "ankle_R":pred_instances.keypoints[0][16],
                        "ankle_L":pred_instances.keypoints[0][15],

                        "shoulder_R_score":pred_instances.keypoint_scores[0][6],
                        "shoulder_L_score":pred_instances.keypoint_scores[0][5],
                        "elbow_R_score":pred_instances.keypoint_scores[0][8],
                        "elbow_L_score":pred_instances.keypoint_scores[0][7],
                        "wrist_R_score":pred_instances.keypoint_scores[0][10],
                        "wrist_L_score":pred_instances.keypoint_scores[0][9], 
                        "knee_R_score":pred_instances.keypoint_scores[0][14],
                        "knee_L_score":pred_instances.keypoint_scores[0][13],
                        "ankle_R_score":pred_instances.keypoint_scores[0][16],
                        "ankle_L_score":pred_instances.keypoint_scores[0][15]}

        preview = frame.copy()
        if self.render_preview:
            img = np.zeros_like(frame)
            if keypoints["detection"]:
                draw_instances = InstanceData()
                draw_instances.keypoints = [pred_instances.keypoints[0]]
                draw_instances.keypoint_scores = [pred_instances.keypoint_scores[0]]
                draw_instances.keypoints_visible = [pred_instances.keypoints_visible[0]]
                self.mmpose_visualizer.draw_pose(img, draw_instances)
            self.signal_detection.emit(img)

            if keypoints["detection"]:
                self.mmpose_visualizer.draw_pose(preview, draw_instances)
            self.signal_preview.emit(preview)

        self.signal_data.emit([keypoints, frame, preview])
        return [keypoints, frame, preview]
