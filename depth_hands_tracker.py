import numpy as np
import matplotlib.pyplot as plt
import torch as tr
from torchvision.transforms import Resize

import os
import cv2 as cv

from model import PixelwiseRegression
from utils import load_model, select_gpus

class Depth_Hands_Tracker():
    def __init__(self, model_name, model_parameters, gpu_id):

        self.label_size = model_parameters["label_size"]

        select_gpus(gpu_id)
        self.device = tr.device('cuda:0' if tr.cuda.is_available() else 'cpu')
        
        self.model = PixelwiseRegression(21, **model_parameters)
        load_model(self.model, os.path.join('Model', model_name), eval_mode=True)
        self.model = self.model.to(self.device)



    def draw_skeleton(self, _img, joints, *, output_size=512, rP = 8, linewidth = 4, draw=False, skeleton_mode=0):

        fig, axes = plt.subplots(figsize=(4, 4))
        if skeleton_mode == 0:
            Index = [0, 1, 2, 3, 4]
            Mid = [0, 5, 6, 7, 8]
            Ring = [0, 9, 10, 11, 12]
            Small = [0, 13, 14, 15, 16]
            Thumb = [0, 17, 18, 19, 20]
            config = [Thumb, Index, Mid, Ring, Small]
        elif skeleton_mode == 1:
            Index = [0, 2, 9, 10, 11]
            Mid = [0, 3, 12, 13, 14]
            Ring = [0, 4, 15, 16, 17]
            Small = [0, 5, 18, 19, 20]
            Thumb = [0, 1, 6, 7, 8]
            config = [Thumb, Index, Mid, Ring, Small]

        np.shape(_img)
        img = cv.resize(_img, (output_size, output_size))
        img3D = np.zeros((img.shape[0], img.shape[1], 3))
        for i in range(3):
            img3D[:, :, i] = img
        is_hand = img3D != 0
        img3D = img3D / np.max(img3D)
        # img3D = img3D * 0.5 + 0.25
        img3D = 1 - img3D
        img3D[is_hand] *= 0.5
        joints = joints * (img.shape[0] - 1) + np.array([img.shape[0] // 2, img.shape[0] // 2])
        _joint = [(int(joints[i][0]), int(joints[i][1])) for i in range(joints.shape[0])]
        colors = [(1, 0, 0), (0.5, 0.5, 0), (0, 1, 0), (0, 0.5, 0.5), (0, 0, 1)]
        for i in range(5):
            for index in config[i]:
                cv.circle(img3D, _joint[index], rP, colors[i], -1)
            for j in range(len(config[i]) - 1):
                cv.line(img3D, _joint[config[i][j]], _joint[config[i][j+1]], colors[i], linewidth)
        if draw:
            axes.imshow(img3D)
            axes.axis("off")
            plt.show()
        else:
            return img3D

    def estimate(self, img):
        
        label_img = Resize(size=[self.label_size, self.label_size])(img)
        label_img = tr.reshape(label_img, (1, 1, self.label_size, self.label_size))
        mask = tr.where(label_img > 0, 1.0, 0.0)

        
        img = img.to(self.device, non_blocking=True)
        label_img = label_img.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)

        self.heatmaps, self.depthmaps, hands_uvd = self.model(img, label_img, mask)[-1]
        hands_uvd = hands_uvd.detach().cpu().numpy()
        self.hands_uvd = hands_uvd

        return hands_uvd