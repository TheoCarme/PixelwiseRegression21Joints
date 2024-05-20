import numpy as np
import matplotlib.pyplot as plt
import torch as tr
from torchvision.transforms import Resize

import os
import cv2 as cv
import scipy

from model import PixelwiseRegression
from utils import load_model, select_gpus, center_crop, recover_uvd



########################################################################################################################
# A class to encapsulate PixelwiseRegression the model that estimate the pose of hands on an image.
# Inputs :  - model_name is a string containing the filename of the model to be used.
#           - model_parameters is a dictionary containing 7 parameters :
#               - stage
#               - label_size
#               - features
#               - level
#               - norm_method
#               - heatmap_method
#           - focal_length is an array containing the two elements in millimeters of the focal legth of the camera used.
#           - cube_size is a scalar that represents half the length of the side of the square.
#           The default value of the models used should be 150 (mm). The best value generally is slightly larger
#           than the size of the hand and depends on how good the segmentations are.
#           - gpu_id is an integer to indicate which gpu to use. If there is only one gpu or none put it to 0.
########################################################################################################################
class Depth_Hands_Tracker():
    def __init__(self, model_name, model_parameters, focal_length, cube_size, gpu_id):

        self.label_size = model_parameters["label_size"]
        self.fx = focal_length[0]
        self.fy = focal_length[1]
        self.cube_size = cube_size

        select_gpus(gpu_id)
        self.device = tr.device('cuda:0' if tr.cuda.is_available() else 'cpu')
        
        self.model = PixelwiseRegression(21, **model_parameters)
        load_model(self.model, os.path.join('Model', model_name), eval_mode=True)
        self.model = self.model.to(self.device)



########################################################################################################################
# A method that given a depth image of a close-up of an hand will compute all the necessary objects to run the model to
# estimate the corrdinates of the hand joints.
# Inputs :  - depth_hand_close_up is a close-up of a detected hands in the depth array
#
# Outputs : - normalized_img
#           - normalized_label_img is 
#           - mask is a tensor containing the mask which is equal to 0 when the label image is too and equal to 1 otherwise.
#           - box_size is a tensor storing the shape of the cropped image before resizing.
#           - cube_size is the same as the cube_size of the class but as a tensor.
#           - com is a tensor storing 3 float : the two coordinates of the center of mass of the pixels of depth array which are
#           superior to 0 and the mean depth of those pixels.
########################################################################################################################
    def get_cropped_and_normalized_hands(self, depth_hand_close_up) :

        cube_size = self.cube_size
        print("\n###\tShape of depth_hand_close_up : ", np.shape(depth_hand_close_up), "\\type of depth_hand_close_up : ",\
              type(depth_hand_close_up), "\ndepth_hand_close_up = ", depth_hand_close_up)
        mean = np.mean(depth_hand_close_up[depth_hand_close_up > 0])
        _com = scipy.ndimage.measurements.center_of_mass(depth_hand_close_up > 0)
        _com = np.array([_com[1], _com[0], mean])

        image = depth_hand_close_up.copy()
        com = _com.copy()

        # crop the image
        du = cube_size / com[2] * self.fx
        dv = cube_size / com[2] * self.fy
        box_size = int(du + dv)
        box_size = max(box_size, 2)

        crop_img = center_crop(image, (com[1], com[0]), box_size)
        crop_img = crop_img * np.logical_and(crop_img > com[2] - cube_size, crop_img < com[2] + cube_size)

        # norm the image and uvd to COM
        crop_img[crop_img > 0] -= com[2] # center the depth image to COM
        
        com[0] = int(com[0])
        com[1] = int(com[1])
        box_size = crop_img.shape[0] # update box_size

        print("\n###\tShape of crop_img : ", np.shape(crop_img), "\\type of crop_img : ",\
              type(crop_img), "\crop_img = ", crop_img)
        # resize the image and uvd
        try:
            img_resize = cv.resize(crop_img, (128, 128))
        except:
            # probably because size is zero
            print("resize error")
            raise ValueError("Resize error")

        # Generate label_image and mask
        label_image = cv.resize(img_resize, (64, 64))
        is_hand = label_image != 0
        mask = is_hand.astype(float)            

        # Just return the basic elements we need to run the network
        # normalize the image first before return
        normalized_img = img_resize / cube_size
        normalized_label_img = label_image / cube_size

        # Convert to torch format
        normalized_img = tr.from_numpy(np.array([normalized_img])).float().unsqueeze(0)
        normalized_label_img = tr.from_numpy(np.array([normalized_label_img])).float().unsqueeze(0)
        mask = tr.from_numpy(np.array([mask])).float().unsqueeze(0)
        box_size = tr.tensor(box_size).float()
        cube_size = tr.tensor(cube_size).float()
        com = tr.from_numpy(com).float()
        
        return normalized_img, normalized_label_img, mask, box_size, cube_size, com



########################################################################################################################
# A method that given a depth image of a hand close-up will estimate the 3D coordinates of the 21 joints composing the hand.
# Inputs :  - depth_hand_close_up is a depth image of close-up of a hand.
#
# Outputs : - hands_uvd is a numpy array containing the 3D coordinates of the 21 joints composing the hand.
########################################################################################################################
    def estimate(self, depth_hand_close_up):
        
        # Get as tensors all the data needed for the estimation of the coordinates the hand joints.
        normalized_img, normalized_label_img, mask, box_size, cube_size, com = self.get_cropped_and_normalized_hands(depth_hand_close_up[0, 0])

        print("\n###\tShape of normalized_img : ", np.shape(normalized_img), "\tshape of normalized_label_img : ",\
              np.shape(normalized_label_img), "\tshape of mask : ", np.shape(mask))
        normalized_img = normalized_label_img.to(self.device, non_blocking=True)
        normalized_label_img = normalized_label_img.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)

        print("\n###\tmemory_allocated = ", tr.cuda.memory_allocated(),\
              "\n###\tmax_memory_allocated = ", tr.cuda.max_memory_allocated(),\
              "\n###\tmemory_reserved = ", tr.cuda.memory_reserved(),\
              "\n###\tmax_memory_reserved = ", tr.cuda.max_memory_reserved())

        # Run the model to estimate the hand joints coordinates.
        self.heatmaps, self.depthmaps, hands_uvd = self.model(normalized_img, normalized_label_img, mask)[-1]
        print("\n###\tShape of hands_uvd : ", np.shape(hands_uvd), "\n###\thands_uvd = ", hands_uvd)
        
        # Recalculate the coodrinates of the hand joints so that they corresponds the close-up image given as parameter.
        hands_uvd = recover_uvd(hands_uvd, box_size, com, cube_size)

        hands_uvd = hands_uvd.detach().cpu().numpy()
        self.hands_uvd = hands_uvd

        return hands_uvd
    


    def draw_skeleton(self, _img, hand_idx, *, output_size=512, rP = 8, linewidth = 4, draw=False, skeleton_mode=0):
        fig, axes = plt.subplots(figsize=(4, 4))
        joints = self.hands_uvd[hand_idx]

        if joints.shape[0] == 21:
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



    def draw_skeleton_on_rgb(self, img, landmarks, output_size=512, rP = 4, linewidth = 2, draw=False, skeleton_mode=0):

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

        # img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

        colors = [(255, 0, 0), (128, 128, 0), (0, 255, 0), (0, 128, 128), (0, 0, 255)]

        for idx, finger in enumerate(config) :

            for joint in finger :
                cv.circle(img, landmarks[joint], rP, colors[idx], -1)
            for phalanx in range(len(finger) - 1):
                cv.line(img, landmarks[finger[phalanx]], landmarks[finger[phalanx+1]], colors[idx], linewidth)

        img  = cv.resize(img, (output_size, output_size))

        if draw:
            axes.imshow(img)
            axes.axis("off")
            plt.show()
        else:
            return img