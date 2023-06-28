import argparse
import os
import numpy as np
import cv2 as cv
import torch as tr
from matplotlib import pyplot as plt
import pyzed.sl as sl
# python -m pip install --ignore-installed /home/labcom/Documents/PixelwiseRegression/pyzed-4.0-cp310-cp310-linux_x86_64.whl

from rgb_hands_tracker import RGB_Hands_Tracker
from depth_hands_tracker import Depth_Hands_Tracker
from utils import center_crop, norm_img

PIXEL_SIZE_2K = 0.002
PIXEL_SIZE_1080 = 0.002
PIXEL_SIZE_720 = 0.004
PIXEL_SIZE_WVGA = 0.008

def make_parser():
    parser = argparse.ArgumentParser("hands tracking")
    parser.add_argument(
        "-nh",
        "--num_hands",
        type=int,
        default=4,
        help="Maximum number of hands to track on the video.",
    )
    parser.add_argument(
        "-res",
        "--resolution",
        type=int,
        default=1080,
        help="Height component of the RGB cameras resolution among 4 options : 376, 720, 1080, 1242.",
    )
    parser.add_argument(
        "-fps",
        "--frame_rate",
        type=int,
        default=30,
        help="Frame rate in fps for the RGB cameras among 4 options which are not available for all resolutions : " +
        "15(for all resolutions), 30(for 376, 720 and 1080 resolutions), 60(for 376 and 720 resolutions) and 100(only for the 376 resolution).",
    )
    parser.add_argument(
        "-dm",
        "--depth_mode",
        type=str,
        default="ULTRA",
        help="Depth Computation Mode to use among 4 options : PERFORMANCE, QUALITY, ULTRA and NEURAL.",
    )
    parser.add_argument(
        "-d",
        "--display",
        action="store_true",
        help="To display the camera streams.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Whether you want to display the results.",
    )
    parser.add_argument(
        '--subject',
        type=int,
        default=0,
        help="the subject of model file"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='HAND17',
        help="choose from MSRA and HAND17"    
    )
    parser.add_argument(
        '--label_size',
        type=int,
        default=64
    )
    parser.add_argument(
        '--cube_size',
        type=int,
        default=10000
    )
    parser.add_argument(
        '--norm_method',
        type=str,
        default='instance',
        help='choose from batch and instance'
    )
    parser.add_argument(
        '--heatmap_method',
        type=str,
        default='softmax',
        help='choose from softmax and sumz'
    )
    parser.add_argument(
        '--gpu_id',
        type=str,
        default='0'
    )
    parser.add_argument(
        '--stages',
        type=int,
        default=2
    )
    parser.add_argument(
        '--features',
        type=int,
        default=128
    )
    parser.add_argument(
        '--level',
        type=int,
        default=4
    )
    return parser



def measure_phalanxes(hands_landmarks_pts, display=False) :

    phalanxes_size = np.zeros((len(hands_landmarks_pts), 5, 3), dtype=float)
    for idx_hand, hand_landmarks_pts in enumerate(hands_landmarks_pts) :
        for idx_finger in range(5) :
            for idx_phalanx in range(3) :
                idx_joint = idx_finger*4 + idx_phalanx
                A = hands_landmarks_pts[idx_hand, idx_joint+1]
                B = hands_landmarks_pts[idx_hand, idx_joint+2]
                size = np.sqrt( np.square(B[0]-A[0]) + np.square(B[1]-A[1]) + np.square(B[2]-A[2]) )
                phalanxes_size[idx_hand, idx_finger, idx_phalanx] = size
                
    if display :
        print("\n[MEASURE PHALANXES]\tPhalanxes sizes :\n", phalanxes_size)

    return phalanxes_size



def compute_palms_cross_products(hands_pts, display=False) :

    palm_cross_products = np.ones((len(hands_pts), 4, 3), dtype=float)
    
    for idx_hands, hand_pts in enumerate(hands_pts) :

        O5 = hands_pts[idx_hands, 5]-hands_pts[idx_hands, 0]
        O9 = hands_pts[idx_hands, 9]-hands_pts[idx_hands, 0]
        O13 = hands_pts[idx_hands, 13]-hands_pts[idx_hands, 0]
        O17 = hands_pts[idx_hands, 17]-hands_pts[idx_hands, 0]

        palm_cross_products[idx_hands, 0] = np.cross(O5, O9)
        palm_cross_products[idx_hands, 1] = np.cross(O5, O13)
        palm_cross_products[idx_hands, 2] = np.cross(O5, O17)
        palm_cross_products[idx_hands, 3] = np.cross(O9, O13)
        
    if display :
        print("\n[COMPUTE PALMS CROSS PRODUCTS]\tPalms cross products :\n", palm_cross_products)

    return palm_cross_products



def get_hands_close_up(bgr_img, depth_img, hand_landmarks, focal_length=None, cube_size=5000, display=False, clean=False) :
    
    img_shape = np.shape(depth_img)
    nb_hands = np.shape(hand_landmarks)[0]
        
    hands_frames = np.zeros((4), dtype=np.int32)

    if clean :
        depth_img = np.where((depth_img == -np.inf) | (depth_img == np.inf) | (depth_img != depth_img), 0.0, depth_img)

    origin = np.array((np.nanmin(hand_landmarks[:, :2], 0))-200, dtype=np.int32)    
    sides_size = int(np.max((np.nanmax(hand_landmarks[:, :2], 0))- origin, 0)+200)

    hands_frames[0] = origin[0] if origin[0] >= 0 else 0
    hands_frames[1] = origin[1] if origin[1] >= 0 else 0
    
    cond = (origin + sides_size >= img_shape)
    if np.any(cond) :
        hands_frames[2] = np.nanmin(img_shape - origin)
    else :
        hands_frames[2] = sides_size

    hands_frames[3] = 128/hands_frames[2]

    end = origin + hands_frames[2]
    depth_hand_close_up = depth_img[hands_frames[0] : end[0], hands_frames[1] : end[1]]
    bgr_hand_close_up = bgr_img[hands_frames[0] : end[0], hands_frames[1] : end[1]]

    if display :
        dist = depth_hand_close_up.astype(np.uint8)
        dist = cv.applyColorMap(dist, cv.COLORMAP_RAINBOW)
        cv.imshow("Hand close-up on the depth frame", dist)
        cv.imshow("Hand close-up on the bgr frame", bgr_hand_close_up)
        cv.waitKey()
        
    return bgr_hand_close_up, depth_hand_close_up, hands_frames



def get_full_size_uvd(hands_frames, depth_hands_landmarks, resolution) :

    for hand_frame, depth_hand_landmarks in zip(hands_frames, depth_hands_landmarks) :

        depth_hand_landmarks[:, :2] = (depth_hand_landmarks[:, :2] + 1) / 2 * 128
        depth_hand_landmarks[:, 2] = (depth_hand_landmarks[:, 2] + 1) * hand_frame[5] / 2 + hand_frame[4]
        
    return depth_hands_landmarks.astype(np.int32)



if __name__ == "__main__":

    # Collect the parameters given as arguments
    args = make_parser().parse_args()
 
    # Get the maximum number of hands to track on the video
    num_hands = int(args.num_hands)
    
    # Get the instructions
    resolution = args.resolution
    fps = args.frame_rate
    depth_mode = args.depth_mode
    display = args.display
    verbose = args.verbose
    cube_size = args.cube_size

    model_parameters = {
        "stage" : args.stages,
        "label_size" : args.label_size,
        "features" : args.features,
        "level" : args.level,
        "norm_method" : args.norm_method,
        "heatmap_method" : args.heatmap_method,
    }

    if args.dataset == 'HAND17' :
        skeleton_mode = 1
        model_name = "HAND17_default_final.pt"
    elif args.dataset == 'MSRA' :
        skeleton_mode = 0
        model_name = "MSRA_default_subject{}_final.pt".format(args.subject)    


    if not os.path.exists("skeleton"):
        os.mkdir("skeleton")

    if not os.path.exists(os.path.join("skeleton", args.dataset)):
        os.mkdir(os.path.join("skeleton", args.dataset))
        os.mkdir(os.path.join("skeleton", args.dataset, "predict"))

    assert os.path.exists('Model'), "Please put the models in ./Model folder"

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.MILLIMETER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    
    # Open the ZED m camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Get ZED camera information
    camera_info = zed.get_camera_information()
    left_focal = np.array([camera_info.camera_configuration.calibration_parameters.left_cam.fx, camera_info.camera_configuration.calibration_parameters.left_cam.fy])*PIXEL_SIZE_1080
    right_focal = np.array([camera_info.camera_configuration.calibration_parameters.right_cam.fx, camera_info.camera_configuration.calibration_parameters.right_cam.fy])*PIXEL_SIZE_1080
    
    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1920), min(camera_info.camera_configuration.resolution.height, 1080))
    image_scale = [display_resolution.width / camera_info.camera_configuration.resolution.width
                 , display_resolution.height / camera_info.camera_configuration.resolution.height]

    # Create ZED objects filled in the main loop
    left_frame = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()

    # Initialise the object containing the hands tracker with the maximum number of hands it has to tracks.
    rgb_hands_tracker = RGB_Hands_Tracker(num_hands)
    depth_hands_tracker = Depth_Hands_Tracker(model_name, model_parameters, left_focal, cube_size, args.gpu_id)

    print("[MAIN]\tRunning Hands Tracking Press 'q' to quit")
    while True :

        # Capture the frame from the ZED m camera
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve left frame
            zed.retrieve_image(left_frame, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            # Retrieve depth Mat. Depth is aligned on the left frame
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            # Retrieve colored point cloud. Point cloud is aligned on the left image
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ)

            # Update OCV view
            left_frame_ocv = left_frame.get_data()
            depth_ocv = depth.get_data()
            point_cloud_np = point_cloud.get_data()

            left_frame_ocv = cv.cvtColor(left_frame_ocv, cv.COLOR_BGRA2BGR)

            # Ask the hands tracker to do detection and/or tracking of the hands on the current frame.
            rgb_hands_tracker.update(left_frame_ocv)
                        
            # For each detected hands, draw on the frame the sqeleton and collect if asked the data to store in the csv file.
            hands_scores, hands_sides, hands_landmarks, hands_world_landmarks = rgb_hands_tracker.draw_and_export_data(left_frame_ocv, draw=False)

            nb_detected_hands = np.count_nonzero(hands_scores)
            depth_hands_landmarks = np.empty(np.shape(hands_landmarks), dtype=np.float32)
            hands_frames = np.empty((nb_detected_hands, 4), dtype=np.int32)

            if nb_detected_hands :

                for idx in range(nb_detected_hands) :
                    
                    bgr_hand_close_up, depth_hand_close_up, hands_frames[idx] = get_hands_close_up(left_frame_ocv, depth_ocv, hands_landmarks[idx], clean=True, display=False)
                    depth_hand_close_up = np.array([[depth_hand_close_up]])
                    depth_hands_landmarks[idx] = depth_hands_tracker.estimate(depth_hand_close_up)
                    hand_close_up_512 = depth_hands_tracker.draw_skeleton(depth_hand_close_up, idx, skeleton_mode=skeleton_mode)
                    hand_close_up_512 = np.clip(hand_close_up_512, 0, 1)
                    print("\n###[MAIN]\tShape of depth_hands_landmarks : ", np.shape(depth_hands_landmarks), "\n###\tdepth_hands_landmarks = ", depth_hands_landmarks)
                            
                    cv.imshow("[HANDS TRACKING ON DEPTH FRAMES HAND N°", idx, "]   Press \'q\' to quit  /  Press \'p\' to play/pause  /  Press \'n\' to print the measured norms of the phalanxes  /  Press \'c\' to print the cross products of the palm  /  Press \'w\' to print the world coordinates of the hands", hand_close_up_512)
            
            
            # cv.imshow("[HANDS TRACKING ON THE LEFT RGB FRAMES]  Press \'q\' to quit  /  Press \'p\' to play/pause  /  Press \'n\' to print the measured norms of the phalanxes  /  Press \'c\' to print the cross products of the palm  /  Press \'w\' to print the world coordinates of the hands", left_frame_ocv)
            
            key = cv.waitKey(1)
            if key == ord('q') :
                break
            elif key == ord('p') :
                while True :
                    button = cv.waitKey(1)
                    # if button == ord('n'):
                        #phalanx_size = measure_phalanxes(camera_hands_landmarks_pts, True)
                        # phalanx_size = measure_phalanxes(mediapipe_hands_landmarks_pts, True)
                    # elif button == ord('c'):
                        #palm_cross_products = compute_palms_cross_products(camera_hands_landmarks_pts, True)
                        # palm_cross_products = compute_palms_cross_products(mediapipe_hands_landmarks_pts, True)
                    # elif button == ord('w'):
                        # print("\n[MAIN]\tPixel coordinates of the hands with depth : ", camera_hands_landmarks)
                        # print("[MAIN]\tWorld coordinates of the hands with camera z : ", camera_hands_landmarks_pts)
                        # print("[MAIN]\tGeometric centers : ", geometric_centers)
                        # print("[MAIN]\tWorld coordinates of the hands with mediapipe z : ", mediapipe_hands_landmarks_pts)
                    if button == ord('p'):
                        break
            # elif key == ord('n'):
                # phalanx_size = measure_phalanxes(camera_hands_landmarks_pts, True)
                # phalanx_size = measure_phalanxes(mediapipe_hands_landmarks_pts, True)
            # elif key == ord('c'):
                # palm_cross_products = compute_palms_cross_products(camera_hands_landmarks_pts, True)
                # palm_cross_products = compute_palms_cross_products(mediapipe_hands_landmarks_pts, True)
            # elif key == ord('w'):
                # print("\n[MAIN]\tPixel coordinates of the hands with depth : ", camera_hands_landmarks)
                # print("[MAIN]\tWorld coordinates of the hands with camera z : ", camera_hands_landmarks_pts)
                # print("[MAIN]\tGeometric centers : ", geometric_centers)
                # print("[MAIN]\tWorld coordinates of the hands with mediapipe z : ", mediapipe_hands_landmarks_pts)

            # if key == ord('s'):
            #     plt.imsave(os.path.join("skeleton", args.dataset, "predict", "{}.jpg".format((datetime.datetime.now()).strftime("%f"))), skeleton_pre)

    left_frame.free(sl.MEM.CPU)
    depth.free(sl.MEM.CPU)
    point_cloud.free(sl.MEM.CPU)
    zed.close()

    cv.destroyAllWindows()