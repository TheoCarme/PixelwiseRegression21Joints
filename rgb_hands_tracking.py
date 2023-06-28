########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

import argparse
import cv2 as cv
import sys
import pyzed.sl as sl
from rgb_hands_tracker import RGB_Hands_Tracker
import numpy as np

def make_parser():
    parser = argparse.ArgumentParser("hands tracking")    
    parser.add_argument(
        "-svo",
        "--svo_path",
        type=str,
        help="Path to your input video.",
    )
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
    return parser



def get_point_cloud_of_hands(hands_landmarks, point_cloud, display=False) :

    point_cloud_hands = np.full(np.shape(hands_landmarks), np.nan)

    for idx_hands, hand_landmarks in enumerate(hands_landmarks) :
        for idx_landmark, landmark in enumerate(hand_landmarks) :
            x = landmark[0]
            y = landmark[1]
            a = round(x)
            b = round(y)
            if (a >= 0 and a < 1080) and (b >= 0 and b < 1920) :
                point_cloud_hands[idx_hands, idx_landmark] = point_cloud[a, b, :3]

            if display :
                print("\n[GET POINTS CLOUD OF HANDS]\tidx_hands = ", idx_hands, "\tidx_landmark = ", idx_landmark, "\nhand_landmarks = ", hand_landmarks,
                      "\n[GET POINTS CLOUD OF HANDS]\tx = ", x, "\ty = ", y, "\ta = ", a, "\tb = ", b,
                      "\n[GET POINTS CLOUD OF HANDS]\tpoint_cloud[a, b, :3] = ", point_cloud[a, b, :3])
    
    if display :
        print("\n[GET POINTS CLOUD OF HANDS]\tShape of point_cloud_hands : ", np.shape(point_cloud_hands), "\npoint_cloud_hands = ", point_cloud_hands)
    
    return point_cloud_hands



def get_points_of_landmarks_camera_z(hands_landmarks, depth_map, display=False) :

    camera_hands_landmarks_pts = np.full(np.shape(hands_landmarks), np.nan)
    for idx_hands, hand_landmarks in enumerate(hands_landmarks) :
        for idx_landmark, landmark in enumerate(hand_landmarks) :
            x = landmark[0]
            y = landmark[1]
            a = round(x)
            b = round(y)
            if (a >= 0 and a < 1080) and (b >= 0 and b < 1920) :
                z = depth_map[a, b]
                camera_hands_landmarks_pts[idx_hands, idx_landmark] = [x, y, z]
            else :                
                camera_hands_landmarks_pts[idx_hands, idx_landmark] = [x, y, np.nan]
    
    if display :
        print("\n[GET POINTS OF LANDMARKS CAMERA Z]\tShape of depth_map : ", np.shape(depth_map), "\tShape of camera_hands_landmarks_pts : ", np.shape(camera_hands_landmarks_pts),
              "\ncamera_hands_landmarks_pts = ", camera_hands_landmarks_pts)
    
    return camera_hands_landmarks_pts



def get_hands_geometric_centers(hands_landmarks_pts, hands_world_landmarks, display=False) :

    geometric_centers = np.full((len(hands_landmarks_pts), 3), 0.0)

    for idx_hands, hand_landmarks in enumerate(hands_landmarks_pts) :
        
        try :
            index_closest_z = np.nanargmin(hand_landmarks, axis=0)[2]
            closest_pt = hand_landmarks[index_closest_z]
            geometric_centers[idx_hands] = closest_pt - 1000*hands_world_landmarks[idx_hands, index_closest_z]
        except ValueError :
            print("\n[GET HANDS GEOMETRIC CENTERS]\tAn all-NaN slice was encountered.")

        if display :
            print("\n[GET HANDS GEOMETRIC CENTERS]\thand_landmarks = ", hand_landmarks,
                  "\n[GET HANDS GEOMETRIC CENTERS]\tindex of closest z = ", index_closest_z, "\n###\tclosest point = ", closest_pt, "\n###\tgeometric center = ", geometric_centers)

    return geometric_centers



def get_points_of_landmarks_mediapipe_z(geometric_centers, hands_world_landmarks) :

    mediapipe_hands_landmarks_pts = np.full(np.shape(hands_world_landmarks), np.nan)
    for idx_hands, hand_world_landmarks in enumerate(hands_world_landmarks) :
        
        x0 = geometric_centers[idx_hands, 0]
        y0 = geometric_centers[idx_hands, 1]
        z0 = geometric_centers[idx_hands, 2]

        for idx_landmark, landmark in enumerate(hand_world_landmarks) :
            x = x0 + 1000*landmark[0]
            y = y0 + 1000*landmark[1]
            z = z0 + 1000*landmark[2]
            mediapipe_hands_landmarks_pts[idx_hands, idx_landmark] = [x, y, z]
    
    return mediapipe_hands_landmarks_pts



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



if __name__ == "__main__":

    # Collect the parameters given as arguments
    args = make_parser().parse_args()
 
    # Get the maximum number of hands to track on the video
    num_hands = int(args.num_hands)
    
    # Get the instructions
    svo_path = args.svo_path
    resolution = args.resolution
    fps = args.frame_rate
    depth_mode = args.depth_mode
    display = args.display
    verbose = args.verbose

    # Create a Camera object
    zed = sl.Camera()

    # Initialise the object containing the hands tracker with the maximum number of hands it has to tracks.
    left_hands_tracker = RGB_Hands_Tracker(num_hands)

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.CENTIMETER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    
    # If applicable, use the SVO given as parameter
    # Otherwise use ZED live stream
    if svo_path :
        print("[MAIN]\tUsing SVO file: {0}".format(svo_path))
        init_params.svo_real_time_mode = True
        init_params.set_from_svo_file(svo_path)

    # Open the ZED m camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Get ZED camera information
    camera_info = zed.get_camera_information()

    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1920), min(camera_info.camera_configuration.resolution.height, 1080))
    image_scale = [display_resolution.width / camera_info.camera_configuration.resolution.width
                 , display_resolution.height / camera_info.camera_configuration.resolution.height]

    # Create ZED objects filled in the main loop
    left_frame = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()

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

            # depth_ocv_3d = np.dstack((depth_ocv, depth_ocv, depth_ocv))
            # depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_ocv, alpha=0.03), cv.COLORMAP_JET)
            left_frame_ocv = cv.cvtColor(left_frame_ocv, cv.COLOR_BGRA2BGR)

            # Ask the hands tracker to do detection and/or tracking of the hands on the current frame.
            left_hands_tracker.update(left_frame_ocv)

            # For each detected hands, draw on the frame the sqeleton and collect if asked the data to store in the csv file.
            hands_scores, hands_sides, hands_landmarks, hands_world_landmarks = left_hands_tracker.draw_and_export_data(left_frame_ocv)
            
            if np.count_nonzero(hands_scores) :
                camera_hands_landmarks = get_points_of_landmarks_camera_z(hands_landmarks, depth_ocv, True)
                camera_hands_landmarks_pts = get_point_cloud_of_hands(hands_landmarks, point_cloud_np)
                geometric_centers = get_hands_geometric_centers(camera_hands_landmarks_pts, hands_world_landmarks)
                mediapipe_hands_landmarks_pts = get_points_of_landmarks_mediapipe_z(geometric_centers, hands_world_landmarks)

            cv.imshow("[HANDS TRACKING ON THE LEFT CAMERA]   Press \'q\' to quit  /  Press \'p\' to play/pause  /  Press \'n\' to print the measured norms of the phalanxes  /  Press \'c\' to print the cross products of the palm  /  Press \'w\' to print the world coordinates of the hands", left_frame_ocv)
            
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('p'):
                while True :
                    button = cv.waitKey(1)
                    if button == ord('n'):
                        #phalanx_size = measure_phalanxes(camera_hands_landmarks_pts, True)
                        phalanx_size = measure_phalanxes(mediapipe_hands_landmarks_pts, True)
                    elif button == ord('c'):
                        #palm_cross_products = compute_palms_cross_products(camera_hands_landmarks_pts, True)
                        palm_cross_products = compute_palms_cross_products(mediapipe_hands_landmarks_pts, True)
                    elif button == ord('w'):
                        print("\n[MAIN]\tPixel coordinates of the hands with depth : ", camera_hands_landmarks)
                        print("[MAIN]\tWorld coordinates of the hands with camera z : ", camera_hands_landmarks_pts)
                        print("[MAIN]\tGeometric centers : ", geometric_centers)
                        print("[MAIN]\tWorld coordinates of the hands with mediapipe z : ", mediapipe_hands_landmarks_pts)
                    elif button == ord('p'):
                        break
            elif key == ord('n'):
                #phalanx_size = measure_phalanxes(camera_hands_landmarks_pts, True)
                phalanx_size = measure_phalanxes(mediapipe_hands_landmarks_pts, True)
            elif key == ord('c'):
                #palm_cross_products = compute_palms_cross_products(camera_hands_landmarks_pts, True)
                palm_cross_products = compute_palms_cross_products(mediapipe_hands_landmarks_pts, True)
            elif key == ord('w'):
                print("\n[MAIN]\tPixel coordinates of the hands with depth : ", camera_hands_landmarks)
                print("[MAIN]\tWorld coordinates of the hands with camera z : ", camera_hands_landmarks_pts)
                print("[MAIN]\tGeometric centers : ", geometric_centers)
                print("[MAIN]\tWorld coordinates of the hands with mediapipe z : ", mediapipe_hands_landmarks_pts)

    left_frame.free(sl.MEM.CPU)
    depth.free(sl.MEM.CPU)
    point_cloud.free(sl.MEM.CPU)
    zed.close()

    cv.destroyAllWindows()