########################################################################
#
# Copyright (c) 2020, STEREOLABS.
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

"""
    Multi cameras sample showing how to open multiple ZED in one program
"""

import argparse
import cv2 as cv
import sys
import pyzed.sl as sl
from rgb_hands_tracker import RGB_Hands_Tracker
import numpy as np
import threading
import time
import signal



zed_list = []
left_list = []
depth_list = []
timestamp_list = []
thread_list = []
cloud_list = []
ocv_left_list = []
ocv_depth_list = []
ocv_cloud_list = []
tracker_list = []
hands_scores_list = []
hands_sides_list = []
hands_landmarks_list = []
hands_world_landmarks_list = []
cameras_hands_landmarks = []
cameras_hands_landmarks_pts = []
geometric_centers_list = []
mediapipe_hands_landmarks_pts_list = []
stop_signal = False



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
        "-cal",
        "--calibration",
        type=str,
        default="/home/labcom/Documents/Fusion/cal.json",
        help="Path to the calibration file.",
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



def signal_handler(signal, frame):
    global stop_signal
    stop_signal=True
    time.sleep(0.5)
    exit()



def grab_run(index):
    global stop_signal
    global zed_list
    global timestamp_list
    global left_list
    global depth_list
    global cloud_list
    global ocv_left_list
    global ocv_depth_list
    global ocv_cloud_list
    global tracker_list
    global hands_scores_list
    global hands_sides_list
    global hands_landmarks_list
    global hands_world_landmarks_list
    global cameras_hands_landmarks
    global cameras_hands_landmarks_pts
    global geometric_centers_list
    global mediapipe_hands_landmarks_pts_list


    runtime = sl.RuntimeParameters()
    while not stop_signal:
        err = zed_list[index].grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            # Retrieve left frame
            zed_list[index].retrieve_image(left_list[index], sl.VIEW.LEFT)
            # Retrieve depth Mat. Depth is aligned on the left frame
            zed_list[index].retrieve_measure(depth_list[index], sl.MEASURE.DEPTH)
            # Retrieve colored point cloud. Point cloud is aligned on the left image
            zed_list[index].retrieve_measure(cloud_list[index], sl.MEASURE.XYZ)
            # Update OCV view
            ocv_left_list[index] = left_list[index].get_data()
            ocv_left_list[index] = cv.cvtColor(ocv_left_list[index], cv.COLOR_BGRA2BGR)
            ocv_depth_list[index] = depth_list[index].get_data()
            ocv_cloud_list[index] = cloud_list[index].get_data()

            # Ask the hands tracker to do detection and/or tracking of the hands on the current frame
            tracker_list[index].update(ocv_left_list[index])
            # For each detected hands, draw on the frame the sqeleton and collect if asked the data to store in the csv file.
            hands_scores_list[index], hands_sides_list[index], hands_landmarks_list[index], hands_world_landmarks_list[index] =\
                tracker_list[index].draw_and_export_data(ocv_left_list[index])            
            if np.count_nonzero(hands_scores_list[index]) :
                cameras_hands_landmarks[index] =\
                    get_points_of_landmarks_camera_z(hands_landmarks_list[index], ocv_depth_list[index], True)
                cameras_hands_landmarks_pts[index] =\
                    get_point_cloud_of_hands(hands_landmarks_list[index], ocv_cloud_list[index])
                geometric_centers_list[index] =\
                    get_hands_geometric_centers(cameras_hands_landmarks_pts[index], hands_world_landmarks_list[index])
                mediapipe_hands_landmarks_pts_list[index] =\
                    get_points_of_landmarks_mediapipe_z(geometric_centers_list[index], hands_world_landmarks_list[index])
            timestamp_list[index] = zed_list[index].get_timestamp(sl.TIME_REFERENCE.CURRENT).data_ns
        time.sleep(0.001) #1ms
    zed_list[index].close()



def main():

    global stop_signal
    global zed_list
    global left_list
    global depth_list
    global cloud_list
    global ocv_left_list
    global ocv_depth_list
    global ocv_cloud_list
    global tracker_list
    global timestamp_list
    global thread_list
    global hands_scores_list
    global hands_sides_list
    global hands_landmarks_list
    global hands_world_landmarks_list
    global cameras_hands_landmarks
    global cameras_hands_landmarks_pts
    global geometric_centers_list
    global mediapipe_hands_landmarks_pts_list
    
    signal.signal(signal.SIGINT, signal_handler)

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
    cal_path = args.calibration

    print("Running...")
    
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.camera_fps = 30  # The framerate is lowered to avoid any USB3 bandwidth issues
    init.coordinate_units = sl.UNIT.CENTIMETER # Set coordinate units
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

    #List and open cameras
    name_list = []
    last_ts_list = []
    cameras = sl.Camera.get_device_list()
    index = 0
    for cam in cameras:
        init.set_from_serial_number(cam.serial_number)
        name_list.append("ZED {}".format(cam.serial_number))
        print("Opening {}".format(name_list[index]))
        zed_list.append(sl.Camera())

        left_list.append(sl.Mat())
        depth_list.append(sl.Mat())
        cloud_list.append(sl.Mat())

        ocv_left_list.append(None)
        ocv_depth_list.append(None)
        ocv_cloud_list.append(None)

        tracker_list.append(RGB_Hands_Tracker(num_hands))

        hands_scores_list.append(None)
        hands_sides_list.append(None)
        hands_landmarks_list.append(None)
        hands_world_landmarks_list.append(None)

        cameras_hands_landmarks.append(None)
        cameras_hands_landmarks_pts.append(None)
        geometric_centers_list.append(None)
        mediapipe_hands_landmarks_pts_list.append(None)

        timestamp_list.append(0)
        last_ts_list.append(0)
        status = zed_list[index].open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            zed_list[index].close()
        index = index +1

    #Start camera threads
    for index in range(0, len(zed_list)):
        if zed_list[index].is_opened():
            thread_list.append(threading.Thread(target=grab_run, args=(index,)))
            thread_list[index].start()
    
    #Display camera images
    print("[MAIN]\tRunning Hands Tracking Press 'q' to quit")
    while True :
        for index in range(0, len(zed_list)):
            if zed_list[index].is_opened():
                if (timestamp_list[index] > last_ts_list[index]):
                    cv.imshow(name_list[index], ocv_left_list[index])
                    x = round(depth_list[index].get_width() / 2)
                    y = round(depth_list[index].get_height() / 2)
                    err, depth_value = depth_list[index].get_value(x, y)
                    if np.isfinite(depth_value):
                        print("{} depth at center: {}MM".format(name_list[index], round(depth_value)))
                    last_ts_list[index] = timestamp_list[index]
        key = cv.waitKey(10)
        if key == ord('q'):
                break
        elif key == ord('p'):
            while True :
                button = cv.waitKey(1)
                if button == ord('p'):
                    break

    cv.destroyAllWindows()

    #Stop the threads
    stop_signal = True
    for index in range(0, len(thread_list)):
        thread_list[index].join()

    print("\nFINISH")

if __name__ == "__main__":
    main()