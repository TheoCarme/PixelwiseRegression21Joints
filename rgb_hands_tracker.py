import numpy as np
import cv2 as cv

import drawing_styles as mp_drawing_styles
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands



########################################################################################################################
# A class to encapsulate Mediapipe Hands the model that tracks the pose estimation of hands on a video stream.
# Inputs :  - num_hands is the maximum number of hands to track.
#           - model_complexity must be equal to 0 or 1. If equal to the model will be more complex and will give better results.
#           - min_detection_confidence is the minimum confidence score for the hand detection to be considered successful
#           in palm detection model.
#           - min_tracking_confidence is the minimum confidence score for the hand tracking to be considered successful.
#           This is the bounding box IoU threshold between hands in the current frame and the last frame.
########################################################################################################################
class RGB_Hands_Tracker():
    def __init__(self, num_hands=2, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        super(RGB_Hands_Tracker, self).__init__()

        # Initializing the Mediapipe Hands object for hands tracking
        self.tracker = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=num_hands,
                        model_complexity=model_complexity,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence)
        
        self.max_num_hands = num_hands
        self.hands_results = None
        

    
########################################################################################################################
# This method make the hands tracking model perform pose estimation of the hands on the given frame
# Input is the image/frame to analyse with Mediapipe Hands
########################################################################################################################
    def update(self, image) :
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # Process the image using the hand tracking model
        self.hands_results = self.tracker.process(image)            
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)



########################################################################################################################
# A method to call after the update method that can draw the skeletons of the detected hands on the given frame
# and can also return all the data linked to those hands
# Inputs :  - image is the one on which will be drawn the skeletons of the tracked hands. It should be the same image
#           that was analysed with the "update" method before. It is expected to be a numpy array.
#           - draw is a flag to set to "True" so that the method will draw the skeletons on the image.
#           - export is a flag to set to "True" so that the method will output all the data concerning all the tracked hands.
#           - height and width are those of the input image
#
# Outputs : - hands_scores is a numpy array with the confidence scores of all the tracked hands.
#           - hands_sides is a numpy array which indicates the handedness (left or rigth) of the tracked hands.
#           ***WARNING*** : Depending on wether the video is captured with the camera placed in front of the subject or
#                           in an egocentric point of view, the handedness can be invereted.
#           - hands_landmarks is a numpy array which for each tracked hand contains 21 hand landmarks, each composed of
#           x, y and z coordinates. The x and y coordinates are normalized to [0.0, 1.0] by the image width and height,
#           respectively. The z coordinate represents the landmark depth, with the depth at the wrist being the origin.
#           The smaller the value, the closer the landmark is to the camera. The magnitude of z uses roughly the same scale as x.
#           - hands_world_landmarks is a numpy array which for each tracked hand contains 21 hand landmarks are also
#           presented in world coordinates. Each landmark is composed of x, y, and z, representing real-world
#           3D coordinates in meters with the origin at the hand’s geometric center.
########################################################################################################################
    def draw_and_export_data(self, image, draw=True, export=True, height=1080, width=1920) :
    
        # A nested function that converts a list of landmarks to a numpy array and can perform a denormalization task if asked
        def landmarks_list_to_array(landmark_list, image_shape, denormalization):
          
            if denormalization :
                cols, rows, _ = image_shape
                return np.asarray([(lmk.y * rows, lmk.x * cols, lmk.z)
                             for lmk in landmark_list.landmark])
            else :
                return np.asarray([(lmk.y, lmk.x, lmk.z)
                             for lmk in landmark_list.landmark])                
            
        
        if export :
            hands_landmarks = None
            hands_world_landmarks = None
            hands_sides = np.full((6), None)
            hands_scores = np.full((6), 0.0)
        else :
            hands_landmarks = None
            hands_world_landmarks = None
            hands_sides = None
            hands_scores = None
        
        # Draw the hand annotations on the image if the hand tracking results are available
        if self.hands_results.multi_hand_landmarks and self.hands_results.multi_hand_landmarks and self.hands_results.multi_handedness :
            
            min_id = [] # Initialize a list to store the index of hands with low scores
        
            # Extract hand sides and scores if exporting data
            if export :
                for idx, hand_handedness in enumerate(self.hands_results.multi_handedness):
                    hands_sides[idx] = hand_handedness.classification[0].label
                    hands_scores[idx] = hand_handedness.classification[0].score
                
                # Remove hands with the lowest scores in the scores and sides array to match the desired number of hands
                excess = np.count_nonzero(hands_scores)
                while excess > self.max_num_hands :
                    min_id.append(np.argmin(hands_scores))
                    np.delete(hands_scores, min_id)
                    np.delete(hands_sides, min_id) 
                    excess = np.count_nonzero(hands_scores)

                hands_scores = np.trim_zeros(hands_scores)
                hands_sides = np.resize(hands_sides, np.shape(hands_scores))
                
                hands_landmarks = np.full((len(hands_scores), 21, 3), 0.0)
                hands_world_landmarks = np.full((len(hands_scores), 21, 3), 0.0)
                
            id = 0
            for idx, (landmarks, world_landmarks, handedness) in enumerate( zip(self.hands_results.multi_hand_landmarks, self.hands_results.multi_hand_world_landmarks, self.hands_results.multi_handedness) ) :
                # Skip hands with low scores based on min_id list                
                if id not in min_id :
                    if draw :
                        # Draw hands landmarks and connections on the image based on the landmarks array
                        if handedness.classification[0].label == "Right" :
                            mp_drawing.draw_landmarks(image,
                                                landmarks,
                                                mp_hands.HAND_CONNECTIONS,
                                                mp_drawing_styles.get_default_left_hand_landmarks_style(),
                                                mp_drawing_styles.get_default_left_hand_connections_style())
                        elif handedness.classification[0].label == "Left" :
                            mp_drawing.draw_landmarks(image,
                                                landmarks,
                                                mp_hands.HAND_CONNECTIONS,
                                                mp_drawing_styles.get_default_right_hand_landmarks_style(),
                                                mp_drawing_styles.get_default_right_hand_connections_style())
                
                    if export :
                        # Convert hand landmarks and world landmarks from list to numpy arrays for export
                        array_hand_landmarks = landmarks_list_to_array(landmarks,\
                                                                      (width,  height, 3), True)
                        hands_landmarks[id] = array_hand_landmarks

                        array_hand_world_landmarks = landmarks_list_to_array(world_landmarks,\
                                                                      (width,  height, 3), False)
                        hands_world_landmarks[id] = array_hand_world_landmarks

                    id += 1
                    
        return hands_scores, hands_sides, hands_landmarks, hands_world_landmarks