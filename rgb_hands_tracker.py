import numpy as np
import cv2 as cv

import drawing_styles as mp_drawing_styles
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands

class RGB_Hands_Tracker():
    def __init__(self, num_hands=2, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        super(RGB_Hands_Tracker, self).__init__()

        self.tracker = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=num_hands,
                        model_complexity=model_complexity,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence)
        
        self.max_num_hands = num_hands
        self.hands_results = None
        

        
    def update(self, image) :
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.hands_results = self.tracker.process(image)            
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)



    def draw_and_export_data(self, image, draw=True, export=True, height=1080, width=1920) :
    
    
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
        
        # Draw the hand annotations on the image.            
        if self.hands_results.multi_hand_landmarks and self.hands_results.multi_hand_landmarks and self.hands_results.multi_handedness :
            
            min_id = []                    
        
            if export :
                for idx, hand_handedness in enumerate(self.hands_results.multi_handedness):
                    hands_sides[idx] = hand_handedness.classification[0].label
                    hands_scores[idx] = hand_handedness.classification[0].score
                
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
                if id not in min_id :
                    if draw :
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
                        array_hand_landmarks = landmarks_list_to_array(landmarks,\
                                                                      (width,  height, 3), True)
                        hands_landmarks[id] = array_hand_landmarks

                        array_hand_world_landmarks = landmarks_list_to_array(world_landmarks,\
                                                                      (width,  height, 3), False)
                        hands_world_landmarks[id] = array_hand_world_landmarks

                    id += 1
                    
        return hands_scores, hands_sides, hands_landmarks, hands_world_landmarks