import mediapipe as mp 

def finger_locator (joint, hand_landmarks, width, height):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands


    finger = mp_hands.HandLandmark.joint
    landmark = hand_landmarks.landmark[finger]
   
    x = int(landmark.x *  width)
    y = int(landmark.y *  height)
    
    return x, y
