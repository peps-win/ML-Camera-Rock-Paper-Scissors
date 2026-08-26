import mediapipe as mp # type: ignore

def finger_locator (joint, hand_landmarks, width, height):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands


    marker = mp_hands.HandLandmark[joint]
    landmark = hand_landmarks.landmark[marker]
   
    x = int(landmark.x *  width)
    y = int(landmark.y *  height)
    z = landmark.z
    
    return x, y, z
