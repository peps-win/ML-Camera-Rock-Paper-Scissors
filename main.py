import webcam # Assumes your custom wrapper works perfectly
from finger_location import finger_locator
import cv2
import mediapipe as mp # type: ignore

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Initializes the webcam and returns camera object
camera, height, width = webcam.webcam_init()

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.9,
        min_tracking_confidence=0.85
    ) as hands:
        
        while camera.isOpened():
            success, frame = camera.read() 
            
            if not success:
                continue
            
            # 1. Convert to RGB for MediaPipe processing
            # We treat 'frame' directly to keep memory references clean
            frame.flags.writeable = False
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_image)
            
            # 2. Allow drawing back on the original frame
            frame.flags.writeable = True
            
            # Loop for drawing onto hand if a hand that needs landmarks drawn on it is present
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    
                    indx_x, indx_y = finger_locator("INDEX_FINGER_TIP", hand_landmarks, width, height)
                    thumb_x, thumb_y = finger_locator("THUMB_TIP", hand_landmarks, width, height)
                    # print(f"Index Finger Tip X:{indx_x} Y:{indx_y}")
                    # print(f"Thumb Finger Tip X:{thumb_x} Y:{thumb_y}")
                    
                    if abs(indx_x - thumb_x) < 2:
                        if abs(indx_y - thumb_y) < 25:
                            print("Thumb and Index Finger touching")
                    
                    mp_drawing.draw_landmarks(
                        frame,  # Draw directly onto frame
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            # 4. CRITICAL FIX: Display the frame AFTER drawing annotations
            cv2.imshow("Webcam Feed", frame)
            
            # Checks for a keypress of the 'q' key to quit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
camera.release()
cv2.destroyAllWindows() # Clean up window assets on exit
