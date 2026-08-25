import webcam
import cv2
import mediapipe as mp 

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

#initalizes the webcam and then finds the frame height
#Saves it inside a tuple
camera, height, width = webcam.webcam_init()
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while camera.isOpened():
            #ret is a bool based on if a frame was captured
            #frame is a numpy array of the image
            success, frame = camera.read() 
            
            # ignores a frame if it empty
            if not success:
                continue
            
            # Label frames as immutable so we can pass by reference instead of value
            image = frame
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Shows a window for the webcam feed
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            cv2.imshow("Webcam Feed", frame)
            
            # checks for a keypress of the 'q' key to quit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Draw the markers on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )
                
                
    
camera.release()
cv2.destroyAllWindows()