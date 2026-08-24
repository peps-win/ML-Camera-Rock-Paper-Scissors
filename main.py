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
        model_complexity = 0
        min_detection_confidence = 0.5
        min_tracking_confidence = 0.5
    ) as hands :
        while cam.isOpened():
            #ret is a bool based on if a frame was captured
            #frame is a numpy array of the image
                success, frame = camera.read() 
                
                # ignores a frame if it empty
                if not success:
                    continue
                
                # Label frames as immutable so we can pass by reference instead of value
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Shows a window for the webcam feed
                cv2.imshow("Webcam Feed", frame)
                
                # checks for a keypress of the 'q' key to quit the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
camera.release()
cv2.destroyAllWindows