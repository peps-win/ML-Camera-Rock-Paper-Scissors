import webcam
import cv2

#initalizes the webcam and then finds the frame height
#Saves it inside a tuple
camera, height, width = webcam.webcam_init()

while True:
    #ret is a bool based on if a frame was captured
    #frame is a numpy array of the image
    ret, frame = camera.read() 
    
    # Shows a window for the webcam feed
    cv2.imshow("Webcam Feed", frame)
    
    # checks for a keypress of the 'q' key to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
camera.release()
cv2.destroyAllWindows