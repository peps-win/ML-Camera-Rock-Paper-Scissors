import cv2

def webcam_init ():
    cam = cv2.VideoCapture(0)
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_hieght = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cam, frame_hieght, frame_width

