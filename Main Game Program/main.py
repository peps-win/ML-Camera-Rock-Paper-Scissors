import webcam # Assumes your custom wrapper works perfectly
from shared.finger_location import finger_locator
import cv2
import mediapipe as mp # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import numpy as np
from shared.fingerNames import joints
import os


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Defines what each tensor means outputted from MLP
labels = ["Rock", "Paper", "Scissors"]

# Initializes the webcam and returns camera object
camera, height, width = webcam.webcam_init()

# Define the Ai class
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.output(x)
        return x

# Define the Ai Architecture set when training
input_dim = 63
hidden_dim = 32
output_dim = 3

# Define the model
model = MLP(input_dim, hidden_dim, output_dim)

# Load the saved model weights into memory
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rps_model.pth")
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

WRIST_INDEX = 0 # index of wrist inside of the joints list
SCALE_JOINT_INDEX = 8 # index of Middle Finger MCP inside of the joints list

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.75,
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
            
            # Loop for drawing onto hand if a hand that needs landmarks is present
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    joint_coordinates = []
                    
                    for joint in joints:
                        x, y, z = finger_locator(joint, hand_landmarks, width, height)
                        joint_coordinates.append(x)
                        joint_coordinates.append(y)
                        joint_coordinates.append(z)

                    coords = np.array(joint_coordinates, dtype = np.float32).reshape(-1, 3)
                    
                    wrist = coords[WRIST_INDEX].copy()
                    coords-=wrist
                    
                    scale = float(np.linalg.norm(coords[SCALE_JOINT_INDEX]))
                    if scale < 1e-6:
                        scale = 1e-6
                    coords /= scale
                    
                    joint_list = coords.flatten().tolist()
                    
                    # Convert the normalizes joint list
                    input_tensor = torch.from_numpy(np.array(joint_list, dtype=np.float32)).unsqueeze(0)
                    
                    # Feed the input_tensor to the model
                    model.eval()
                    with torch.no_grad():
                        output = model(input_tensor)
                    
                    predicted_class = torch.argmax(output, dim=1).item()
                    print(f"Prediction: {labels[predicted_class]} (raw: {output.tolist()})")
                    
                    mp_drawing.draw_landmarks(
                        frame,  # Draw directly onto frame
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            # Display the frame AFTER drawing annotations
            cv2.imshow("Webcam Feed", frame)
            
            # Checks for a keypress of the 'q' key to quit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
camera.release()
cv2.destroyAllWindows() # Clean up window assets on exit
