import cv2
import mediapipe as mp # type: ignore
import csv
import os
from pathlib import Path
from dataclasses import fields
from shared.models import hand, joint
from shared.fingerNames import joints
from shared.finger_location import finger_locator

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

DATA_DIRECTORY = input("What is the dataset directory")
TRAIN_DATAPATH = os.path.join(DATA_DIRECTORY, "train")
ROCK_DATAPATH = os.path.join(TRAIN_DATAPATH, "rock")
PAPER_DATAPATH = os.path.join(TRAIN_DATAPATH, "paper")
SCISSORS_DATAPATH = os.path.join(TRAIN_DATAPATH, "scissors")

LABELS = {0: "rock", 1: "paper", 2: "scissors"}

def build_header():
    header = ["label", "filename"]
    for f in fields(hand):
        joint_name = f.name
        header += [f"{joint_name}_x", f"{joint_name}_y", f"{joint_name}_z"]
    return header

header = build_header()

# Open the CSV once for the whole run, in append-safe write mode
# More efficent than opening twice
with open('data.csv', 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header)

    for i in range(3):
        match i:
            case 0:
                currentPath = ROCK_DATAPATH
            case 1:
                currentPath = PAPER_DATAPATH
            case 2:
                currentPath = SCISSORS_DATAPATH

        label = LABELS[i]

        for entry in Path(currentPath).iterdir():
            if not entry.is_file():
                continue

            with mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5) as hands:

                image = cv2.flip(cv2.imread(str(entry)), 1)
                if image is None:
                    continue  # skip unreadable files

                height, width, _ = image.shape
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        joint_list = []
                        for id in joints:
                            x, y, z = finger_locator(id, hand_landmarks, width, height)
                            joint_list.extend([x, y, z])

                        row = [label, entry.name] + joint_list
                        writer.writerow(row)