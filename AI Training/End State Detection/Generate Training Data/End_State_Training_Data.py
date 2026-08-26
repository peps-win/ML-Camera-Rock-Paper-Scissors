import cv2
import mediapipe as mp # type: ignore
import csv
import sys
import os
from pathlib import Path
from dataclasses import fields
from shared.models import hand, joint

# Setup Paths
DATA_DIRECTORY = input("What is the dataset directory")
TRAIN_DATAPATH = os.path.join(DATA_DIRECTORY, "train")
ROCK_DATAPATH = os.path.join(DATA_DIRECTORY, "rock")
PAPER_DATAPATH = os.path.join(DATA_DIRECTORY, "paper")
SCISSORS_DATAPATH = os.path.join(DATA_DIRECTORY, "scissors")

# Builds the header row from my 
def build_header():
    header = ["label", "filename"]
    for f in fields(hand):
        joint_name = f.name
        header += [f"{joint_name}_x", f"{joint_name}_y", f"{joint_name}_z"]
    return header

header = build_header()
print(header)