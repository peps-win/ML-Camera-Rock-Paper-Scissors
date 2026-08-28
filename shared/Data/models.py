from dataclasses import dataclass

# This is a basis of all dataclasses we use in python

@dataclass
class joint:
    x: float
    y: float
    z: float

@dataclass
class hand:
    wrist: joint
    thumb_cmc: joint
    thumb_mcp: joint
    thumb_ip: joint
    thumb_tip: joint
    indx_fing_mcp: joint
    indx_fing_pip: joint
    indx_fing_dip: joint
    indx_fing_tip: joint
    middle_fing_mcp: joint
    middle_fing_pip: joint
    middle_fing_dip: joint
    middle_fing_tip: joint
    ring_fing_mcp: joint
    ring_fing_pip: joint
    ring_fing_dip: joint
    ring_fing_tip: joint
    pinky_mcp: joint
    pinky_pip: joint
    pinky_dip: joint
    pinky_tip: joint
