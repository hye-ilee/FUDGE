######################################################################################
# config.py
# This module contains configuration settings for the FUDGE application.
# It includes paths for video, music, and motion data directories, as well as constants
# for FPS, segment length, motion commands, and joint names.
######################################################################################

MODIFIED_VIDEO_DIR = "/root/CS470/LODGE/cs470/data/modified_videos"
MUSIC_DIR = "/root/CS470/LODGE/data/finedance/new_music" # Added in this path since we couldn't resolve path issues
MOTION_DIR =  "/root/CS470/LODGE/cs470/data/original_npy"
ORIGINAL_VIDEO_DIR = "/root/CS470/LODGE/cs470/data/original_videos"

FPS = 30
SEGMENT_LEN = 256

MOTION_COMMANDS = [
    "turn left", "turn right", "bend forward", "bend backward", "tilt left", "tilt right",
    "bend torso forward", "bend torso back", "twist torso left", "twist torso right",
    "chest tilt left", "chest tilt right", "nod head up", "nod head down",
    "shake head left", "shake head right", "tilt head left", "tilt head right",
    "raise left arm", "lower left arm", "raise right arm", "lower right arm",
    "lift left leg", "lift right leg", "kick left", "kick right",
    "twist left wrist CW", "twist right wrist CW"
]

SMPL_JOINT_NAMES = [
    "root", "lhip", "rhip", "belly", "lknee", "rknee", "spine",
    "lankle", "rankle", "chest", "ltoes", "rtoes", "neck", "linshoulder", "rinshoulder",
    "head", "lshoulder", "rshoulder", "lelbow", "relbow", "lwrist", "rwrist"
]

ROTATION_AXES = [
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
]

ROTATION_COMMANDS = {
    "turn left": (0, 2), "turn right": (0, 3),
    "bend forward": (3, 0), "bend backward": (3, 1),
    "tilt right": (3, 4), "tilt left": (3, 5),
    "bend torso forward": (6, 0), "bend torso back": (6, 1),
    "twist torso left": (9, 2), "twist torso right": (9, 3),
    "chest tilt right": (9, 4), "chest tilt left": (9, 5),
    "nod head down": (12, 0), "nod head up": (12, 1),
    "shake head left": (12, 2), "shake head right": (12, 3),
    "tilt head right": (12, 4), "tilt head left": (12, 5),
    "raise left arm": (16, 4), "lower left arm": (16, 5),
    "raise right arm": (17, 5), "lower right arm": (17, 4),
    "lift left leg": (1, 1), "lift right leg": (2, 1),
    "kick left": (4, 0), "kick right": (5, 0),
    "twist left wrist CW": (20, 2), "twist right wrist CW": (21, 2)
}

VIDEO_CHOICES = {
    1: ("Pink Venom", "hiphop"),
    2: ("Waves", "Krump"),
    3: ("Do It!", "House"),
    4: ("Take Me To Church", "Jazz Ballet"),
    5: ("Misfit", "Breaking"),
    6: ("Mamacita", "Street Jazz"),
    7: ("Violet", "hiphop"),
    8: ("Born This Way", "waacking"),
    9: ("Feels This Good", "locking"),
    10: ("Bra Bra", "popping"),
    11: ("The Monkey Whistle", "Breaking"),
    12: ("Got Ya Money", "Locking")
}