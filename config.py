# CAMERA SETTINGS
FACE_QUEUE_SIZE = 5
DETECT_QUEUE_SIZE = 5
FRAME_INTERVAL = 1  # seconds or frame skip
MAX_CAM_DEVICES = 10
DEFAULT_CAMERA = None  # Set to an integer like 0 to skip selection prompt
# MODEL CONFIGURATION
## Body and Face Detection
DETECT_CFG_PATH = "models/people-r-people.cfg"
DETECT_WEIGHTS_PATH = "models/people-r-people.weights"
DETECT_NAMES_PATH = "models/people-r-people.names"
## Facial Recognition Model
RECOG_CFG_PATH = "models/people-r-people.cfg"
RECOG_WEIGHTS_PATH = "models/people-r-people.weights"
RECOG_NAMES_PATH = "models/people-r-people.names"
# TARGET CONFIGURATION
## Sooooooooo this is I guess where we would be loading the "owner" data for the facial recog.
## which i think would be based on the recog so might end up putting it there.
