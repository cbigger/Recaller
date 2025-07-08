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
## Facial Recognition Model - right now is built into cv2
TARGET_FACE_PATH = "target/trained_target.xml"
#RECOG_CFG_PATH = "models/people-r-people.cfg"
#RECOG_WEIGHTS_PATH = "models/people-r-people.weights"
#RECOG_NAMES_PATH = "models/people-r-people.names"
# TARGETTER CONFIGURATION
INPUT_DIR = "target/raw"        # Directory with full unprocessed images of target
OUTPUT_DIR = "target/processed"      # Where to save the detected and cropped faces of the target
OFFTARGET_INPUT_DIR = "offtarget/raw"  # same as above but for images that well be used for testing recog rates.
OFFTARGET_OUTPUT_DIR = "offtarget/processed"
CONFIDENCE_THRESHOLD = 0.5
FACE_CLASS_ID = 1
