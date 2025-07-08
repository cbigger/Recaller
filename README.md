# 🪃 Recaller 
A python library that combines person detection and facial recognition to "pick out" a specific person from a live video feed.
The ultimate goal of this is to provide an additional path by which mobile agentic systems can perform recall, like a friendly dog or parrot or something.

### What I'm Working On
#### Near
- Add live feed recognition labelling i.e. make the face_worker actually work
- Proper packaging and completion of single point of entry, still have oddly packaged run_stub file. Should be renamed to `run.py` and handle all arguments
#### Far
- Incorporate **dual-movement tracking**: track target with a moving camera and a moving recall target
- Extend target packaging to work with a directory of images that have multiple people in them, isolating the one that is common across them all. Must remember to include option to force targetter to choose one of the extracted faces. Profile shots are also a real problem and might need their own detection layer or something.
- Add target packaging mode to the live feed by disabling the recognition worker and adding a packaging worker. This dude will chop faces out every so many frames that they are captured and fill up a directory that can then be used to isolate individual faces from the boxes.


## OVERVIEW
The main part of this program is an opencv and Darknet NN based live video analyzer that will detect bodies and faces and pass those faces against a target face encoding. There's also a target packaging script which can create the face encoding from still images, and which also includes the ability to test the encoding against offtarget test references built in the same manner.


## INSTALLATION
1. Clone the repository
2. Install the requirements (opencv-contrib-python is the only external one atm)
3. Test your install with run.py

This repository includes everything needed for this to work out of the box, including the model weights and examples in the targetting folders for using the target packager script.


## USAGE
When you invoke the run_system() function from the coordinator.py script, the program will check your linux box for camera feeds, and offer a list to choose from. You can set a default camera in the config.py file.

After selecting a camera, you should see the program start in the console. If you ran run.py with no arguments, then a successful run will simply sit there in your window. You can run `--debug` to have the two workers (detection and recognition) print their results to console, and you can run `--show_feed` to be shown the camera view complete with bounding boxes for detections and labels for recognition scores that pass the threshold.

