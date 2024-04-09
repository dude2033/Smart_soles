# Smart_soles

The Directory is split into four main folders:

**Data**
Put all csv and other datafiles in here

**Images**
Base images for the animation and testing. You should not need to touch this. Both with and without sensors are currently in here.

**Output**
All output files, whther temporary or otherwise are put in here

**Code**
All code files are in here, some for testing. The most recent working file is ***HeatmapAnimation.ipynb***.


## Current state

**Imports**
All imports go here

**IDs and Data-Path**
This is the code cell that needs changes if you want to change input or output paths, use different data or different base pictures.

**Data Pruning**
The data is prepared reformating the tables to synchronise the two soles and refactoring the string types to floats. Also the max value in the data is extracted to base the heatmap of.

**Region settup and sensor locations**
The sensor locations are currently set for the the images in the Image folder and would need to be manually redone for other images to fit the dimensions.

**Dimension Image**
A first image to base the video dimensions is generated here and saved into the Output folder. It uses generic datapoints where at least one is not zero to avoid a divide by zero error.

**Creating the Video**
The video is created frame by frame. Every frame is destroyed immediately to save space. The "spacing" variable defines the distance between the two images and is inverse meaning a larger value brings the images closer together.

## TODO
Adaptive Framerate

Better synchronisation