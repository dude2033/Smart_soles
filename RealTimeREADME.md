# Smart_soles real time data visualisation

Folders relevant to realtime visualisation:

**Images**
Base images for the for the video and overlays.

**Code**
Relevant file to Realtime data visualisation is ***LiveVisualisationPOC.py***


## Libraries

**cv2**
Python implementation of OpenCV. Used for video rendering and image processing

**numpy**
used for vector implementation and array modification.

**matplotlib**
Used for image modification.

**scipy.interpolate**
Used for gridinterpolation of sensor data.

**SmartSoleBle**
Provided by Pacemaker Technologies. Used for connecting with the smart sole hardware and tranferring data from it to the visualisation software. Also handles debug logging.

**PIL**
Used for image importing.



## Code

**Connection Setup**
The debug logger and Bluetooth connection with the smart sole hardware are established. Type of datapackages to be received is set here.

**Visualisation Settings**
The parameters for base images, spacing between images and IDs for smart sole hardware are set here. Furthermore the expected max value in the pressure data is set here. 

**Image Setup**
All preperatory calculations are done here. Setting up region of influence, setting coordinates for sensors and referrence points for interpolation calculation, Setting min and max values. 

**Visualisation**
The visualisation loop. Every run through this loop is one frame in the visualisation. Every entry in the data queue is only for a single sole so in the loop it is first determined which of the soles is being updated. The the heatmap is interpolated and the base image is overlayed. lastly it is combined with the last updated image of the other sole to have a complete image and fed into the video buffer.