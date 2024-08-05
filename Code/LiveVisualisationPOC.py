#!/usr/bin/env python
import sys
import logging
import os
import argparse
import time
import readchar
import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
from scipy.interpolate import griddata
from pprint import pformat
import SmartSoleBle as sble
from PIL import Image


"""
DataPackage:
    head: int
    mac: str
    deviceTimeStamp: int 
    date: str 
    imu_data: dict
    ps_data: list[int]

"""
def initLogging(name: str, loglevel: int) -> logging.Logger:
    """
        initialize logger:
            on linux:  log to stderr
            on windows: log to %APPDATA%/appname/ble_smartsole.log
        
        :param name: name of the logger
        :return: return initialized logger

    """

    log = logging.getLogger(name)
    log.propagate = False

    handler = None

    if sys.platform == "win32":
        appdata = os.path.expandvars("%APPDATA%")
        logfile = os.path.join(appdata, name, "ble_smartsole.log")
        
        try:
            os.mkdir(os.path.dirname(logfile))
        except FileExistsError:
            pass

        handler = logging.FileHandler(logfile, mode="w")
    else:
        handler = logging.StreamHandler(stream=sys.stdout)

    formatter = logging.Formatter("%(name)s (%(filename)s at Line %(lineno)d): %(message)s")

    handler.setFormatter(formatter)
    log.addHandler(handler)
    
    log.setLevel(loglevel)
        
    return log

def main(loglevel):


    #
    # Connection Setup
    # --------------------------------------------------------------------------------------------
    # Connecting the devices via BLE and initiating logging
    #
    log = initLogging("SmartSoleConnector", loglevel)
    ble = sble.BLE(log) 
    log.debug("test function starting")
    log.debug("start to scan for devices")
    scan_begin = time.time()
    ble.startScan(5)
    scan_end = time.time()
    sc_result = ble.getScanResult()
    log.debug(f"scanned for {scan_end - scan_begin} and found {len(sc_result)} devices")
    log.debug("connected to devices")
    ble.connectDevices()
    log.debug(f"connection sucess ? new ConnectionStatus is: {ble.getConnectionStatus()}")
    log.debug("setup characteristics!")
    ble.setupServicesCharacteristic()
    ble.setPackageHeadToReceive([23])
    log.debug("Devices have been configured")



    #
    # Visualisation Settings
    # --------------------------------------------------------------------------------------------
    # Settings Relevant to the Visualisation
    #   spacing: Space between left and right shoe in visualisation
    #   max_value: maximum expected pressure value in the visualisation
    #   IDs: mac adresses of left and right sole
    #
    spacing = 225
    max_value = 3
    ID_left = 'C6:22:F2:44:38:49'
    ID_right = 'FE:F7:D0:12:2E:8D'

    left_image_name = 'shoe_outline_left_with_sensors.png'
    right_image_name = 'shoe_outline_right_with_sensors.png'
    image_path = 'Images/'




    #
    # Image Setup
    #  ------------------------------------------------
    #   All of the one time setup before the visualisation
    #
    full_image_path_left = image_path + left_image_name
    full_image_path_right = image_path + right_image_name

    shoe_outline_right = Image.open(full_image_path_right)
    shoe_outline_left = Image.open(full_image_path_left)



    sensor_locations_pixels_left = np.array([(145, 513), (200, 200), (195, 78), (135, 190), 
                                        (83, 508), (85, 345), (65, 213), (103, 93)])


    roi_left, roi_top, roi_right, roi_bottom = (0, 0, shoe_outline_left.width, shoe_outline_left.height)

    x_grid_roi = np.linspace(roi_left, roi_right, 100)
    y_grid_roi = np.linspace(roi_top, roi_bottom, 100)
    x_mesh_roi, y_mesh_roi = np.meshgrid(x_grid_roi, y_grid_roi)

    sensor_locations_pixels_left[:, 1] = roi_bottom - sensor_locations_pixels_left[:, 1]

    corners = [(roi_left, roi_top), (roi_left, roi_bottom), (roi_right, roi_top), (roi_right, roi_bottom)]
    corners = np.array([(x, roi_bottom - y) for x, y in corners])  


    x_min = min(sensor_locations_pixels_left[:, 0])
    x_max = max(sensor_locations_pixels_left[:, 0])

    x_range = x_max - x_min
    x_min_new = x_min - x_range + 40
    x_max_new = x_max + x_range - 40

    corners[:, 0] = np.array([x_min_new, x_min_new, x_max_new, x_max_new])

    midpoint1_x = (corners[0][0] + corners[1][0]) / 2  
    midpoint2_x = (corners[2][0] + corners[3][0]) / 2  
    midpoint_y = (roi_bottom + roi_top) / 2  


    midpoint1 = (midpoint1_x, midpoint_y)
    midpoint2 = (midpoint2_x, midpoint_y)


    corners_with_midpoints = np.concatenate([corners, [midpoint1, midpoint2]])


    sensor_locations_pixels_left = np.concatenate([sensor_locations_pixels_left, corners_with_midpoints])

    sensor_locations_pixels_right = np.array([(182, 510), (180, 345), (200, 213), (161, 92), (135, 185), 
                                            (120, 515), (65, 200), (72, 77)])

    roi_left, roi_top, roi_right, roi_bottom = (0, 0, shoe_outline_right.width, shoe_outline_right.height)

    x_grid_roi = np.linspace(roi_left, roi_right, 100)
    y_grid_roi = np.linspace(roi_top, roi_bottom, 100)
    x_mesh_roi, y_mesh_roi = np.meshgrid(x_grid_roi, y_grid_roi)

    sensor_locations_pixels_right[:, 1] = roi_bottom - sensor_locations_pixels_right[:, 1]

    corners = [(roi_left, roi_top), (roi_left, roi_bottom), (roi_right, roi_top), (roi_right, roi_bottom)]
    corners = np.array([(x, roi_bottom - y) for x, y in corners])  


    x_min = min(sensor_locations_pixels_right[:, 0])
    x_max = max(sensor_locations_pixels_right[:, 0])

    x_range = x_max - x_min
    x_min_new = x_min - x_range + 40
    x_max_new = x_max + x_range - 40

    corners[:, 0] = np.array([x_min_new, x_min_new, x_max_new, x_max_new])

    midpoint1_x = (corners[0][0] + corners[1][0]) / 2  
    midpoint2_x = (corners[2][0] + corners[3][0]) / 2  
    midpoint_y = (roi_bottom + roi_top) / 2  


    midpoint1 = (midpoint1_x, midpoint_y)
    midpoint2 = (midpoint2_x, midpoint_y)

    corners_with_midpoints = np.concatenate([corners, [midpoint1, midpoint2]])
    sensor_locations_pixels_right = np.concatenate([sensor_locations_pixels_right, corners_with_midpoints])

    shoe_outline_right = cv2.imread(full_image_path_right, cv2.IMREAD_UNCHANGED)
    shoe_outline_left = cv2.imread(full_image_path_left, cv2.IMREAD_UNCHANGED)

    #
    # Visualisation
    # -----------------------------------------------------------------------------
    # Actual visualisation
    #
    #

    q = ble.getData()
    time.sleep(1)
    log.info(f"queue len is: {q.getQueueSize()}") 
    
    last_left = np.array([0,0,0,0,0,0,0,1,0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    last_right = np.array([0,0,0,0,0,0,0,1,0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cnt: int = 0

    while True:
        q = ble.getData()
        if q.peek() != None:
            
            data = q.get()
            log.info(f'{pformat(data)}')

            if(data.mac == ID_left):
                data_left = data.ps_data + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                data_right = last_right
                last_left = data_left
            elif(data.mac == ID_right):
                data_right = data.ps_data + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                data_left = last_left
                last_right = data_right

            
            interpolated_pressure_roi_left = griddata(sensor_locations_pixels_left, data_left,
                                                      (x_mesh_roi, y_mesh_roi), method='cubic')
            plt.imshow(interpolated_pressure_roi_left, cmap='coolwarm', origin='lower', vmax=max_value,
                       extent=(roi_left, roi_right, roi_top, roi_bottom))
            plt.imshow(shoe_outline_left, extent=(roi_left, roi_right, roi_top, roi_bottom), alpha=1)
            plt.axis("off")
            buf_left = io.BytesIO()
            plt.savefig(buf_left, format='png')
            buf_left.seek(0)
            heatmap_img_left = cv2.imdecode(np.frombuffer(buf_left.getvalue(), dtype=np.uint8), 1)
            buf_left.close()
            plt.clf()

            
            interpolated_pressure_roi_right = griddata(sensor_locations_pixels_right, data_right,
                                                       (x_mesh_roi, y_mesh_roi), method='cubic')
            plt.imshow(interpolated_pressure_roi_right, cmap='coolwarm', origin='lower', vmax=max_value,
                       extent=(roi_left, roi_right, roi_top, roi_bottom))
            plt.imshow(shoe_outline_right, extent=(roi_left, roi_right, roi_top, roi_bottom), alpha=1)
            plt.axis("off")
            buf_right = io.BytesIO()
            plt.savefig(buf_right, format='png')
            buf_right.seek(0)
            heatmap_img_right = cv2.imdecode(np.frombuffer(buf_right.getvalue(), dtype=np.uint8), 1)
            buf_right.close()
            plt.clf()

            
            heatmap_img_left = heatmap_img_left[:, :-spacing]
            heatmap_img_right = heatmap_img_right[:, spacing:]
            height = max(heatmap_img_left.shape[0], heatmap_img_right.shape[0])
            heatmap_img_left = cv2.resize(heatmap_img_left, (int(heatmap_img_left.shape[1] * height / heatmap_img_left.shape[0]), height))
            heatmap_img_right = cv2.resize(heatmap_img_right, (int(heatmap_img_right.shape[1] * height / heatmap_img_right.shape[0]), height))

            combined_image = cv2.hconcat([heatmap_img_left, heatmap_img_right])
            cv2.imshow('Live Video Feed', combined_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            cv2.waitKey(10)

    cv2.destroyAllWindows


    ble.disconnectDevices()

    print(f"cnt: {cnt}")
    print("Press Any Key To Exit")

    k = readchar.readchar()   	
    
    print("Exit")
    sys.exit()
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Connect with Pacemaker smart soles and create data recording')
    parser.add_argument('-v','--verbose',help='enable verbose option', action='store_const',dest="loglevel", const=logging.INFO,default=logging.ERROR)
    parser.add_argument('-d','--debug',help='enable debug output', action='store_const',dest="loglevel", const=logging.DEBUG,default=logging.ERROR)

    args = parser.parse_args()

    main(args.loglevel)