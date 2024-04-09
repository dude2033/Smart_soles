import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from PIL import Image
import pandas as pd


shoe_outline_image = Image.open('shoe_outline_right.png')

df = pd.read_csv('Data/pt_2024-02-15 13:30:08.619670.csv')

df_filtered = df[df['data'].str.contains("PS: \[.*\]")]

sensor_locations_pixels = np.array([(190, 535), (200, 345), (200, 210), (145, 100),
                                    (120, 160), (120, 535), (50, 160), (65, 85)])

roi_left, roi_top, roi_right, roi_bottom = (0, 0, shoe_outline_image.width, shoe_outline_image.height)


x_grid_roi = np.linspace(roi_left, roi_right, 100)
y_grid_roi = np.linspace(roi_top, roi_bottom, 100)
x_mesh_roi, y_mesh_roi = np.meshgrid(x_grid_roi, y_grid_roi)

sensor_locations_pixels[:, 1] = roi_bottom - sensor_locations_pixels[:, 1]

corners = [(roi_left, roi_top), (roi_left, roi_bottom), (roi_right, roi_top), (roi_right, roi_bottom)]
corners = np.array([(x, roi_bottom - y) for x, y in corners])  


x_min = min(sensor_locations_pixels[:, 0])
x_max = max(sensor_locations_pixels[:, 0])

x_range = x_max - x_min
x_min_new = x_min - x_range
x_max_new = x_max + x_range

corners[:, 0] = np.array([x_min_new, x_min_new, x_max_new, x_max_new])

sensor_locations_pixels = np.concatenate([sensor_locations_pixels, corners])




pressure_readings = np.array([2.56, 2.28, 2.43, 2.26, 2.23, 1.97, 1.95, 1.94])

pressure_readings_normalized = (pressure_readings - np.min(pressure_readings)) / (np.max(pressure_readings) - np.min(pressure_readings))

corner_pressures = np.zeros(4)
pressure_readings_normalized = np.concatenate([pressure_readings_normalized, corner_pressures])


interpolated_pressure_roi = griddata(sensor_locations_pixels, pressure_readings_normalized,
                                     (x_mesh_roi, y_mesh_roi), method='cubic')


plt.imshow(interpolated_pressure_roi, cmap='coolwarm', origin='lower',
           extent=(roi_left, roi_right, roi_top, roi_bottom))

plt.imshow(shoe_outline_image, extent=(roi_left, roi_right, roi_top, roi_bottom), alpha=1)

plt.colorbar(label='Pressure')

plt.show()
