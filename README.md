Mission Space Lab is a free space science task designed for young people, particularly those aged 19 and under. Teams of 2-6 participants, supervised by a mentor, can write Python programs to solve a scientific task on the International Space Station (ISS). The goal is to calculate the speed at which the ISS is traveling by using the Astro Pi computers' sensors or camera to gather data about the ISS's orientation and motion.

Participants will learn about the ISS, data gathering, and computer programming. Successful programs will be deployed on the ISS, and teams will receive certificates and data collected in space. The mission runs from September 16, 2024, to February 24, 2025.

## Introduction
The program is designed to calculate the speed of the ISS by capturing images and gathering data using the Astro Pi's sensors. The main steps involved in the program are:

1. **Image Capture**: The program uses the PiCamera to capture images at regular intervals.
2. **Data Logging**: The SenseHat sensors are used to gather data such as temperature, pressure, humidity, and orientation. This data is logged along with the timestamps and ISS coordinates.
3. **Image Processing**: The captured images are processed to identify features and calculate distances between them.
4. **Speed Calculation**: The program calculates the Ground Sample Distance (GSD) and uses it to estimate the linear distance between images. The speed of the ISS is then calculated based on the distance and time difference between image captures.
5. **Cloud Classification**: The program uses a TensorFlow Lite model with Edge TPU to classify cloud types in the images and apply corrections to the GSD for more accurate distance measurements.

## Project Structure:
- `main.py`: The main script that handles image capture, processing, and data logging.
- `piorbit.log`: Log file that records the ISS position, image capture details, and processing results.
- `result.txt`: File that contains the estimated ISS speed.
- `piorbit_data.csv`: The data logging file for SenseHat.
- `piorbit_model.tflite`: The TFLite model used with Edge TPU for cloud classification.
- `piorbit_labels.txt`: The labels file associated with the TFLite model.

## Requirements:
- Python 3.x
- Required Python libraries:
  - pathlib
  - datetime
  - time
  - math
  - collections
  - csv
  - logging
  - logzero
  - exif
  - skyfield
  - numpy
  - cv2
  - matplotlib
  - PIL
  - pycoral
  - sense_hat

## Installation:
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/ESA_MSL2025.git
   cd ESA_MSL2025
   ```

2. Install the required Python libraries:
   pip install -r requirements.txt

Usage:
1. Run the main script:
   python main.py

2. The script will capture images, process them, and log the results in piorbit.log.

Functions:

get_cam_angle(sense: SenseHat) -> float
Returns the camera angle using the SenseHat accelerometer sensor.

get_image_distance(vertical_image_distance: float, cam_angle: float) -> float
Returns the image distance using the ISS elevation (nadir) and the angle between the camera movement direction.

get_gsd(i_with, i_height, focal, s_with, s_height, height)
Calculates the Ground Sample Distance (GSD) using the image distance, lens, and sensor specifications.

get_lap_distance(feature_distance, GSD)
Returns the linear distance in kilometers given the distance in pixels and the GSD.

calculate_orbit_distance(linear_distance, iss_height)
Calculates the real orbit distance (circle arc) given the ISS height and linear distance between coordinates.

get_speed_in_kmps(distance, time_difference)
Returns the estimated ISS speed in km/s based on a given distance and time.

Logs:
The piorbit.log file contains detailed logs of the ISS position, image capture details, and processing results. Each log entry includes timestamps, ISS coordinates, elevation, and other relevant information.

License:
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments:
- ESA for the MSL2025 mission.
- Contributors and maintainers of the required Python libraries.