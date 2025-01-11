# Import the Path class for get the base folder of the app
from pathlib import Path
# Import logger for applicarion log file.
import logging
import logzero
from logzero import logger, logfile
# Import the PiCamera class from the picamera module
# from picamera import PiCamera
# Import the Image class from exif module for get image information
from exif import Image
# Import skyfield library to get the ISS coordinates and ephemeris data.
from skyfield.api import Topos, load
from skyfield.api import EarthSatellite
# Import datetime class for manage data/time data
from datetime import datetime, timedelta
# Import sleep for set the capture delay between images 
from time import sleep
# Import a dictionary for store the data
from collections import defaultdict
# Import libraries for OpenCV image processing
import numpy
import cv2
from math import sqrt, hypot, cos
from matplotlib import pyplot
# Import  libraries for Tensorflow lite (Coral)
from PIL import Image as ImageCoral
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file
# Import SenseHat and csv for sense data logging file
from sense_hat import SenseHat
from csv import writer

BASE_FOLDER = ""                # Const for base folder path
EPHEMERIS_FILE = "de421.bsp"    # Const for ephemeris file name
TLE_FILE = "stations.tle"       # Const for stations TLE file name
ISS_TLE1 = "1 25544U 98067A   25007.20984286  .00021466  00000+0  37141-3 0  9990" # if can't load the TLE file
ISS_TLE2 = "2 25544  51.6399  24.4706 0006536  53.8303  40.3485 15.50861227490194" # if can't load the TLE file
EXECUTION_TIME_DELTA = 9        # Const for control max program execution time in minutes
EXECUTION_IMAGES_TIME_DELTA = 5 # Const for control max images capture time in minutes
IMAGE_CAPTURE_DELAY = 2         # Const for control delay time in seconds between images capture
MAX_IMAGES = 42                 # Const for control max amount of images
RES_WITH = 4056                 # Image with resolutiom
RES_HEIGHT = 3040               # Image height resolution
FOCAL = 5                       # Kawa focal lens in mm
SENSOR_WITH = 6.287             # Raspberry Pi HQ sensor with
SENSOR_HEIGHT = 4.712           # Raspberry Pi HQ sensor height
GENERIC_GSD = 12648             # Generic GSD values if is not possible to calculate the finest value
FEATURE_NUMBER = 1000           # Const for set the images feature number
MIN_MATCH_COUNT = 10            # Const for min valid image features
MIN_ISS_LRO = 370               # Minimum ISS LRO distance in Km
MAX_ISS_LRO = 460               # Maximum ISS LRO distance in Km
HIGH_CLOUDS_HEIGHT = 16         # Hight clouds heigh used for elevation correction in Km
MID_CLOUDS_HEIGHT = 8           # Mid clouds height used for elevation correction in Km
LOW_CLOUDS_HEIGHT = 2           # Low clouds height used for elevation correction in Km

# Fuction that returns the ISS object from the TLE data
def get_ISS():
    try:
        ts = load.timescale()
        try:
            satellites = load.tle_file(TLE_FILE)
            by_name = {sat.name: sat for sat in satellites}
            iss = by_name['ISS (ZARYA)']
            logger.info(f"ISS epoch from TLE file: {iss}.")
        except Exception as eTLE:
            iss = EarthSatellite(ISS_TLE1, ISS_TLE2, 'ISS (ZARYA)', ts)
            logger.info(f"ISS epoch from const: {iss}.")
        
        return iss
    except Exception as e:
        logger.error("Error getting the ISS object")
        raise e

# Function that returns a log data record
# with IIS coordinates and sensehat sensors values
def get_log_data(iss,sense):
    try:
        log_data = []
        # Get the date and time
        log_data.append(datetime.now())
        # Get the ISS coordinates
        t = load.timescale().now()
        position = iss.at(t)
        location = position.subpoint()
        log_data.append(f"{location.latitude.degrees:.5f}")
        log_data.append(f"{location.longitude.degrees:.5f}")
        log_data.append(f"{location.elevation.km:.3f}")
        latitude = location.latitude.dstr(format="{0}{1}º {2:02}' {3:02}.{4:0{5}}\"")
        longitude = location.longitude.dstr(format="{0}{1}º {2:02}' {3:02}.{4:0{5}}\"")
        log_data.append(f"{latitude}")
        log_data.append(f"{longitude}")
        # Get the sunlit                    
        ephemeris = load(EPHEMERIS_FILE) # Load the DE421 ephemeris file
        if iss.at(t).is_sunlit(ephemeris):
            sunlit = "sunlight"
        else:
            sunlit = "darkness"
        log_data.append(f"{sunlit}")
        # Get environmental data
        log_data.append("{:.5f}".format(sense.get_temperature()))
        log_data.append("{:.5f}".format(sense.get_pressure()))
        log_data.append("{:.5f}".format(sense.get_humidity()))
        # Get orientation data
        orientation = sense.get_orientation()
        log_data.append("{:.5f}".format(orientation["yaw"]))
        log_data.append("{:.5f}".format(orientation["pitch"]))
        log_data.append("{:.5f}".format(orientation["roll"]))
        # Get accelerometer data
        acc = sense.get_accelerometer_raw()
        log_data.append("{:.5f}".format(acc["x"]))
        log_data.append("{:.5f}".format(acc["y"]))
        log_data.append("{:.5f}".format(acc["z"]))
        #Get gyroscope data
        gyro = sense.get_gyroscope_raw()
        log_data.append("{:.5f}".format(gyro["x"]))
        log_data.append("{:.5f}".format(gyro["y"]))
        log_data.append("{:.5f}".format(gyro["z"]))
        # Get compass data
        mag = sense.get_compass_raw()
        log_data.append("{:.5f}".format(mag["x"]))
        log_data.append("{:.5f}".format(mag["y"]))
        log_data.append("{:.5f}".format(mag["z"]))
        
        return log_data
    except Exception as e:
        logger.error("Error in get_log_data function")
        raise e

# Function that converts a `skyfield` Angle to an Exif-appropriate
# representation (positive rationals)
# e.g. 98° 34' 58.7 to "98/1,34/1,587/10"
# Return a tuple containing a Boolean and the converted angle,
# with the Boolean indicating if the angle is negative
def convert(angle):
    try:
        sign, degrees, minutes, seconds = angle.signed_dms()        
        exif_angle = f'{degrees:.0f}/1,{minutes:.0f}/1,{seconds*10:.0f}/10'
        return sign < 0, exif_angle
    except Exception as e:
        logger.error("Error in convert function")
        raise e

# Function that captures a pic setting the ISS location in the EXIF data.
def custom_capture(iss, cam, image):
    try:
        # Get the position of the ISS
        ts = load.timescale()
        t = ts.now()
        position = iss.at(t)
        location = position.subpoint()

        # Convert the latitude and longitude to Exif-appropriate representations
        latRef, exif_latitude = convert(location.latitude)
        longRef, exif_longitude = convert(location.longitude)
        latRef = "S" if latRef else "N"
        longRef = "W" if longRef else "E"

        # Set the Exif tags specifying the current location
        cam.exif_tags['GPS.GPSLatitude'] = exif_latitude
        cam.exif_tags['GPS.GPSLatitudeRef'] = latRef
        cam.exif_tags['GPS.GPSLongitude'] = exif_longitude
        cam.exif_tags['GPS.GPSLongitudeRef'] = longRef

        # Capture the image
        cam.capture(image)
    except Exception as e:
        logger.error("Error in custom_capture function")
        raise e

# Function that returns the original taken datetime
# in format %Y:%m:%d %H:%M:%S
def get_time(image):
    try:
        with open(image, 'rb') as image_file:
            img = Image(image_file)
            time_str = img.get("datetime_original")
            time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
            return time
    except Exception as e:
        logger.error("Error in get_time function")
        raise e

# Function that returns the time diference in seconds
# between two images
def get_time_difference(prev_image, new_image):
    try:
        time_prev_image = get_time(prev_image)
        time_new_image = get_time(new_image)
        time_difference = time_new_image - time_prev_image
        return time_difference.seconds
    except Exception as e:
        logger.error("Error in get_time_difference function")
        raise e

# Function that converts two images to a CV images
def convert_to_cv(prev_image, new_image):
    try:
        prev_image_cv = cv2.imread(prev_image, cv2.IMREAD_GRAYSCALE)
        new_image_cv = cv2.imread(new_image, cv2.IMREAD_GRAYSCALE)
        return prev_image_cv, new_image_cv
    except Exception as e:
        logger.error("Error in convert_to_cv function")
        raise e

# Function that returns the keypoints and descriptors of
# two given images based on a given feature number using SIFT algorithm.
def calculate_features(prev_image_cv, new_image_cv, feature_number):
    try:
        sift = cv2.SIFT_create(nfeatures = feature_number)
        keypoints_prev_image, descriptors_prev_image = sift.detectAndCompute(prev_image_cv, None)
        keypoints_new_image, descriptors_new_image = sift.detectAndCompute(new_image_cv, None)
        
        return keypoints_prev_image, keypoints_new_image, descriptors_prev_image, descriptors_new_image
    except Exception as e:
        logger.error("Error in calculate_features function")
        raise e
   
# Function that returns the matches between the given
# descriptors of two images using SIFT algorithm.
def calculate_matches(descriptors_prev_image, keypoints_prev_image, descriptors_new_image, keypoints_new_image):
    try:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(descriptors_prev_image,descriptors_new_image,k=2)
        good_matches = []
        for m,n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        if len(good_matches)>MIN_MATCH_COUNT:
            src_pts = numpy.float32([keypoints_prev_image[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
            dst_pts = numpy.float32([keypoints_new_image[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
        else:
            matchesMask = None
        
        return good_matches, matchesMask
    except Exception as e:
        logger.error("Error in calculate_matches function")
        raise e
   
# JUST FOR TEST PURPOSES. NOT USE IN THE ISS EXECUTION
# Function that shows the matches between two images. 
def display_matches(prev_image_cv, keypoints_prev_image, new_image_cv, keypoints_new_image, matches, matchesMask):
    try:        
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = None,
                           matchesMask = matchesMask,
                           flags = 2)
        
        match_img = cv2.drawMatches(prev_image_cv, keypoints_prev_image, new_image_cv, keypoints_new_image, matches, None, **draw_params)
        resize = cv2.resize(match_img, (1600,600), interpolation = cv2.INTER_AREA)
        cv2.imshow('matches', resize)
        cv2.waitKey(0)
        cv2.destroyWindow('matches')
    except Exception as e:
        logger.error("Error in display_matches function")
        raise e

# Function that returns an array for each image with the
# matching coordinates between two images.
def find_matching_coordinates(keypoints_prev_image, keypoints_new_image, matches):
    try:
        coordinates_prev_image = []
        coordinates_new_image = []
        for match in matches:
            prev_image_idx = match.queryIdx
            new_image_idx = match.trainIdx
            (x1,y1) = keypoints_prev_image[prev_image_idx].pt
            (x2,y2) = keypoints_new_image[new_image_idx].pt
            coordinates_prev_image.append((x1,y1))
            coordinates_new_image.append((x2,y2))
        return coordinates_prev_image, coordinates_new_image
    except Exception as e:
        logger.error("Error in find_matching_coordinates function")
        raise e

# Function that returns the distance between two images using
# all matching coordinates. 
def calculate_mean_distance(coordinates_prev_image, coordinates_new_image):
    try:
        all_distances = 0
        merged_coordinates = list(zip(coordinates_prev_image, coordinates_new_image))
        if len(merged_coordinates) > 0:
            for coordinate in merged_coordinates:
                x_difference = coordinate[0][0] - coordinate[1][0]
                y_difference = coordinate[0][1] - coordinate[1][1]
                distance = hypot(x_difference, y_difference)
                all_distances = all_distances + distance
            return all_distances / len(merged_coordinates)
        else:
            raise ValueError("No matching coordinates found")
    except Exception as e:
        logger.error(f"Error in calculate_mean_distance function: {e}")
        raise e

# Function that uses the Tensorflow Lite with Coral and a definition model
# for classify type of clouds (low, mid, hight) that appears in images.
# With each % of cloud type is calculed a distance correction for the GSD
# acuracy
def get_clouds_correction_distance(model_path, label_path, image_file):
    try:
        interpreter = make_interpreter(f"{model_path}")
        interpreter.allocate_tensors()
        
        size = common.input_size(interpreter)
        image = ImageCoral.open(image_file).convert('RGB').resize(size, ImageCoral.NEAREST)
        
        common.set_input(interpreter, image)
        interpreter.invoke()
        classes = classify.get_classes(interpreter, top_k=4)
        
        distance = 0
        labels = read_label_file(label_path)
        for c in classes:
            if labels.get(c.id, c.id) == "LOW_CLOUDS":
                distance = distance + LOW_CLOUDS_HEIGHT * c.score
            elif labels.get(c.id, c.id) == "MID_CLOUDS":
                distance = distance + MID_CLOUDS_HEIGHT * c.score
            elif labels.get(c.id, c.id) == "HIGH_CLOUDS":
                distance = distance + HIGH_CLOUDS_HEIGHT * c.score
            logger.info(f"Clouds detection {labels.get(c.id, c.id)}: {c.score*100:.2f}%")            
        logger.info(f"Clouds distance correction: {distance} Km.") 
        return distance
    except Exception as e:
        logger.error("Error in get_clouds_correction_distance function")
        raise e

# Function that return the angle between the IIS direction or camera direction
# and the nadir axis.
def get_nadir_cam_angle(sense):
    try:
        acc = sense.get_accelerometer_raw()
        ori = sense.get_orientation()
        x = acc["x"]
        y = acc["y"]
        z = acc["z"]
        if x > y and x > z:
            g_axis = ori["yaw"]
        elif y > x and y > z:
            g_axis = ori["pitch"]
        else:
            g_axis = ori["roll"]
        
        if g_axis > 180:
            nadir_cam_angle = 360 - g_axis
        else:
            nadir_cam_angle = 180 - g_axis
            
        return nadir_cam_angle
    except Exception as e:
        logger.error("Error in get_nadir_cam_angle function")
        raise e

# Function that returns the image distance, using the ISS elevation (nadir)
# and the angle between the movement direction.
def get_image_distance(vertical_image_distance, nadir_cam_angle):
    try:
        img_distance = vertical_image_distance / cos(nadir_cam_angle)
        return img_distance
    except Exception as e:
        logger.error("Error in get_image_distance function")
        raise e
    
# Function that calculates the GSD using the image distance
# (IIS elevation + cam angle correction) with the lens and
# sensor especifications.
def get_gsd(i_with, i_height, focal, s_with, s_height, height):
    try:
        # calculate GSDh and GSHw
        gsd_h = (height * s_height) / (focal * i_height) * 100
        gsd_w = (height * s_with) / (focal * i_with) * 100
        #return the lowest GSD value to ensure using the worst-case scenario
        if gsd_h < gsd_w:
            return int(gsd_h)
        else:
            return int(gsd_w)
    except Exception as e:
        logger.error("Error in get_gsd function")
        raise e

# Function that returns the lineal distance in Km
# with given distance in px and the GSD.
def get_lap_distance(feature_distance, GSD):
    try:
        distance = feature_distance * GSD / 100000
        return distance
    except Exception as e:
        logger.error("Error in get_estimate_lap_distance function")
        raise e

# Function that returns the estimated ISS speed in km/s based
# on a given distance and time
def get_speed_in_kmps(distance, time_difference):
    try:
        estimate_kmps = distance / time_difference
        return estimate_kmps
    except Exception as e:
        logger.error("Error in calculate_speed_in_kmps function")
        raise e

# Main program
try:
    # Create a variable to store the start time and compare with running time
    start_time = datetime.now()

    # Set the base folder for the app and the files path
    BASE_FOLDER = Path(__file__).parent.resolve()
    log_file_path = BASE_FOLDER / "piorbit.log" # Application log file
    result_file_path = BASE_FOLDER / "result.txt" # File that contains the estimated ISS speed
    log_data_file_path = BASE_FOLDER / "piorbit_data.csv" # The data logging file for sensehat
    model_path = BASE_FOLDER / "piorbit_model.tflite" # The TFLite converted to be used with edgetpu
    label_path = BASE_FOLDER / "piorbit_labels.txt" # The path to labels.txt associated with the model

    # Set the logfile
    logzero.loglevel(logzero.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s');
    logzero.formatter(formatter)
    logfile(log_file_path, disableStderrLogger=True)
except Exception as e:
    print(f"Fatal error creating requiered files acces: {e}")

try:
    logger.info(f"Start time: {start_time}")
    try:
        # Create an instance of the PiCamera class and set the resolution
        #cam = PiCamera()
        #cam.resolution = (RES_WITH, RES_HEIGHT)
        cam=0
    except Exception as eCam:
        logger.error("Camera not initialized")
        raise eCam

    # Start speed calculation and SenseHat data logging process
    try:
        # open the speed result file
        speed_file = open(result_file_path, 'w', buffering=1)
        speed_file.write("N/A")
        
        # Open the data logger file and set the header for collect data from ISS coordinates and sensehat sensor.
        sense = SenseHat()
        sense.clear()
        log_data_file = open(log_data_file_path, 'w', buffering=1, newline='')
        log_data_writer = writer(log_data_file)
        log_data_writer.writerow(['Datetime','Lat','Long','Elevation','Latitude','Longitude','Sunlit','Temp','Pres','Hum','Yaw','Pitch','Roll','Acc_x','Acc_y','Acc_z','Gyro_x','Gyro_y','Gyro_z','Mag_x','Mag_y','Mag_z'])
        
        # Get the ISS object
        iss = get_ISS()

        # Set the variables used for speed calculation and init loop control flow
        total_estimate_kmps = 0
        i_image = 1
        data_dict = defaultdict(dict) # Initialize the dictionary to store data
        now_time = datetime.now()
        
        # Loop for get images and data with exit control requirements (execution time and maximun number of images)
        while (now_time < start_time + timedelta(minutes = EXECUTION_IMAGES_TIME_DELTA) and i_image <= MAX_IMAGES):
            # Capture an image
            new_image = str(BASE_FOLDER) + "/gps_image" + str(i_image) + ".jpg"
            logger.info(f"Requesting pic: {new_image}")
            capture_init_time = datetime.now()
            #custom_capture(iss,cam,new_image)
            capture_end_time = datetime.now()
            logger.info(f"Image capture spend time (seconds): {(capture_end_time - capture_init_time).total_seconds():.2f} sec.")
                
            # Save a record in data log file
            log_data = get_log_data(iss,sense)
            logger.info(f"ISS position: Latitude={log_data[1]} Longitude={log_data[2]} Elevation={log_data[3]}, Sunlit={log_data[6]}")
            log_data_writer.writerow(log_data)
            logger.info("Data record saved to csv file.")
            coord = Topos(latitude_degrees=float(log_data[1]), longitude_degrees=float(log_data[2]), elevation_m=float(log_data[3]))    

            # Store data in the dictionary
            data_dict[i_image]['image'] = new_image
            data_dict[i_image]['time'] = capture_end_time
            data_dict[i_image]['latitude_degrees'] = float(log_data[1])
            data_dict[i_image]['longitude_degrees'] = float(log_data[2])
            data_dict[i_image]['elevation_m'] = float(log_data[3])
            data_dict[i_image]['sunlit'] = log_data[6]
            data_dict[i_image]['coord'] = coord
            data_dict[i_image]['a_nadir'] = get_nadir_cam_angle(sense) #Get the camera angle using the sense hat accelerometer sensor.
            logger.info("Data record saved in dictionary.")

            # Values for next iteration and control time for loop exit
            sleep(IMAGE_CAPTURE_DELAY)
            i_image = i_image + 1
            now_time = datetime.now()
        # Out of the get images and data loop
        log_data_file.close()
        #cam.close

        total_estimate_kmps = 0
        i_Data = 1
        now_time = datetime.now()
        while(now_time < start_time + timedelta(minutes = EXECUTION_TIME_DELTA) and i_Data < len(data_dict)):
            try:
                logger.info(f"START dataset iteration: {i_Data}")    

                # Calculate the distance between coordinates
                coord1 = data_dict[i_Data]['coord']
                coord2 = data_dict[i_Data + 1]['coord']
                time1 = data_dict[i_Data]['time']
                time2 = data_dict[i_Data + 1]['time']
                ts = load.timescale()
                t = ts.now()
                position1 = coord1.at(t).position.km
                position2 = coord2.at(t).position.km
                coord_distance = sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2 + (position1[2] - position2[2])**2)
                coord_time_difference = (time2 - time1).seconds
                coord_speed = coord_distance / coord_time_difference

                # Calculate the distance between images
                image1 = data_dict[i_Data]['image']
                image2 = data_dict[i_Data + 1]['image']
                nadir_cam_angle1 = data_dict[i_Data]['a_nadir']
                nadir_cam_angle2 = data_dict[i_Data + 1]['a_nadir']
                img_time_difference = get_time_difference(image1, image2) # Get time difference between images.
                image1_cv, image2_cv = convert_to_cv(image1, image2) # Create OpenCV image objects.
                    
                # SIFT algorithm for feature matching
                keypoints_image1, keypoints_image2, descriptors_image1, descriptors_image2 = calculate_features(image1_cv, image2_cv, FEATURE_NUMBER) # Get keypoints and descriptors.
                matches, matchesMask = calculate_matches(descriptors_image1, keypoints_image1, descriptors_image2, keypoints_image2) # Match descriptors.
                if (matches == 0):
                    logger.warning("No matches found between images. Trying next image.")
                else:
                    #display_matches(image1_cv, keypoints_image1, image2_cv, keypoints_image2, matches, matchesMask) # Display matches.                                
                    coordinates_image1, coordinates_image2 = find_matching_coordinates(keypoints_image1, keypoints_image2, matches) # Match the keypoints between images.
                    lap_average_distance = calculate_mean_distance(coordinates_image1, coordinates_image2) # Calculate the average feature distance between images
                    
                    # Image distance and GSD calculation
                    lap_elevation_dif = 0
                    nadir_cam_angle = 0
                    img_distance = 0
                    gsd = GENERIC_GSD
                    img_elevation1 = data_dict[i_Data]['elevation_m']
                    img_elevation2 = data_dict[i_Data + 1]['elevation_m']
                    
                    if((img_elevation1 >= MIN_ISS_LRO and img_elevation1 <= MAX_ISS_LRO) and
                    (img_elevation2 >= MIN_ISS_LRO and img_elevation2 <= MAX_ISS_LRO)):
                        try:
                            img_elevation_dif = img_elevation2 - img_elevation1 #Information log about the elevation difference between two images in meters
                            vertical_distance = ((img_elevation1 + img_elevation2)/2) #Average elevation between two images used for GSD calculation
                            logger.info(f"Processing clouds image: {image1}")
                            vertical_distance1 = vertical_distance - get_clouds_correction_distance(model_path,label_path,image1) #Vertical distance from orbit minus the average height of clouds
                            logger.info(f"Processing clouds image: {image2}")
                            vertical_distance2 = vertical_distance - get_clouds_correction_distance(model_path,label_path,image2) #Vertical distance from orbit minus the average height of clouds
                            vertical_distance = (vertical_distance1 + vertical_distance2) / 2 #Average vertical distance between two images used for GSD calculation                            
                            cam_distance1 = get_image_distance(vertical_distance1,nadir_cam_angle1) #Get the image distance with known camera angle and vertical distance.
                            cam_distance2 = get_image_distance(vertical_distance2,nadir_cam_angle2) #Get the image distance with known camera angle and vertical distance.
                            cam_distance = (cam_distance1 + cam_distance2) / 2 #Average camera distance between two images used for GSD calculation
                            gsd = get_gsd(RES_WITH,RES_HEIGHT,FOCAL,SENSOR_WITH,SENSOR_HEIGHT,cam_distance * 1000) #Get the GSD with known values and distance in meters.
                        except Exception as eGSD:                        
                            logger.error(f"Error calculating the GSD: {eGSD}")
                    else:                    
                        logger.warning("Unable to calculate the GSD")
                    
                    img_distance = get_lap_distance(lap_average_distance, gsd) # Calculate the distance lap in Km using the GSD
                    estimate_kmps = get_speed_in_kmps(img_distance, img_time_difference) # Calculate the speed lap in Km/s between images
                    
                    # Write lap values to log file
                    logger.info(f"RESULTS dataset iteration: {i_Data}")      
                    logger.info(f"Lap time diference between images: {img_time_difference:.2f} sec.")
                    logger.info(f"Lap estimated elevation image{i_Data}: {img_elevation1:.3f} m.")
                    logger.info(f"Lap estimated elevation image{i_Data + 1}: {img_elevation2:.3f} m.")
                    logger.info(f"Lap estimated elevation variation: {img_elevation_dif:.2f} m.")
                    logger.info(f"Lap estimated elevation with cloud correction: {vertical_distance:.3f} m.")
                    logger.info(f"Lap estimated nadir cam angle: {nadir_cam_angle:.2f} deg.")
                    logger.info(f"Lap estimated cam distance: {img_distance:.5f} Km.")
                    logger.info(f"Using GSD value: {gsd} px/cm.")
                    logger.info(f"Lap estimated features distance (SIFT): {lap_average_distance:.5f} px.")
                    logger.info(f"Lap estimated lineal image distance: {img_distance:.5f} Km.")  
                    logger.info(f"Lap estimated lineal speed using images: {estimate_kmps:.5f} kmps.")
                    logger.info(f"Lap estimated lineal coordinates distance: {coord_distance:.5f} Km.")
                    logger.info(f"Lap estimated lineal speed using coordinates: {coord_speed:.5f} kmps.")

                    # Save the average estimated speed to the results file
                    if i_Data == 1:
                        total_estimate_kmps = estimate_kmps
                    else:
                        total_estimate_kmps = (total_estimate_kmps + estimate_kmps) / 2
                    speed_file.seek(0)
                    speed_file.write(f"{total_estimate_kmps:.5f}")                    
                    logger.info(f"Estimated speed using images: {total_estimate_kmps:.5f} kmps saved to results file.")
            except Exception as eLap:
                logger.error("Error processing lap data. Trying next image.")

            # Control time for loop exit and delay between images
            now_time = datetime.now()
            i_Data = i_Data + 1

        # Out of the loop — stopping
        speed_file.close()

    except Exception as eLoop:
        logger.error("Error processing data.")
        
    end_time = datetime.now()
    logger.info(f"End time: {end_time}")
    logger.info(f"Total execution time: {end_time - start_time}")
except Exception as e:
    logger.exception(f"Program ended abnormaly: {e}")