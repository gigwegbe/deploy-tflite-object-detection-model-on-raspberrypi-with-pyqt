import numpy as np
import tensorflow as tf
#import cv2
import PIL 
import sys, time, threading, cv2
import queue as Queue

import picamera
import argparse

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from visualization_utils import draw_bounding_boxes_on_image_array
from utils import load_image
from detector import ObjectDetectorLite



IMG_SIZE    = 1280,720          # 640,480 or 1280,720 or 1920,1080
IMG_FORMAT  = QImage.Format_RGB888
DISP_SCALE  = 2                # Scaling factor for display image
DISP_MSEC   = 50                # Delay between display cycles
CAP_API     = cv2.CAP_ANY       # API: CAP_ANY or CAP_DSHOW etc...
EXPOSURE    = 0                 # Zero for automatic exposure
TEXT_FONT   = QFont("Courier", 10)

camera_num  = 1                 # Default camera (first in list)
image_queue = Queue.Queue()     # Queue to hold images
capturing   = True              # Flag to indicate capturing

# Tensorflow 
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="../../model_quantized.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']



# Grab images from the camera (separate thread)
def grab_images(cam_num, queue):
    #camera = cv2.VideoCapture(0)
    start = time.time()
    cap = cv2.VideoCapture(cam_num-1 + CAP_API)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_SIZE[1])
    if EXPOSURE:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    while capturing:
        if cap.grab():
            retval, image = cap.retrieve(0)
            img = PIL.Image.fromarray(image)
            jpeg = img.resize((640, 480))
            input_data = np.array(jpeg)
            input_data = np.expand_dims(input_data, axis=0)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            output_data = interpreter.get_tensor(output_details[0]['index'])
            if output_data[0][1] > output_data[0][0]:
                print("Person detected!")
            else:
                print("No Person detected!")
                
            if image is not None and queue.qsize() < 2:
                queue.put(image)
            else:
                time.sleep(DISP_MSEC / 1000.0)
        else:
            print("Error: can't grab camera image")
            break
       
    cap.release()



def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection')
    parser.add_argument('--model_path', type=str, help='Specify the model path', required=True)
    parser.add_argument('--label_path', type=str, help='Specify the label map', required=True)
    parser.add_argument('--confidence', type=float, help='Minimum required confidence level of bounding boxes',
                        default=0.6)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    detector = ObjectDetectorLite(model_path=args.model_path, label_path=args.label_path)
    input_size = detector.get_input_size()

    plt.ion()
    plt.tight_layout()
    
    fig = plt.gcf()
    fig.canvas.set_window_title('Object Detection')
    fig.suptitle('Detecting')
    ax = plt.gca()
    ax.set_axis_off()
    tmp = np.zeros(input_size + [3], np.uint8)
    preview = ax.imshow(tmp)
    

    with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)
        while True:
            stream = np.empty((480, 640, 3), dtype=np.uint8)
            camera.capture(stream, 'rgb')
          
            image = load_image(stream)
            boxes, scores, classes = detector.detect(image, args.confidence)
            for label, score in zip(classes, scores):
                print(label, score)
  
            if len(boxes) > 0:
                draw_bounding_boxes_on_image_array(image, boxes, display_str_list=classes)
            
            preview.set_data(image)
            fig.canvas.get_tk_widget().update()
            
    detector.close()