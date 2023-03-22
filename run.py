import os
import time
import sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from time import time

import face_recognition
import cv2
import numpy as np

from utils.utils import *
from human_detection.processing import *

import speech_recognition as sr
import time
import pyaudio

from gtts import gTTS
import os
from playsound import playsound

from configs.config import Config

from logger import Path, Logger

__path__ = Path('logger.txt')
print(__path__.cat_path())
__logger__ = Logger(path=__path__.cat_path())





FACE_RECOGNITION = ['who is the front', 'who is here', 'who is that', 'who' , 'so this is who', 'who are they', 'identify that',\
                     'recognize that guy','who is that person','what is the person up front name']
IMAGE_CAPTION = ['what is ahead', 'what is in front', 'what is up ahead', 'the situation ahead', 'front caption',\
                'what lies ahead' , 'what is being displayed','what is it actually showing']
HUMAN_DETECTION = ['how many people are there', 'how many men are in front',\
                      'how many women are in front', 'how many men and women are there in front', 'who are those people',\
                     'how many guys are there', 'the number of people ahead','the amount of persons ahead','the numbers of people present ahead']

def speech2text_mic():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        # print('Say Something:')
        audio = r.listen(source)
        try:
            # using google speech recognition
            #Adding hindi langauge option
            text = r.recognize_google(audio, language = 'en-En')
            return text

            
        except:
            print("Sorry.. run again...")
            speech2text_mic()

def text2speech(text):
    tts = gTTS(text=text, tld='com.vn', lang='en')
    tts.save("audio/audio.mp3")
    playsound("audio/audio.mp3")

def check_mode(text, FACE_RECOGNITION=FACE_RECOGNITION, IMAGE_CAPTION=IMAGE_CAPTION, HUMAN_DETECTION=HUMAN_DETECTION):
    # text = text.lower()
    if text in IMAGE_CAPTION:
        return 0
    elif text in FACE_RECOGNITION:
        return 1
    elif text in HUMAN_DETECTION:
        return 2
    elif text=='exit':
        return 3
    else:
        return 4

# sys.exit()

def read_video(caption_predictor, human_detector,  known_face_names, known_face_encodings, display=False):

    vid = cv2.VideoCapture(0)
    while True:
        line = None
        ret, frame = vid.read()
 
        text = speech2text_mic()
        mode = check_mode(text)
        # mode=1

        print(mode)
        text = ''

        if display:
            cv2.imshow('Display', frame)


        if mode==0:
            # start = time()
            image = Image.fromarray(frame, mode="RGB")
            image = processing_img_caption(image)
            text_out = caption_predictor.predict(image)
            # end = time()
            line = f'{text}\t{text_out}\n'
            __logger__.write_logger(line)
            text2speech(text_out)


        elif mode==1:

            face_names = processing_image_face_recogntion(frame, known_face_names, known_face_encodings)
            text_out = processing_text_face_recognition(face_names)

            line = f'{text}\t{text_out}\n'
            __logger__.write_logger(line)            
            text2speech(text_out)
            
        elif mode==2:
            # start = time()
            image = preprocess(frame, [640,640])
            output = human_detector.predict([image], model_name='yolov7')
            det_boxes, det_classes, det_scores, num_dets = output[0], output[1], output[2], output[3]
            detected_objects = postprocess(num_dets, det_boxes, det_scores, det_classes, frame.shape[1], frame.shape[0], [640, 640])
            text_out = processing_text_detection(detected_objects)
            # end = time()
            line = f'{text}\t{text_out}\n'
            __logger__.write_logger(line)            
            text2speech(text_out)
        elif mode==4:
            # After the loop release the cap object
            vid.release()

            # Destroy all the windows
            cv2.destroyAllWindows()
            read_video(caption_predictor, human_detector, known_face_names, known_face_encodings, display=False)
        elif mode==3:
            sys.exit()

    # After the loop release the cap object
    vid.release()

    # Destroy all the windows
    cv2.destroyAllWindows()


def main():
    config = Config()
    config.infer()
    caption_predictor = ImageCaptioningPredictor(config)
    human_detector = detector
    objs = load_pickle('/home/phong/system_project/face_recognition/face_recognition_face.pkl')


    # Create arrays of known face encodings and their names
    known_face_encodings = []
    known_face_names = []
    for i, obj in enumerate(objs):
        known_face_encodings.append(obj['feature_face'])
        known_face_names.append(obj['name'])



    read_video(caption_predictor, human_detector, known_face_names, known_face_encodings, display=False)

main()
