import os
import time
import sys
import numpy as np
import cv2
# sys.path.append('./image_captioning')
# from utils import *
from config import Config
import torch
from PIL import Image
from torchvision import transforms
#################### import image captioning ####################

from beam_search import ImageCaptioningPredictor

config = Config()
config.infer()
predictor = ImageCaptioningPredictor(config)

####################### processing image #######################

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

###################### speech to text ##########################

sys.path.append('../speech_text')
from speech2text_mic import speech2text_mic
from text2speech import text2speech

######################### function run ##########################
__TEXT_REQUIRES_IMG_CAP__ = ["hello"]

def run(frame):

    # text = speech2text_mic()
    text = 'hello'
    if text not in __TEXT_REQUIRES_IMG_CAP__:
        run(frame)
    else:
        sentence_ = predictor.predict(frame_)
        print(sentence_)
        text2speech(sentence_)
        # run(frame)
##################### load video capture #######################
if __name__ == '__main__':
    VIDEO_PATH = 0
    cap = cv2.VideoCapture(0)

    if (cap.isOpened()==False):
        print("Error opening video stream")

    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.resize(frame, dsize = (224,224), interpolation = cv2.INTER_AREA)
        frame_ = Image.fromarray(frame)
        frame_ = transform(frame_)
        if ret==True:
            # cv2.imshow('Video', frame)
            run(frame_)
            # if cv2.waitKey(25) & 0xFF == ord('q'):
                # break
        
        else:
            break

    cap.release()

    cv2.destroyAllWindows()

