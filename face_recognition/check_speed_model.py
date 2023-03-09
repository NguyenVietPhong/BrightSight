import face_recognition
# import cv2
# import numpy as np
# 
# from util import load_pickle
from time import time
from tqdm import tqdm
# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
start = time()
unknown_image = face_recognition.load_image_file("/home/phong/system_project/face_recognition/images_face/cong.jpg")
known_image = face_recognition.load_image_file("/home/phong/system_project/face_recognition/images_face/phong.jpg")
unknown_encoding = face_recognition.face_encodings(unknown_image, model='large')[0]

for i in tqdm(range(100)):
    # unknown_image = face_recognition.load_image_file("/home/phong/system_project/face_recognition/images_face/cong.jpg")

    biden_encoding = face_recognition.face_encodings(known_image, model='large')[0]

    results = face_recognition.compare_faces([biden_encoding], unknown_encoding)

end = time()

print("Time infer: ", (end-start)/100)