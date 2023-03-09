import argparse
import numpy as np
import sys
import time
import tritonclient.grpc as grpcclient
import os
# # Include base functions
# import cv2
# import threading
# from datetime import datetime
# import base64
# import numpy as np
# import time
# import os

# def save_img(img, name):
#     cv2.imwrite(name, img)

# def save_log_image(img, prefix, filename=None):
#     today = datetime.now()
#     log_folder = prefix + '/' + today.strftime('%Y%m%d')
#     if not os.path.isdir(log_folder):
#         os.makedirs(log_folder)
#     if filename is None:
#         filename = 'id_' + str(int(time.time()*1000.0)) + '.jpg'
#     img_path = log_folder +'/'+ filename

#     th = threading.Thread(target=save_img, args=(img, img_path))
#     th.start()
#     return filename

# def base64_to_image(base64_string):
#     base64_string = base64_string.replace('data:image/jpeg;base64,','')
#     base64_string = base64_string.replace('data:image/jpg;base64,', '')
#     base64_string = base64_string.replace('data:image/png;base64,', '')

#     im_bytes = base64.b64decode(base64_string)
#     im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
#     img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return img

# def image_to_base64(image):
#     cv2.imwrite('data/align.jpg', image)
#     with open('data/align.jpg', 'rb') as f:
#         b64 = base64.b64encode(f.read())
#     b64 = str(b64)[2:-1]
#     return b64
