import argparse
import numpy as np
import sys
import time
import tritonclient.grpc as grpcclient
import os
sys.path.append('image_captioning/base')

from triton_infer import BaseTriton

sys.path.append('system_project/image_captioning/configs')

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../configs'))

from config import Config
cfg = Config()
# URL = '192.168.1.140:8001'
cfg.infer()
URL = cfg.URL
model_name = 'encoder'

class BaseEncoder(BaseTriton):
    def __init__(self, input_name=['images'], input_type=['FP32'], input_dim=[[3,224,224]], output_name=['features'], url=URL, verbose=False, ssl=False, root_certificates=None, private_key=None, certificate_chain=None, client_timeout=None, grpc_compression_algorithm=None, static=False):
        super().__init__(input_name, input_type, input_dim, output_name, url, verbose, ssl, root_certificates, private_key, certificate_chain, client_timeout, grpc_compression_algorithm, static)
    
    def forwark(self, input_data, model_name):
        outputs = self.predict(inputs_data=input_data, model_name=model_name)
        return outputs



def encoder(images):
    encoder = BaseEncoder()
    images = np.array(images, dtype=np.float32)
    return encoder.forwark([images], model_name='encoder')



if __name__ == "__main__":
    # encoder = BaseEncoder()
    input = np.random.rand(3,224,224)
    # input = np.array(input, dtype=np.float32)
    # model_name = 'encoder'
    t0 = time.time()
    output = encoder(input)
    print(output)
    print("time", time.time() - t0)