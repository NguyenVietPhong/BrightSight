import argparse
import numpy as np
import sys
import time
import tritonclient.grpc as grpcclient
import os
sys.path.append('image_captioning/base')

from triton_infer import BaseTriton
sys.path.append('system_project/image_captioning/configs')
from config import Config
cfg = Config()
# URL = '192.168.1.140:8001'
cfg.infer()
URL = cfg.URL

class BaseLstm(BaseTriton):
    def __init__(self, input_name=['input','ho','co'], input_type=['FP32','FP32','FP32'], input_dim=[[1,1,256],[2,1,512],[2,1,512]], output_name=['output','hn','cn'], url=URL, verbose=False, ssl=False, root_certificates=None, private_key=None, certificate_chain=None, client_timeout=None, grpc_compression_algorithm=None, static=False):
        super().__init__(input_name, input_type, input_dim, output_name, url, verbose, ssl, root_certificates, private_key, certificate_chain, client_timeout, grpc_compression_algorithm, static)
    
    def forwark(self, input_data, model_name='lstm'):
        output = self.predict(inputs_data=input_data,model_name=model_name)
        return output

def lstm(input, ho, co):
    lstm = BaseLstm()

    input = np.array(input, dtype=np.float32)
    ho = np.array(ho, dtype=np.float32)
    co = np.array(co, dtype=np.float32)
    return lstm.forwark([input, ho, co], model_name='lstm')
if __name__ == "__main__":
    input = np.random.rand(1,1,256)
    h0 = np.random.rand(2,1,512)
    c0 = np.random.rand(2,1,512)
    t0 = time.time()
    out = lstm(input, h0, c0)      
    print(out[1].shape)
    print("time", time.time() - t0)
