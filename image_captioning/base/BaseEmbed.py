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
model_name = 'embed'

class BaseEmbed(BaseTriton):
    def __init__(self, input_name=['captions'], input_type=['INT64'], input_dim=[[-1]], output_name=['embeddings'], url=URL, verbose=False, ssl=False, root_certificates=None, private_key=None, certificate_chain=None, client_timeout=None, grpc_compression_algorithm=None, static=False):
        super().__init__(input_name, input_type, input_dim, output_name, url, verbose, ssl, root_certificates, private_key, certificate_chain, client_timeout, grpc_compression_algorithm, static)
    
    def forwark(self, input_data, model_name):
        outputs = self.predict(inputs_data=input_data, model_name=model_name)
        return outputs



def embed(captions):
    embed = BaseEmbed()
    captions = np.array(captions, dtype=np.int64)
    return embed.forwark([captions], model_name='embed')



if __name__ == "__main__":
    # encoder = BaseEncoder()
    input = np.random.rand(30)
    # input = np.array(input, dtype=np.float32)
    # model_name = 'encoder'
    t0 = time.time()
    output = embed(input)
    print(output)
    print("time", time.time() - t0)