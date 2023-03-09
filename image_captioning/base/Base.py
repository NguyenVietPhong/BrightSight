import numpy as np
import sys
import time
import tritonclient.grpc as grpcclient
import os
sys.path.append('image_captioning/base')
from triton_infer import BaseTriton
URL = '192.168.1.140:8001'

class BaseEncoder(BaseTriton):
    def __init__(self, input_name=['images'], input_type=['FP32'], input_dim=[[3,224,224]], output_name=['features'], url=URL, verbose=False, ssl=False, root_certificates=None, private_key=None, certificate_chain=None, client_timeout=None, grpc_compression_algorithm=None, static=False):
        super().__init__(input_name, input_type, input_dim, output_name, url, verbose, ssl, root_certificates, private_key, certificate_chain, client_timeout, grpc_compression_algorithm, static)
    
    def forwark(self, input_data):
        outputs = self.predict(inputs_data=input_data, model_name='encoder')
        return outputs

class BaseEmbed(BaseTriton):
    def __init__(self, input_name=['captions'], input_type=['INT64'], input_dim=[[-1]], output_name=['embeddings'], url=URL, verbose=False, ssl=False, root_certificates=None, private_key=None, certificate_chain=None, client_timeout=None, grpc_compression_algorithm=None, static=False):
        super().__init__(input_name, input_type, input_dim, output_name, url, verbose, ssl, root_certificates, private_key, certificate_chain, client_timeout, grpc_compression_algorithm, static)
    
    def forwark(self, input_data):
        outputs = self.predict(inputs_data=input_data, model_name='embed')
        return outputs

class BaseLinear(BaseTriton):
    def __init__(self, input_name=['hiddens_0'], input_type=['FP32'], input_dim=[[1,1,512]], output_name=['linear_output'], url=URL, verbose=False, ssl=False, root_certificates=None, private_key=None, certificate_chain=None, client_timeout=None, grpc_compression_algorithm=None, static=False):
        super().__init__(input_name, input_type, input_dim, output_name, url, verbose, ssl, root_certificates, private_key, certificate_chain, client_timeout, grpc_compression_algorithm, static)
    
    def forwark(self, input_data):
        outputs = self.predict(inputs_data=input_data, model_name='linear')
        return outputs

class BaseLogsoftmax(BaseTriton):
    def __init__(self, input_name=['linear_out'], input_type=['FP32'], input_dim=[[1,1,11312]], output_name=['logsoftmax_output'], url=URL, verbose=False, ssl=False, root_certificates=None, private_key=None, certificate_chain=None, client_timeout=None, grpc_compression_algorithm=None, static=False):
        super().__init__(input_name, input_type, input_dim, output_name, url, verbose, ssl, root_certificates, private_key, certificate_chain, client_timeout, grpc_compression_algorithm, static)
    
    def forwark(self, input_data):
        outputs = self.predict(inputs_data=input_data, model_name='logsoftmax')
        return outputs

class BaseLstm(BaseTriton):
    def __init__(self, input_name=['input','ho','co'], input_type=['FP32','FP32','FP32'], input_dim=[[1,1,256],[2,1,512],[2,1,512]], output_name=['output','hn','cn'], url=URL, verbose=False, ssl=False, root_certificates=None, private_key=None, certificate_chain=None, client_timeout=None, grpc_compression_algorithm=None, static=False):
        super().__init__(input_name, input_type, input_dim, output_name, url, verbose, ssl, root_certificates, private_key, certificate_chain, client_timeout, grpc_compression_algorithm, static)
    
    def forwark(self, input_data):
        output = self.predict(inputs_data=input_data,model_name='lstm')
        return output

class BaseLstmInit(BaseTriton):
    def __init__(self, input_name=['h0'], input_type=['FP32'], input_dim=[[1,1,256]], output_name=['output','hn','cn'], url=URL, verbose=False, ssl=False, root_certificates=None, private_key=None, certificate_chain=None, client_timeout=None, grpc_compression_algorithm=None, static=False):
        super().__init__(input_name, input_type, input_dim, output_name, url, verbose, ssl, root_certificates, private_key, certificate_chain, client_timeout, grpc_compression_algorithm, static)
    
    def forwark(self, input_data):
        output = self.predict(inputs_data=input_data,model_name='lstm_init')
        return output
