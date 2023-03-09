import os
from pathlib import Path

root_path = Path(os.path.dirname(os.path.abspath(__file__))).parent

def join_root_path(path, mkdir=True):
    join_path = os.path.join(root_path, path)
    if mkdir and not os.path.exists(join_path):
        os.makedirs(join_path)
    return join_path

def join_path(parent_path, child_path, mkdir=True):
    join_path = os.path.join(parent_path, child_path)
    if mkdir and not os.path.exists(join_path):
        os.makedirs(join_path)
    return join_path


MODEL_ONNX_PATH = ./checkpoint/onnx
MODEL_PYTORCH_PATH = ./checkpoint/pytorch 
MODEL_TENSORRT_PATH = ./checkpoint/TensorRT 

ENVIRONMENT_CONFIG_FILE = join_root_path('.env')


