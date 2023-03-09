import os
import pickle
from select import select
import torch

class Config(object):
    # Set the relative path from the directory of this file
    # Settings of general (hyper)parameters
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.WORD_TO_ID_PATH = '/home/phong/system_project/image_captioning/vocab/word_to_id.pkl'
        self.ID_TO_WORD_PATH = '/home/phong/system_project/image_captioning/vocab/id_to_word.pkl'
        self.URL = '192.168.1.140:8001'
        # Change relative path to absolute path
        self.TEXT_REQUIRES_IMG_CAPTION = ['what was before', 'what is in front']

    # Settings of (hyper)parameters for inference
    def infer(self):
        self.BEAM_SIZE = 1
        self.MAX_SEG_LENGTH=20

        self.INFER_IMAGE_PATH = '/home/phong/system_project/image_captioning/images/infer'
        self.INFER_RESULT_PATH = '/home/phong/system_project/image_captioning/logs/infer_results.txt'

        with open(self.ID_TO_WORD_PATH, 'rb') as f:
            self.ID_TO_WORD = pickle.load(f)
        self.VOCAB_SIZE = len(self.ID_TO_WORD)

        self.END_ID = [k for k, v in self.ID_TO_WORD.items() if v == '<end>'][0]