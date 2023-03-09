import os
import pickle
from select import select
import torch

class Config(object):
    # Set the relative path from the directory of this file
    # Settings of general (hyper)parameters
    def __init__(self):
        self.PREPARE_VOCAB = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.EMBEDDING_DIM = 256
        self.HIDDEN_DIM = 512
        self.NUM_LAYERS = 2

        self.TRAIN_CAPTION_PATH = 'annotations/captions_train2014.json'
        self.WORD_TO_ID_PATH = 'vocab/word_to_id.pkl'
        self.ID_TO_WORD_PATH = 'vocab/id_to_word.pkl'
        self.URL = '192.168.1.140:8001'
        # Change relative path to absolute path
        self.TRAIN_CAPTION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.TRAIN_CAPTION_PATH)
        self.WORD_TO_ID_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.WORD_TO_ID_PATH)
        self.ID_TO_WORD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.ID_TO_WORD_PATH)
        self.TEXT_REQUIRES_IMG_CAPTION = ['what was before', 'what is in front']
    # Settings of (hyper)parameters for training
    def train(self):
        self.LEARNING_RATE = 0.001
        self.BATCH_SIZE = 128
        self.NUM_EPOCHS = 20
        self.CHECKPOINT = None
        self.TRAIN_IMAGE_PATH = 'Images/train2014'
        self.MODEL_PATH = 'model'

        # Change relative path to absolute path
        self.TRAIN_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.TRAIN_IMAGE_PATH)
        self.MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.MODEL_PATH)

        with open(self.WORD_TO_ID_PATH, 'rb') as f:
            self.WORD_TO_ID = pickle.load(f)
        self.VOCAB_SIZE = len(self.WORD_TO_ID)

        if not(os.path.isdir(self.MODEL_PATH)):
            os.makedirs(self.MODEL_PATH)

    # Settings of (hyper)parameters for evaluation
    def eval(self):
        self.BEAM_SIZE = 5
        self.MAX_SEG_LENGTH = 20
        self.LOG_STEP = 1000
        self.NUM_EVAL_IMAGES = 40504

        self.TEST_CAPTION_PATH = 'annotations/captions_val2014.json'
        self.TEST_IMAGE_PATH = 'Images/val2014'
        self.TEST_RESULT_PATH = 'log/test_results.txt'

        self.NUM_LAYERS = 2
        self.ENCODER_PATH = 'model/encoder.pth'
        self.DECODER_PATH = 'model/decoder.pth'

        # Change relative path to absolute path
        self.TEST_CAPTION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.TEST_CAPTION_PATH)
        self.TEST_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.TEST_IMAGE_PATH)
        self.TEST_RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.TEST_RESULT_PATH)
        self.ENCODER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.ENCODER_PATH)
        self.DECODER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.DECODER_PATH)

        with open(self.ID_TO_WORD_PATH, 'rb') as f:
            self.ID_TO_WORD = pickle.load(f)
        self.VOCAB_SIZE = len(self.ID_TO_WORD)

        self.END_ID = [k for k, v in self.ID_TO_WORD.items() if v == '<end>'][0]

    # Settings of (hyper)parameters for inference
    def infer(self):
        self.BEAM_SIZE = 1
        self.MAX_SEG_LENGTH=20

        self.NUM_LAYERS = 2

        self.ENCODER_PATH = 'model/encoder.pth'
        self.DECODER_PATH = 'model/decoder.pth'

        self.INFER_IMAGE_PATH = 'data/val'
        self.INFER_RESULT_PATH = 'log/infer_results.txt'

        # Change relative path to absolute path
        self.ENCODER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.ENCODER_PATH)
        self.DECODER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.DECODER_PATH)
        self.INFER_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.INFER_IMAGE_PATH)
        self.INFER_RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.INFER_RESULT_PATH)

        with open(self.ID_TO_WORD_PATH, 'rb') as f:
            self.ID_TO_WORD = pickle.load(f)
        self.VOCAB_SIZE = len(self.ID_TO_WORD)

        self.END_ID = [k for k, v in self.ID_TO_WORD.items() if v == '<end>'][0]

    # Settings of (hyper)parameters for comparison
    def compare(self):
        self.BEAM_SIZE_LIST = [1, 5, 10] # Set beam size that you want to try
        self.MAX_SEG_LENGTH = 20
        self.NUM_COMPARE_IMAGES = 10

        self.TEST_CAPTION_PATH = 'annotations/captions_val2014.json'
        self.TEST_IMAGE_PATH = 'data/val/images'
        self.COMPARE_IMAGE_PATH = 'log/compare_images'
        self.COMPARE_RESULT_PATH = 'log/compare_results.txt'

        # Set encoder and decoder model pathes to compare
        self.ENCODER_PATH_LIST = ['model/encoder.pth']
        self.DECODER_PATH_LIST = ['model/decoder.pth']

        # Set the number of layers of the models to compare
        self.NUM_LAYERS_LIST = [2, 2]

        # Change relative path to absolute path
        self.TEST_CAPTION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.TEST_CAPTION_PATH)
        self.TEST_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.TEST_IMAGE_PATH)
        self.COMPARE_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.COMPARE_IMAGE_PATH)
        self.COMPARE_RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.COMPARE_RESULT_PATH)
        self.ENCODER_PATH_LIST = [os.path.join(os.path.dirname(os.path.abspath(__file__)), f) for f in self.ENCODER_PATH_LIST]
        self.DECODER_PATH_LIST = [os.path.join(os.path.dirname(os.path.abspath(__file__)), f) for f in self.DECODER_PATH_LIST]

        with open(self.ID_TO_WORD_PATH, 'rb') as f:
            self.ID_TO_WORD = pickle.load(f)
        self.VOCAB_SIZE = len(self.ID_TO_WORD)

        self.END_ID = [k for k, v in self.ID_TO_WORD.items() if v == '<end>'][0]
    
    def onnx(self):

        self.MAX_SEG_LENGTH = 20
        self.PATH_MODEL_DIR = "image_captioning/model_onnx"
        
        self.ENCODE_NAME = "encoder.onnx"
        self.ENC_INPUT_NAME = ['images']
        self.ENC_OUTPUT_NAME = ['features']
        self.ENC_DYNAMIC_AXES = {'images': {0: 'batch_size'}, 'features': {0: 'batch_size'}}
        
        self.DECODER_LSTM_NAME = "lstm.onnx"
        self.DEC_INPUT_LSTM_NAME = ['feature']
        self.DEC_OUTPUT_LSTM_NAME = ['output_lstm']
        self.DEC_LSTM_DYNAMIC_AXES = {'feature': {0: 'batch_size'},\
            'output_lstm': {0: 'batch_size'}}

        self.DECODER_LINEAR_NAME = "linear.onnx"
        self.DEC_INPUT_LINEAR_NAME = ['hiddens']
        self.DEC_OUTPUT_LINEAR_NAME = ['output']
        self.DEC_LINEAR_DYNAMIC_AXES = {'hiddens': {0: 'batch_size'},\
            'output': {0: 'batch_size'}}
        self.ENCODER_PATH = 'model/encoder-r34-checkpoint.pth'
        self.DECODER_PATH = 'model/decoder-r34-checkpoint.pth'
        # self.ENCODER_PATH = "model/encoder.pth"
        # self.DECODER_PATH = "model/decoder.pth"

        with open(self.ID_TO_WORD_PATH, 'rb') as f:
            self.ID_TO_WORD = pickle.load(f)
        self.VOCAB_SIZE = len(self.ID_TO_WORD)
        