import os
import time
import sys
import numpy as np
# sys.path.append('./image_captioning')
# from utils import *
from config import Config
import torch
from PIL import Image
from torchvision import transforms
# triton inference server
sys.path.append('base')
from base.BaseEncoder import encoder
from base.BaseLstmInit import lstm_init
from base.BaseEmbed import embed
from base.BaseLinear import linear
from base.BaseLogsoftmax import logsoftmax
from base.BaseLstm import lstm
from tqdm import tqdm
# Choose Device
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# print("Running in %s." % device)
# """
class ImageCaptioningPredictor():
    def __init__(self, config):
        self.config = config
    def beam_search(self, config, image):

        # config.infer()
        EMBEDDING_DIM = self.config.EMBEDDING_DIM
        HIDDEN_DIM = self.config.HIDDEN_DIM
        NUM_LAYERS = self.config.NUM_LAYERS
        VOCAB_SIZE = self.config.VOCAB_SIZE
        BEAM_SIZE = self.config.BEAM_SIZE
        MAX_SEG_LENGTH = self.config.MAX_SEG_LENGTH
        ID_TO_WORD = self.config.ID_TO_WORD
        END_ID = self.config.END_ID

        device = 'cpu'  
        features = encoder(image)
        # Generate captions for given image features using beam search
        hiddens, h1, c1 = lstm_init(features)
        states = (h1, c1)

        outputs = linear(hiddens)
        outputs = np.array(outputs)
        outputs = logsoftmax(outputs)
        outputs = np.array(outputs)
        outputs = torch.tensor(outputs, device='cpu')
        outputs = outputs.squeeze(0)

        prob, predicted = outputs.max(1)

        sampled_ids = [(predicted, prob)]        
        beam = []
        for s, _ in sampled_ids:
            s = np.array(s.to('cpu'))
            em = embed(s)
            beam.append((em, h1, c1))

        for _ in range(MAX_SEG_LENGTH-1):
            h1_list = []
            c1_list = []
            prob_list = torch.tensor([]).to(device)
            idx_list = []  
            for i, (inputs, h1, c1) in enumerate(beam):       
                # If the last word is end, skip infering
                if sampled_ids[i][0][-1] == END_ID:

                    h1_list.append(h1)
                    c1_list.append(c1)
                    prob_list = torch.cat((prob_list, sampled_ids[i][1][None]))
                    idx_list.extend([i, END_ID])
                else:
                    hiddens, h1, c1 = lstm(inputs, h1, c1)
                    outputs = linear(hiddens)
                    
                    outputs = logsoftmax(outputs)
                    # print(outputs.device)

                    outputs = np.array(outputs)
                    outputs = torch.tensor(outputs, device='cpu')
                    outputs = outputs.squeeze(0) + sampled_ids[i][1].to('cpu')
                    # outputs = outputs.squeeze(0) + sampled_ids[i][1]
                    h1_list.append(h1)
                    c1_list.append(c1)            
                    idxs = zip([i] * VOCAB_SIZE, list(range(VOCAB_SIZE)))           # idx: [(beam_idx, vocab_idx)] * (VOCAB_SIZE) 
                    idx_list.extend(idxs)
                    # outputs = outputs.to('cuda')
                    # print(outputs.device)

                    prob_list = torch.cat((prob_list, outputs[0]))

                    # print(outputs)
            sorted, indices = torch.sort(prob_list, descending=True)                # sorted: sorted probabilities in the descending order, indices: idx of the sorted probabilities in the descending order
            prob = sorted[:BEAM_SIZE]

            beam = []
            tmp_sampled_ids = []
            for i in range(BEAM_SIZE):
                word_id = torch.Tensor([indices[0]]).to('cpu').long()

                tmp_sampled_ids.append((torch.cat((sampled_ids[0][0], word_id),0), prob[i]))
                # word_id = word_id.to('cpu')
                # word_id = np.array(word_id)     
                inputs = embed(word_id)
                # print(inputs.device)
                beam.append((inputs, h1_list[0], c1_list[0]))
            sampled_ids = tmp_sampled_ids
        return sampled_ids    
    
    def predict(self, image):
        with torch.no_grad():
            sampled_ids = self.beam_search(self.config, image)
        # Convert word_ids to words
        for i, (sampled_id, prob) in enumerate(sampled_ids):
            sampled_id = sampled_id.cpu().numpy()
            sampled_caption = []
            for word_id in sampled_id:
                word = self.config.ID_TO_WORD[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            sentence = ' '.join(sampled_caption)[7:-5]
            return sentence

def load_image(image_file, transform=None):
    image = Image.open(image_file)
    image = image.resize([224, 224], Image.LANCZOS)
    if transform is not None:
        image = transform(image)
    image = np.array(image, dtype=np.float32)
    # image = torch.tensor(image, device='cuda')
    return image

if __name__ == '__main__':
    img_path = "0.jpg"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    config = Config()
    config.infer()
    start = time.time()

    image = load_image(image_file=img_path, transform=transform)
    predictor = ImageCaptioningPredictor(config)

    # start = time.time()
    # for i in range(1000):
        # print(image)

    # for i in range(100):
        # predictor = ImageCaptioningPredictor(config)
    predictor.predict(image)
    end = time.time()

    print('time infer = ', (end-start))
