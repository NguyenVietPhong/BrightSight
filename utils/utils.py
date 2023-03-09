import pickle
import os 
import face_recognition
import sys
from PIL import Image
from torchvision import transforms
import time
import sys
import numpy as np
import torch
import cv2
from PIL import Image


def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        obj = pickle.load(f)

        return obj

sys.path.append('/home/phong/system_project')
from client_triton.base.Base import BaseEmbed, BaseEncoder, BaseLinear, \
        BaseLogsoftmax, BaseLstm, BaseLstmInit, BaseYolov7

from configs.config import Config

URL = '192.168.1.140:8001'

embed = BaseEmbed()
encoder = BaseEncoder()
linear = BaseLinear()
logsoftmax = BaseLogsoftmax()
lstm = BaseLstm()
lstm_init = BaseLstmInit()
detector = BaseYolov7()


def _embed(captions, embed):
    captions = np.array(captions, dtype=np.int64)
    return embed.forwark([captions], model_name='embed')

def _encoder(images, encoder, model_name='encoder'):
    images = np.array(images, dtype=np.float32)
    return encoder.forwark([images], model_name=model_name)

def _linear(input, linear):
    input = np.array(input, dtype=np.float32)
    return linear.forwark([input], model_name='linear')

def _logsoftmax(input, linear):
    input = np.array(input, dtype=np.float32)
    return linear.forwark([input], model_name='logsoftmax')

def _lstm(input, ho, co, BaseLstm):
    input = np.array(input, dtype=np.float32)
    ho = np.array(ho, dtype=np.float32)
    co = np.array(co, dtype=np.float32)
    return lstm.forwark([input, ho, co], model_name='lstm')

def _lstm_init(input, lstm_init):
    input = np.array(input, dtype=np.float32)
    return lstm_init.forwark([input], model_name='lstm_init')

def _detector(images, detector):
    images = np.array(images, dtype=np.float32)
    return detector.forwark([images], model_name='yolov7')



class ImageCaptioningPredictor():
    def __init__(self, config):
        self.config = config
    def beam_search(self, config, image):

        config.infer()
        # EMBEDDING_DIM = self.config.EMBEDDING_DIM
        # HIDDEN_DIM = self.config.HIDDEN_DIM
        # NUM_LAYERS = self.config.NUM_LAYERS
        VOCAB_SIZE = self.config.VOCAB_SIZE
        BEAM_SIZE = self.config.BEAM_SIZE
        MAX_SEG_LENGTH = self.config.MAX_SEG_LENGTH
        ID_TO_WORD = self.config.ID_TO_WORD
        END_ID = self.config.END_ID

        device = 'cpu'  
        features = _encoder(image, encoder)
        # features = encoder.forwark([image], model_name='encoder')
        # Generate captions for given image features using beam search
        hiddens, h1, c1 = _lstm_init(features, lstm_init)

        outputs = _linear(hiddens, linear)
        outputs = np.array(outputs)
        outputs = _logsoftmax(outputs, logsoftmax)
        outputs = np.array(outputs)
        outputs = torch.tensor(outputs, device='cpu')
        outputs = outputs.squeeze(0)

        prob, predicted = outputs.max(1)

        sampled_ids = [(predicted, prob)]        
        beam = []
        for s, _ in sampled_ids:
            s = np.array(s.to('cpu'))
            em = _embed(s, embed)
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
                    hiddens, h1, c1 = _lstm(inputs, h1, c1, lstm)
                    outputs = _linear(hiddens, linear)
                    
                    outputs = _logsoftmax(outputs, logsoftmax)
                    # print(outputs.device)

                    outputs = np.array(outputs)
                    outputs = torch.tensor(outputs, device='cpu')
                    outputs = outputs.squeeze(0) + sampled_ids[i][1].to('cpu')
                    # outputs = outputs.squeeze(0) + sampled_ids[i][1]
                    h1_list.append(h1)
                    c1_list.append(c1)            
                    idxs = zip([i] * VOCAB_SIZE, list(range(VOCAB_SIZE)))           # idx: [(beam_idx, vocab_idx)] * (VOCAB_SIZE) 
                    idx_list.extend(idxs)

                    prob_list = torch.cat((prob_list, outputs[0]))

            # sorted: sorted probabilities in the descending order, indices: idx of the sorted probabilities in the descending order
            sorted, indices = torch.sort(prob_list, descending=True)                
            prob = sorted[:BEAM_SIZE]

            beam = []
            tmp_sampled_ids = []
            for i in range(BEAM_SIZE):
                word_id = torch.Tensor([indices[0]]).to('cpu').long()

                tmp_sampled_ids.append((torch.cat((sampled_ids[0][0], word_id),0), prob[i]))
                # word_id = word_id.to('cpu')
                # word_id = np.array(word_id)     
                inputs = _embed(word_id, embed)
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


def load_image_file(image_file, transform=None):
    image = Image.open(image_file)
    image = image.resize([224, 224], Image.LANCZOS)
    if transform is not None:
        image = transform(image)
    image = np.array(image, dtype=np.float32)
    # image = torch.tensor(image, device='cuda')
    return image


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def processing_img_caption(image, transform=transform):
    image = image.resize([224, 224], Image.LANCZOS)
    image = transform(image)
    # image = np.array(image, dtype=np.float32)

    return image

def processing_image_face_recogntion(frame, known_face_names, known_face_encodings):
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        # # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]
        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
    return face_names

def processing_text_face_recognition(face_names):

    if len(face_names)>1:
        text = 'In front are '
    elif len(face_names)==1:
        text = 'In front is '
    else:
        return 'No one in front'
    count = 0
    for name in face_names:
        if name != "Unknown":
            text = text + name
        else:
            count = count+1
    if count > 0:
        text = text + f'and {count} unknown'
    return text

def processing_text_detection(detected_objects):
    num_men = 0
    num_woman = 0
    num_human = len(detected_objects)
    if num_human>0:
        for box in detected_objects:
            num_woman = num_woman + box.classID
        num_men = num_human - num_woman
        text = f'there are {num_human} people in front, including {num_men} men and {num_woman} woman'
        return text
    else: 
        return 'No one in front'


