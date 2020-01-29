#!/usr/bin/env python
# coding: utf-8

# !pip install mxboard
# !pip install matplotlib
# !pip install tqdm
# !pip install opencv-python
# !pip install pandas
import json
import multiprocessing
import os
import random
import string
import time
import matplotlib.pyplot as plt
from mxboard import SummaryWriter
import mxnet as mx
from mxnet import nd, autograd, gluon
# from mxnet.gluon.model_zoo.vision import resnet34_v1
from mxnet.gluon.model_zoo.vision import resnet50_v1
import numpy as np
from skimage import transform as skimage_tf
from skimage import exposure
from tqdm import tqdm
np.seterr(all='raise')

mx.random.seed(1)

from dataloader import IAMDataset
from drawTextOnImage import draw_text_on_image

ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()

####################### BASIC CONFIGRATION #############################

__version__ = 1.5

ROOT_DIR = 'dataset' # use your image root directory
ANNOTATION_IMGS_DIR = 'images_v{}'.format(__version__)
ANNOTATION_FILE_DIR = 'annotaiton_v{}.txt'.format(__version__)

TRANING_TYPE = "line" # use your training type word or line

LOG_DIR = "./logs/handwriting_recognition" 
CHECKPOINT_DIR = "model_checkpoint"
CHECKPOINT_NAME = "handwriting.params" # model 

max_seq_len = 32 # 160 max sequence 

EPOCH = 200
LEARNING_RATE = 0.0001

BATCH_SIZE = 1

PRINT_EVERY_N = 5 #5
SEND_IMAGE_EVERY_N =5 #5
NUM_DOWNSAMPLES = 2
RESNET_LAYER_ID = 4
LSTM_HIDDEN_STATES = 512
LSTM_LAYERS = 2
SEND_IMAGE_EVERY_N = 5 #5

IMAGE_PER_EPOCH = 32000 # not use 

###################### Class set ###########################

CAPITAL_CHARACTER_SET ="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SMOLL_CHARACTER_SET = "abcdefghijklmnopqrstuvwxyz"
NUMBER_SET = "0123456789"
PUNCTUATION_SET = '$%!"#&\'()`*+,-./:;?_@=|\\'
SAPCE = " "
CLASS_SET = SAPCE+PUNCTUATION_SET+NUMBER_SET+SMOLL_CHARACTER_SET+CAPITAL_CHARACTER_SET

ALL_COMBINE = [i for i in CLASS_SET]

alphabet_dict = {ALL_COMBINE[i]:i for i in range(len(ALL_COMBINE))}

alphabet_encoding_dic = {i:ALL_COMBINE[i] for i in range(len(ALL_COMBINE))}


class EncoderLayer(gluon.HybridBlock):
    '''
    The encoder layer takes the image features from a CNN. The image features are transposed so that the LSTM 
    slices of the image features can be sequentially fed into the LSTM from left to right (and back via the
    bidirectional LSTM). 
    '''
    def __init__(self, hidden_states=200, rnn_layers=1, max_seq_len=100, **kwargs):
        self.max_seq_len = max_seq_len
        super(EncoderLayer, self).__init__(**kwargs)
        with self.name_scope():
            self.lstm = mx.gluon.rnn.LSTM(hidden_states, rnn_layers, bidirectional=True)
            
    def hybrid_forward(self, F, x):
        x = x.transpose((0, 3, 1, 2))
        x = x.flatten()
        x = x.split(num_outputs=self.max_seq_len, axis=1) # (SEQ_LEN, N, CHANNELS)
        x = F.concat(*[elem.expand_dims(axis=0) for elem in x], dim=0)
        x = self.lstm(x)
        x = x.transpose((1, 0, 2)) #(N, SEQ_LEN, HIDDEN_UNITS)
        return x

class CNNBiLSTM(gluon.HybridBlock):
    '''
    The CNN-biLSTM to recognise handwriting text given an image of handwriten text.
    Parameters
    ----------
    num_downsamples: int, default 2
        The number of times to downsample the image features. Each time the features are downsampled, a new LSTM
        is created. 
    resnet_layer_id: int, default 4
        The layer ID to obtain features from the resnet34
    lstm_hidden_states: int, default 200
        The number of hidden states used in the LSTMs
    lstm_layers: int, default 1
        The number of layers of LSTMs to use
    '''
    FEATURE_EXTRACTOR_FILTER = 64
    def __init__(self, num_downsamples=2, resnet_layer_id=4, rnn_hidden_states=200, rnn_layers=1, max_seq_len=100, ctx=mx.gpu(0), **kwargs):
        super(CNNBiLSTM, self).__init__(**kwargs)
        self.p_dropout = 0.5
        self.num_downsamples = num_downsamples
        self.max_seq_len = max_seq_len
        self.ctx = ctx
        with self.name_scope():
            self.body = self.get_body(resnet_layer_id=resnet_layer_id)

            self.encoders = gluon.nn.HybridSequential()
            with self.encoders.name_scope():
                for i in range(self.num_downsamples):
                    encoder = self.get_encoder(rnn_hidden_states=rnn_hidden_states, rnn_layers=rnn_layers, max_seq_len=max_seq_len)
                    self.encoders.add(encoder)
            self.decoder = self.get_decoder()
            self.downsampler = self.get_down_sampler(self.FEATURE_EXTRACTOR_FILTER)

    def get_down_sampler(self, num_filters):
        '''
        Creates a two-stacked Conv-BatchNorm-Relu and then a pooling layer to
        downsample the image features by half.
        
        Parameters
        ----------
        num_filters: int
            To select the number of filters in used the downsampling convolutional layer.
        Returns
        -------
        network: gluon.nn.HybridSequential
            The downsampler network that decreases the width and height of the image features by half.
        
        '''
        out = gluon.nn.HybridSequential()
        with out.name_scope():
            for _ in range(2):
                out.add(gluon.nn.Conv2D(num_filters, 3, strides=1, padding=1))
                out.add(gluon.nn.BatchNorm(in_channels=num_filters))
                out.add(gluon.nn.Activation('relu'))
            out.add(gluon.nn.MaxPool2D(2))
            out.collect_params().initialize(mx.init.Normal(), ctx=self.ctx)
        out.hybridize()
        return out

    def get_body(self, resnet_layer_id):
        '''
        Create the feature extraction network based on resnet34.
        The first layer of the res-net is converted into grayscale by averaging the weights of the 3 channels
        of the original resnet.
        
        Parameters
        ----------
        resnet_layer_id: int
            The resnet_layer_id specifies which layer to take from 
            the bottom of the network.
        Returns
        -------
        network: gluon.nn.HybridSequential
            The body network for feature extraction based on resnet
        '''
        
        pretrained = resnet50_v1(pretrained=True, ctx=self.ctx)
        pretrained_2 = resnet50_v1(pretrained=True, ctx=mx.cpu(0))
        first_weights = pretrained_2.features[0].weight.data().mean(axis=1).expand_dims(axis=1)
        # First weights could be replaced with individual channels.
        
        body = gluon.nn.HybridSequential()
        with body.name_scope():
            first_layer = gluon.nn.Conv2D(channels=64, kernel_size=(7, 7), padding=(3, 3), strides=(2, 2), in_channels=1, use_bias=False)
            first_layer.initialize(mx.init.Xavier(), ctx=self.ctx)
            first_layer.weight.set_data(first_weights)
            body.add(first_layer)
            body.add(*pretrained.features[1:-resnet_layer_id])
        return body

    def get_encoder(self, rnn_hidden_states, rnn_layers, max_seq_len):
        '''
        Creates an LSTM to learn the sequential component of the image features.
        
        Parameters
        ----------
        
        rnn_hidden_states: int
            The number of hidden states in the RNN
        
        rnn_layers: int
            The number of layers to stack the RNN
        Returns
        -------
        
        network: gluon.nn.Sequential
            The encoder network to learn the sequential information of the image features
        '''

        encoder = gluon.nn.HybridSequential()
        with encoder.name_scope():
            encoder.add(EncoderLayer(hidden_states=rnn_hidden_states, rnn_layers=rnn_layers, max_seq_len=max_seq_len))
            encoder.add(gluon.nn.Dropout(self.p_dropout))
        encoder.collect_params().initialize(mx.init.Xavier(), ctx=self.ctx)
        return encoder
    
    def get_decoder(self):
        '''
        Creates a network to convert the output of the encoder into characters.
        '''

        alphabet_size = len(alphabet_encoding_dic) + 1
        decoder = mx.gluon.nn.Dense(units=alphabet_size, flatten=False)
        decoder.collect_params().initialize(mx.init.Xavier(), ctx=self.ctx)
        return decoder

    def hybrid_forward(self, F, x):
        features = self.body(x)
        hidden_states = []
        hs = self.encoders[0](features)
        hidden_states.append(hs)
        for i, _ in enumerate(range(self.num_downsamples - 1)):
            features = self.downsampler(features)
            hs = self.encoders[i+1](features)
            hidden_states.append(hs)
        hs = F.concat(*hidden_states, dim=2)
        output = self.decoder(hs)
        return output


# ### Helper functions to train the network 

def transform(image, label):
    '''
    This function resizes the input image and converts so that it could be fed into the network.
    Furthermore, the label (text) is one-hot encoded.
    '''
    image = np.expand_dims(image, axis=0).astype(np.float32)
    if image[0, 0, 0] > 1:
        image = image/255.
    image = (image - 0.942532484060557) / 0.15926149044640417
    label_encoded = np.zeros(max_seq_len, dtype=np.float32)-1
    i = 0
    for word in label:
        word = word.replace("&quot", r'"')
        word = word.replace("&amp", r'&')
        word = word.replace('";', '\"')
        for letter in word:
            label_encoded[i] = alphabet_dict[letter]
            i += 1
    return image, label_encoded

def augment_transform(image, label):
    '''
    This function randomly:
        - translates the input image by +-width_range and +-height_range (percentage).
        - scales the image by y_scaling and x_scaling (percentage)
        - shears the image by shearing_factor (radians)
    '''
    
    ty = random.uniform(-random_y_translation, random_y_translation)
    tx = random.uniform(-random_x_translation, random_x_translation)

    sx = random.uniform(1. - random_y_scaling, 1. + random_y_scaling)
    sy = random.uniform(1. - random_x_scaling, 1. + random_x_scaling)

    s = random.uniform(-random_shearing, random_shearing)
    
    gamma = random.uniform(0.001, 2)
    image = exposure.adjust_gamma(image, gamma)
    st = skimage_tf.AffineTransform(scale=(sx, sy),
                                    shear=s,
                                    translation=(tx*image.shape[1], ty*image.shape[0]))
    augmented_image = skimage_tf.warp(image, st, cval=1.0)
    
    return transform(augmented_image*255., label)


def decode(prediction):
    '''
    Returns the string given one-hot encoded vectors.
    '''
    results = []
    for word in prediction:
        result = []
        for i, index in enumerate(word):
#             print(i,index)
            if i < len(word) - 1 and word[i] == word[i+1] and word[-1] != -1: #Hack to decode label as well
                continue
            if index == len(alphabet_dict) or index == -1:
                continue
            else:
                result.append(alphabet_encoding_dic[int(index)])
#                 result.append(alphabet_encoding[int(index)])
                
#         print(result)
        
        results.append(result)
    words = [''.join(word) for word in results]
    return words

def run_epoch(e, network, dataloader, trainer, log_dir, print_name, is_train):
    total_loss = nd.zeros(1, ctx)
    for i, (x, y) in enumerate(dataloader):
        x = x.as_in_context(ctx)
        y = y.as_in_context(ctx)

        with autograd.record(train_mode=is_train):
            output = network(x)
            loss_ctc = ctc_loss(output, y)

        if is_train:
            loss_ctc.backward()
            trainer.step(x.shape[0])

        if i == 0 and e % SEND_IMAGE_EVERY_N == 0 and e > 0:
            predictions = output.softmax().topk(axis=2).asnumpy()
            decoded_text = decode(predictions)
#             print(decoded_text)
            output_image = draw_text_on_image(x.asnumpy(), decoded_text)
            output_image[output_image < 0] = 0
            output_image[output_image > 1] = 1
            print("{} first decoded text = {}".format(print_name, decoded_text[0]))
            with SummaryWriter(logdir=log_dir, verbose=False, flush_secs=5) as sw:
                sw.add_image('bb_{}_image'.format(print_name), output_image, global_step=e)

        total_loss += loss_ctc.mean()

    epoch_loss = float(total_loss.asscalar())/len(dataloader)
    with SummaryWriter(logdir=log_dir, verbose=False, flush_secs=5) as sw:
        sw.add_scalar('loss', {print_name: epoch_loss}, global_step=e)

    return epoch_loss

def get_dataload(Traning_type,dataset_root,img_dir,annotation_file):
    print("################# Data loading ####################")
    train_ds = IAMDataset(Traning_type,dataset_root,img_dir,annotation_file, train=True)
    print("Number of training samples: {}".format(len(train_ds)))       
    test_ds = IAMDataset(Traning_type,dataset_root,img_dir,annotation_file,train=False)
    print("Number of testing samples: {}".format(len(test_ds)))

    return train_ds,test_ds

if __name__=="__main__":
    RANDOM_DATALOAD = 28000
    
    random_y_translation, random_x_translation = 0.03, 0.03
    random_y_scaling, random_x_scaling = 0.1, 0.1
    random_shearing = 0.7
    train_ds,test_ds= get_dataload(TRANING_TYPE,ROOT_DIR,ANNOTATION_IMGS_DIR,ANNOTATION_FILE_DIR)
    train_data = gluon.data.DataLoader(train_ds.transform(augment_transform), BATCH_SIZE, shuffle=True, last_batch="rollover", num_workers=4)
    test_data = gluon.data.DataLoader(test_ds.transform(transform), BATCH_SIZE, shuffle=True, last_batch="keep", num_workers=4)#, num_workers=multiprocessing.cpu_count()-2)
    print("################## Network Initialization #################")
    net = CNNBiLSTM(num_downsamples=NUM_DOWNSAMPLES, resnet_layer_id=RESNET_LAYER_ID , rnn_hidden_states=LSTM_HIDDEN_STATES, rnn_layers=LSTM_LAYERS, max_seq_len=max_seq_len, ctx=ctx)
    net.hybridize()
    ctc_loss = gluon.loss.CTCLoss(weight=0.2)
    best_test_loss = 10e5

    if (os.path.isfile(os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME))):
        net.load_parameters(os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME))
        print("Parameters loaded")
        print(run_epoch(0, net, test_data, None, LOG_DIR, print_name="pretrained", is_train=False))    
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': LEARNING_RATE})
    print("Traning Start ...................................\n")
    for e in range(EPOCH):
        train_loss = run_epoch(e, net, train_data, trainer, LOG_DIR, print_name="train", is_train=True)
        test_loss = run_epoch(e, net, test_data, trainer, LOG_DIR, print_name="test", is_train=False)    
        if test_loss < best_test_loss:
            print("Saving network, previous best test loss {:.6f}, current test loss {:.6f}".format(best_test_loss, test_loss))
            net.save_parameters(os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME))
            best_test_loss = test_loss
        if e % PRINT_EVERY_N == 0 and e > 0:
            print("Epoch {0} train_loss {1:.6f}, test_loss {2:.6f}".format(e,train_loss, test_loss))




