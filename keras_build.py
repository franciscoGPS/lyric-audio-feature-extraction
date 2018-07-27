#!/usr/bin/python

import sys, getopt
from keras.models import Sequential
from keras.layers import Activation
import ipdb
import numpy as np
import pandas as pd
import os
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from gammatone import gtgram
import tensorflow as tf
from sklearn.cross_validation import train_test_split

def split_data(labels, X, size):
   X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=size )
   return X_train, X_test, y_train, y_test

   

def extract_data(audioset_path):
   MOODS = ['aggressive', 'bittersweet', 'calm', 'depressing', 'dreamy', 'fun', 'gay', 'happy', 'heavy', 'intense', 'melancholy', 'playful', 'quiet', 'quirky', 'sad', 'sentimental', 'sleepy', 'soothing', 'sweet']
   saved_files = [os.path.join(audioset_path,fn) for fn in os.listdir(audioset_path) if fn.endswith('.wav')]
   labels = []
   x_mfcc = []
   x_gtg = []
   counter = 0
   errors = {"error":[], "counter":[]}
   for audiopath in saved_files[:30]:
      
      if counter == 28:
            ipdb.set_trace()
      
      try:
         
         (rate,sig) = wav.read(audiopath)
         duration = len(sig) / rate
         if duration >= 30:
            mfcc, gtg = get_features(rate, sig)
                    
            mood = audiopath.split("/")[-1].split("_")[0]
            labels.append(int(MOODS.index(mood)))
            x_mfcc.append(mfcc.reshape(mfcc.size,))
            x_gtg.append(gtg.reshape(gtg.size,))
      except Exception as e:
         ipdb.set_trace()
         errors['error'].append(e)
         errors['counter'].append(e)
      print(counter)
      counter+=1   
      
         
   labels_np = np.array(labels).astype(dtype=np.uint8)
   x_mfcc_np = np.matrix(x_mfcc).astype(np.float32) 
   x_gtg_np = np.matrix(x_gtg).astype(np.float32)
   #MOODS = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
   """
   labels_np = np.matrix(labels)
   """
   labels_onehot = (np.arange(19) == labels_np[:, None]).astype(np.float32)

   return labels_onehot, x_mfcc_np, x_gtg_np

def start_process(inputfile):
   tf.enable_eager_execution()
   (labels, x_mfcc, x_gtg ) =  extract_data(inputfile)
   test_size = 0.5

   dataset1 = tf.data.Dataset.from_tensor_slices((x_mfcc, labels)).shuffle(500)
   dataset2 = tf.data.Dataset.from_tensor_slices((x_gtg, labels)).shuffle(500)

   mfcc_X_train, mfcc_X_test, y_train, y_test = split_data(labels, x_mfcc, test_size)
   gtg_X_train, gtg_X_test, gtg_y_train, gtg_y_test = split_data(labels, x_gtg, test_size )



   ipdb.set_trace()
   train_gtg_model(labels, x_mfcc)

   train_mfcc_model(labels, x_gtg )


   
def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print( 'keras_build.py -i <inputfile> ')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print( 'test.py -i <inputfile> ')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg.strip()
      elif opt in ("-o", "--ofile"):
         outputfile = arg.strip()

   #print( 'Input file is "', inputfile+'"')
   #print( 'Output file is "', outputfile+'"')
   start_process(inputfile)

def get_features(rate, signal):
   twin = 0.250
   thop = twin/2
   channels = 2
   fmin = 20
   duration = len(signal) / rate
   mfcc_feat = mfcc(signal,nfilt=40, samplerate=rate, winlen=twin, winstep=thop,nfft=5515, winfunc=np.hamming )
   mfcc_feat = mfcc_feat.reshape(mfcc_feat.size, )
   gtg = gtgram.gtgram(signal, rate, twin, thop, channels, fmin)
   
   gtg = np.flipud(20 * np.log10(gtg))
   gtg = gtg.reshape(gtg.size, )


   print("mfcc_feat.shape", mfcc_feat.shape)
   print("gtg shape", gtg.shape )
   return mfcc_feat, gtg

def build_model(X_train, y_train):
   
   model = Sequential()

   #cifar10_dir = 'datasets/cifar-10-batches-py'
   #X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
   ##
   model.add(Dense(units=64, input_dim=X_train.shape[1]))
   model.add(Activation('relu'))
   model.add(Dense(units=10))
   model.add(Activation('softmax'))
   model.compile(loss='categorical_crossentropy',
      optimizer='sgd',
      metrics=['accuracy'])
   model.fit(X_train, y_train, epochs=5, batch_size=32)  


def train_model(X_train, y_train):
   pass



if __name__ == "__main__":
   main(sys.argv[1:])
   

