#!/usr/bin/python

import os
import sys, getopt
import numpy as np
import ipdb
import pandas as pd
import scipy.io.wavfile as wav
import os
import keras
import tensorflow as tf
from keras import optimizers
from keras import callbacks
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn import pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
from numpy import argmax
from python_speech_features import mfcc
from gammatone import gtgram
from matplotlib import pyplot as plt
import skfuzzy as fuzz

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
filesystem_path = (os.path.abspath(os.path.join(os.path.dirname("__file__"), '../')) + '/spotify/')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess) 
MOODS = ['aggressive', 'bittersweet', 'calm', 'depressing', 'dreamy', 'fun', 'gay', 'happy', 'heavy', 'intense', 'melancholy', 'playful', 'quiet', 'quirky', 'sad', 'sentimental', 'sleepy', 'soothing', 'sweet']



def start_process(inputfile):
   results = pd.DataFrame()
   #tf.enable_eager_execution()
   devide_mfcc = False
   (labels1,  x_mfcc1) =  extract_data(inputfile, devide_mfcc)
   test_size = 0.3
   hidden_size = 50
   mfcc1_X_train, mfcc1_X_test, mfcc1_y_train, mfcc1_y_test = split_data(labels1, x_mfcc1, test_size)
   num_classes = 19

   print(mfcc1_X_train.shape)

   """
   mfcc2_X_train, mfcc2_X_test, mfcc2_y_train, mfcc2_y_test = split_data(labels2, x_mfcc2, test_size )
   gtg_X_train, gtg_X_test, gtg_y_train, gtg_y_test = split_data(labels_gtg, x_gtg, test_size )

   gtg_model_1 = Sequential()
   results['gtg'] = train_double_LSTM(gtg_X_train, gtg_y_train, gtg_model_1, hidden_size)
   results['gtg_test'] = test_model(gtg_X_test, gtg_y_test, gtg_model_1)
   
   mfcc_model_1a = Sequential()
   results['mfcc1'] = train_double_LSTM(mfcc1_X_train, mfcc1_y_train, mfcc_model_1a, hidden_size)
   results['mfcc1_test'] = test_model(mfcc1_X_test, mfcc1_y_test, mfcc_model_1a)

   
   mfcc_model2a = Sequential()
   results['mfcc2']  = train_double_LSTM(mfcc2_X_train, mfcc2_y_train, mfcc_model2a, hidden_size)
   results['mfcc2_test'] = test_model(mfcc2_X_test, mfcc2_y_test, mfcc_model2a)
   
   



   
   #alldata = np.vstack((mfcc1_X_train, mfcc1_y_train))
   mfcc1_X_trainT = mfcc1_X_train.reshape(mfcc1_X_train.shape[0],mfcc1_X_train.shape[2]).T
   mfcc2_X_trainT = mfcc2_X_train.reshape(mfcc2_X_train.shape[0],mfcc2_X_train.shape[2]).T
   mfcc2_X_testT = mfcc2_X_test.reshape(mfcc2_X_test.shape[0], mfcc2_X_test.shape[2]).T
   cntr, uc, u0c, dc, jm, pc, fpc = fuzz.cluster.cmeans(mfcc2_X_trainT, 19, 2, error=0.005, maxiter=1000, init=None)
   u0p, up, dp, jm, pp, fpc fuzz.cluster.cmeans_predict(mfcc2_X_testT, cntr, 4, error=0.0000005, maxiter=1000)
   ipdb.set_trace()
   

   gtg_model_2b = Sequential()
   results['gtg_b'] = train_LSTM_model(gtg_X_train, gtg_y_train, gtg_model_2b, hidden_size)
   results['gtg_test_b'] = test_model(gtg_X_test, gtg_y_test, gtg_model_2b)
   
   mfcc_model_2b = Sequential()
   results['mfcc1_b'] = train_LSTM_model(mfcc1_X_train, mfcc1_y_train, mfcc_model_2b, hidden_size)
   results['mfcc1_test_b'] = test_model(mfcc1_X_test, mfcc1_y_test, mfcc_model_2b)

   
   mfcc_model_2b = Sequential()
   history = train_LSTM_model(mfcc2_X_train, mfcc2_y_train, mfcc_model_2b, hidden_size)
   results['mfcc2_test_b'] = test_model(mfcc2_X_test, mfcc2_y_test, mfcc_model_2b)
   
   plt.plot(history.history['loss'])
   plt.title('Bidirectional LSTM loss')
   plt.ylabel('loss')
   plt.xlabel('epoch')
   plt.legend(['train', 'test'], loc='upper left')
   model.summary()
   #plt.show()
   plt.show()
   """

   
   #scikitmodel = create_LSTM_model(mfcc1_X_train, mfcc2_y_train, hidden_size)
   model_classifier = KerasClassifier(build_fn=create_BLSTM_model, epochs=150, batch_size=10, verbose=1)
   validator = GridSearchCV(model_classifier,
                         param_grid={# epochs is avail for tuning even when not
                                     # an argument to model building function
                                     #'input_size': [int(mfcc1_X_train.shape[2])],
                                     'epochs': [10, 50, 100],
                                     #'hidden_size':[25,50,100],
                                     },
                         scoring='neg_log_loss',
                         n_jobs=1)
   validator.fit(mfcc1_X_train, mfcc1_y_train)

   print('The parameters of the best model are: ')
   print(validator.best_params_)

   # validator.best_estimator_ returns sklearn-wrapped version of best model.
   # validator.best_estimator_.model returns the (unwrapped) keras model
   best_model = validator.best_estimator_.model
   best_model.summary()
   metric_names = best_model.metrics_names
   metric_values = best_model.evaluate(mfcc1_X_test, mfcc1_y_test)
   for metric, value in zip(metric_names, metric_values):
       print(metric, ': ', value)
   ipdb.set_trace()

   """
   # evaluate using 10-fold cross validation
   kfold = StratifiedKFold(n_splits=2, shuffle=True)
   results = cross_val_score(best_model, mfcc1_X_train, mfcc1_y_train, cv=kfold)
   print(results.mean())
   
  

   #early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')

   pipe = pipeline.Pipeline([
       ('nn', KerasClassifier(build_fn=create_BLSTM_model, nb_epoch=100, batch_size=15,
                              validation_split=0.2, callbacks=[ tbCallBack]))
   ])


   pipe.fit(mfcc1_X_train , mfcc1_y_train )

   #pipe.steps.append(('nn', model))

   pred = pipe.predict_proba(mfcc1_X_test)[:, 0]



   directory = os.path.dirname(os.path.realpath(__file__))
   model_step = pipe.steps.pop(-1)[1]
   joblib.dump(pipe, os.path.join(directory, 'pipeline.pkl'))
   models.save_model(model_step.model, os.path.join(directory, 'model.h5'))
   """
   ipdb.set_trace

   
def get_lyrics_dict():

   files = [os.path.join(filesystem_path,fn) for fn in os.listdir(filesystem_path) if fn.endswith("integral_synced_SongsMoodsFile.csv")]
   audios_data = pd.read_csv(files[0], delimiter=',', encoding="utf-8",quotechar='"')
   df = pd.DataFrame.from_dict(audios_data, orient='columns')
   return df


def get_lyrics(video_id):
   query_result = lyrycs_df.query('youtube_video_id ==  @video_id')
   bow = query_result['bow'].values[0]
   lyric = query_result['lyric'].values[0]

def extract_data(audioset_path, devide_mfcc):
   saved_files = [os.path.join(audioset_path,fn) for fn in os.listdir(audioset_path) if fn.endswith('.wav')]
   labels_1 = []
   labels_2 = []
   labels_gtg = []
   x_mfcc_1 = []
   x_mfcc_2 = []
   x_gtg = []
   counter = 0
   errors = {"error":[], "counter":[]}
   twin = 0.5
   thop = .250
   channels = 4
   fmin = 20


   #lyrycs_df = get_lyrics_dict()
   


   for audiopath in saved_files[:10]:
      
      try:

        # if audiop5000.split("/")[-1].split("_")[0] in ['aggressive', 'happy', 'sad', 'sweet']:
         
         (rate,sig) = wav.read(audiopath)
         duration = len(sig) / rate

         if duration >= 30:
            #signal, rate, freq = 0.5, numcep=1, winlen = 1, winstep = 0.5 #
            mfcc_1 = get_mfcc(sig, rate, 1, 4, 2, 1)
            print("mfcc_1.shape:", mfcc_1.shape)

            """
            gtg = get_gtg(rate, sig, twin, thop, channels, fmin)
            print("gtg.shape: ", gtg.shape)
            gtg_to_mfcc = get_gtg(rate, sig, twin, thop, 4, fmin)
            mfcc_2 = get_mfcc(gtg_to_mfcc, rate, 0.3, 13, 1, .5)
            print("mfcc_2.shape", mfcc_2.shape)
            labels_gtg.append(int(MOODS.index(mood)))
            x_gtg.append(np.array(gtg.reshape(1, gtg.size)).astype(np.float32))

            """


            audio_name = audiopath.split("/")[-1]
            mood = audio_name.split("_")[0]
            video_id = audio_name[-15:-4]
            
            
            if devide_mfcc:
               for seq in range(len(mfcc_1)):
                  
                  x_mfcc_1.append(np.array(mfcc_1[seq].reshape(1, mfcc_1[seq].size)).astype(np.float32))
            else:
               labels_1.append(int(MOODS.index(mood)))
   
               x_mfcc_1.append(np.array(mfcc_1.reshape(1, mfcc_1.size)).astype(np.float32))
               #x_gtg.append(np.array(gtg.reshape(1, gtg.size)).astype(np.float32))
               #x_gtg.append(gtg.reshape(gtg.size,1))
      except Exception as e:
         ipdb.set_trace()
         errors['error'].append(e)
         errors['counter'].append(e)
      print(counter)
      counter+=1   
      
      
   labels_np1 = np.array(labels_1).astype(dtype=np.uint8)
   #labels_np2 = np.array(labels_2).astype(dtype=np.uint8)
   #labels_gtg = np.array(labels_gtg).astype(dtype=np.uint8)

   
   x_mfcc_np1 = np.array(x_mfcc_1).astype(np.float32)
   #x_mfcc_np2 = np.array(x_mfcc_2).astype(np.float32)
   #x_gtg_np = np.array(x_gtg).astype(np.float32)

   labels_onehot1 = (np.arange(19) == labels_np1[:, None]).astype(np.float32)
   #labels_onehot2 = (np.arange(19) == labels_np2[:, None]).astype(np.float32)
   #labels_onehotgtg = (np.arange(19) == labels_gtg[:, None]).astype(np.float32)

   return labels_onehot1, x_mfcc_np1



   

def test_model(X_test, y_test, model):
   
   correct = 0
   yhat = model.predict_classes(X_test, verbose=1)
   #labels_np = np.array(yhat).astype(dtype=np.uint8)
   #labels_onehot = (np.arange(19) == labels_np[:, None]).astype(np.float32)
   scores = model.evaluate(X_test, y_test, verbose=1)
   for i in range(X_test.shape[0]):
      if(np.argmax(y_test[i]) == yhat[i]):
         print('Expected:', MOODS[np.argmax(y_test[i])], 'Predicted', MOODS[yhat[i]])
         correct+=1
         print()


   model.summary()
   accuracy = correct/X_test.shape[0]
   print("accuracy: %.6f" % accuracy*100 )
   print("Accuracy: %.2f%%" % (scores[1]*100))
   return accuracy
   
   


def train_double_LSTM(X, y, model, hidden_size):
   loss = list()
   print(X.shape)
   print(y.shape)
   
   model.add(LSTM(hidden_size, return_sequences=True, stateful=False, input_shape=(1, X.shape[2])))
   model.add(Dropout(0.5))
   #model.add(LSTM(hidden_size, return_sequences=False))
   #model.add(Dropout(0.2))
   model.add(Dense(19, activation='softmax'))
   model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])  
   history = model.fit(X, y, epochs=100, verbose=1, batch_size=20, callbacks=[tbCallBack], shuffle=True)

   # summarize history for loss
   plt.plot(history.history['loss'])
   """
   plt.title('double LSTM model loss')
   plt.ylabel('loss')
   plt.xlabel('epoch')
   plt.legend(['train', 'test'], loc='upper left')
   """
   model.summary()
   #plt.show()
   return history



# Function to create model, required for KerasClassifier
def create_BLSTM_model():
   # create model

   hidden_size=50
   input_size=116
   
   
   model = Sequential()
   model.add(Bidirectional(LSTM(hidden_size, stateful=False), input_shape=(1,input_size)))
   model.add(Dropout(0.5))
   model.add(Dense(19, activation='softmax'))
   # Compile model
   model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])  
   
   return model



def train_LSTM_model(X, y, model, hidden_size): 
   loss = list()
   print(X.shape)
   print(y.shape)
   ipdb.set_trace()
   model.add(Bidirectional(LSTM(hidden_size, stateful=False), input_shape=(1, X.shape[2])))
   model.add(Dropout(0.5))
   model.add(Dense(19, activation='softmax'))
   model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])  
   history = model.fit(X, y, epochs=100, verbose=1, batch_size=20, callbacks=[tbCallBack], shuffle=True)
   """
   loss.append(history.history['loss'])
   plt.plot(history.history['loss'])
   plt.title('Bidirectional LSTM loss')
   plt.ylabel('loss')
   plt.xlabel('epoch')
   plt.legend(['train', 'test'], loc='upper left')
   """
   model.summary()
   #plt.show()
   return history


def get_gtg(rate, signal, twin, thop, channels, fmin):


   gtg = gtgram.gtgram(signal, rate, twin, thop, channels, fmin)
   Z = np.flipud(20 * np.log10(gtg))
   gtg = Z.reshape(1, Z.size)
   return gtg


def get_mfcc(signal, rate, freq = 0.5, numcep=1, twin = 1, thop = 0.5 ):
   return mfcc(signal, samplerate=rate*freq, numcep=numcep, winlen=twin,winstep=thop )


def split_data(labels, X, size):
   X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=size )
   return X_train, X_test, y_train, y_test

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
   start_process(inputfile)

if __name__ == "__main__":
   main(sys.argv[1:])
   

