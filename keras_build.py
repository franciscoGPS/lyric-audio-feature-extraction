#!/usr/bin/python

import os
import sys, getopt
import numpy as np
import ipdb
import pandas as pd
import scipy.io.wavfile as wav
from scipy.stats import rankdata
import os
import keras
import tensorflow as tf
from keras import optimizers
from keras import callbacks
import keras.backend as K
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.models import Model
from keras.layers import *
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn import pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics.scorer import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
from sklearn import svm
from numpy import argmax
from python_speech_features import mfcc
from gammatone import gtgram
from matplotlib import pyplot as plt
import skfuzzy as fuzz
import corenlp
import doc2vec_lyrics as d2v
from cca_layer import CCA


tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
filesystem_path = (os.path.abspath(os.path.join(os.path.dirname("__file__"), '../')) + '/spotify/')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess) 
MOODS = ['aggressive', 'bittersweet', 'calm', 'depressing', 'dreamy', 'fun', 'gay', 'happy', 'heavy', 'intense', 'melancholy', 'playful', 'quiet', 'quirky', 'sad', 'sentimental', 'sleepy', 'soothing', 'sweet']

SUBMOODS = ['aggressive', 'fun', 'melancholy', 'quirky']


def start_process(inputfile, samples, gtg):
   results = pd.DataFrame()
   #tf.enable_eager_execution()
   devide_mfcc = False
   calculate_mixed = False
   test_size = 0.3
   if gtg:
      (labels1,  x_mfcc1, labels_gtg, x_gtg, bow_infered, lyrics_infered, mfcc_mixed) =  extract_data(inputfile,
       devide_mfcc, gtg, samples, calculate_mixed=calculate_mixed)
      #gtg_X_train, gtg_X_test, gtg_y_train, gtg_y_test = split_data(labels_gtg, x_gtg, test_size )
   


   else:
      (labels1,  x_mfcc1, bow_infered, lyrics_infered) =  extract_data(inputfile, devide_mfcc, gtg, samples)



   concat_gtg = np.concatenate((x_gtg.squeeze(), bow_infered), axis=1)
   concat_gtg = concat_gtg.reshape(concat_gtg.shape[0], 1, concat_gtg.shape[1])
   concat_mfcc = np.concatenate((x_mfcc1.squeeze(), bow_infered), axis=1)
   if calculate_mixed:
      concat_mfcc = concat_mfcc.reshape(concat_mfcc.shape[0], 1, concat_mfcc.shape[1])
      concat_mfcc_mixed = np.concatenate((mfcc_mixed.squeeze(), bow_infered), axis=1)
      concat_mfcc_mixed = concat_mfcc_mixed.reshape(concat_mfcc_mixed.shape[0], 1, concat_mfcc_mixed.shape[1])
      gtg_X_train, gtg_X_test, gtg_y_train, gtg_y_test, bow_X_train, bow_X_test, lyr_X_train, lyr_X_test, concat_mfcc_mixed_train, concat_mfcc_mixed_test = train_test_split(x_gtg, labels_gtg, bow_infered, lyrics_infered, concat_mfcc_mixed,  test_size=test_size )
   else:
         gtg_X_train, gtg_X_test, gtg_y_train, gtg_y_test, bow_X_train, bow_X_test, lyr_X_train, lyr_X_test = train_test_split(x_gtg, labels_gtg, bow_infered, lyrics_infered,  test_size=test_size )

   
   #mfcc_mixed = mfcc_mixed.reshape(mfcc_mixed.shape[0], 1, mfcc_mixed.shape[1])


   ipdb.set_trace()
   train_dcca(x_gtg.squeeze(), labels_gtg.squeeze(), bow_infered.squeeze(), labels_gtg.squeeze(), 300, batch_size=20)
   #evaluate_dataset(mfcc1_X_train, mfcc1_y_train,  mfcc1_X_test , mfcc1_y_test)

   #evaluate_dataset( gtg_X_train, gtg_y_train, gtg_X_test , gtg_y_test)



   #history = train_LSTM_model(concat_gtg, labels_gtg, 50)
   """
   epochs = 50
   hidden_size = gtg_y_train.shape[1]

   history2, model = train_RNNLSTM_model(concat_mfcc_mixed_train, gtg_y_train, hidden_size, epochs)
   test_model(concat_mfcc_mixed_test, gtg_y_test, model)
   plot_results(history2, epochs)


   ipdb.set_trace()

   input_sequences = Input(shape=(1,hidden))
   processed_sequences = TimeDistributed(audio_model)(input_sequences)
   lstm_out = LSTM(32)(processed_sequences)
   mergedOut = Concatenate()([audio_model.output,lyrics_model.output])
   """

   
   #scikitmodel = create_LSTM_model(mfcc1_X_train, mfcc2_y_train, hidden_size) 
   #, fit_params = dict(callbacks=[tbCallBack, early_stopping])
   


   """

   # evaluate using 10-fold cross validation
   kfold = StratifiedKFold(n_splits=2, shuffle=True)
   results = cross_val_score(best_model, mfcc1_X_train, mfcc1_y_train, cv=kfold)
   print(results.mean())
   
  
   early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')

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


def train_dcca(X1, y1, X2, y2, epoch_num, batch_size=200):

   test_size = 0.4  

   X1_train, X1_test, y1_train, y1_test, X2_train, X2_test, y2_train, y2_test = train_test_split(X1, y1, X2, y2, test_size=test_size)
   X1_test, X1_valid, y1_test, y1_valid, X2_test, X2_valid, y2_test, y2_valid = train_test_split(X1_test, y1_test, X2_test, y2_test, test_size=0.5)

   ipdb.set_trace()

   input_shape1= X1_train.shape[1]
   input_shape2= X2_train.shape[1]

   input1 = Input(shape=(input_shape2, ), name='audios')
   input2 = Input(shape=(input_shape2, ), name='lyrics')


   activation_model = 'sigmoid'
   dense1_1 = Dense(1024, activation=activation_model, name='view_1_1')(input1)
   dense1_2 = Dense(1024, activation=activation_model, name='view_1_2')(dense1_1)
   dense1_3 = Dense(1024, activation=activation_model,  name='view_1_3')(dense1_2)
   output1 = Dense(10, activation='linear', name='view_1_4')(dense1_3)

   dense2_1 = Dense(1024, activation=activation_model,  name='view_2_1')(input2)
   dense2_2 = Dense(1024, activation=activation_model,  name='view_2_2')(dense2_1)
   dense2_3 = Dense(1024, activation=activation_model, name='view_2_3')(dense2_2)
   output2 = Dense(10, activation='linear', name='view_2_4')(dense2_3)

   shared_layer = concatenate([output1, output2], name='shared_layer')

   cca_layer = CCA(1, name='cca_layer')(shared_layer)

   model = Model(inputs=[input1, input2], outputs=cca_layer)
   model.compile(optimizer='rmsprop', loss="mean_squared_error", metrics=[mean_pred])
   hist = model.fit([X1_train, X2_train], np.zeros(len(X1_train)), batch_size=batch_size, epochs=epoch_num, shuffle=True, verbose=1,validation_data=([X1_valid, X2_valid], np.zeros(len(X2_valid))))
   
   current_dcca = Model(input=model.input, output=model.get_layer(name='shared_layer').output)
   pred_out = current_dcca.predict([X1_test, X2_test])
   half = int(pred_out.shape[1] / 2)
   current_expert_data.append([pred_out[:, :half], pred_out[:, half:]])

   ##TODO: Add SVM classification here

   


def ranking_average_prescision(y, y_pred_probs):
   cntr, uc, u0c, dc, jm, pcd, fpc = fuzz.cluster.cmeans(X.T, 19, 2, error=0.005, maxiter=1000, init=None)
   u0p, up, dp, jm, pp, fpc =  fuzz.cluster.cmeans_predict(up, cntr, 4, error=0.0005, maxiter=1000)
   ranking_score = label_ranking_average_precision_score(y, u0p.T)

def ranking_precision(y, y_pred_probs):
   presicion = 0
   for test in range(len(y)):
      presicion += 1 - rankdata(y_pred_probs[test])[np.argmax(y[test])-1]/len(y_pred_probs[test])

   score = presicion/len(y_pred_probs)

   return score

def evaluate_dataset(X_train, y_train, X_test, y_test):
   early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=2, mode='auto')
   
   model_classifier = KerasClassifier(build_fn=create_BLSTM_model, epochs=50, batch_size=10,
    verbose=1, validation_split=0.3)


   my_scorer = make_scorer(ranking_precision, greater_is_better=True, needs_proba=True)

   size = X_train.shape[2]
   validator = GridSearchCV(model_classifier,
                         param_grid={# epochs is avail for tuning even when not
                                     # an argument to model building function
                                     #'input_size': [int(mfcc1_X_train.shape[2])],
                                     'epochs': [ 100, 120],
                                     'hidden_size':[50 , 75, size, size*2],
                                     'dropout': [0.5,0.6],
                                     'size': [size]
                                     },
                         scoring=my_scorer,
                         n_jobs=1)
   
   validator.fit(X_train, y_train)

   print('The parameters of the best model are: ')
   print(validator.best_params_)

   # validator.best_estimator_ returns sklearn-wrapped version of best model.
   # validator.best_estimator_.model returns the (unwrapped) keras model
   best_model = validator.best_estimator_.model
   best_model.summary()

   metric_names = best_model.metrics_names
   metric_values = best_model.evaluate(X_test, y_test)
   

   for metric, value in zip(metric_names, metric_values):
       print(metric, ': ', value)
   
   
def get_lyrics_dict():

   files = [os.path.join(filesystem_path,fn) for fn in os.listdir(filesystem_path) if fn.endswith("integral_synced_SongsMoodsFile.csv")]
   audios_data = pd.read_csv(files[0], delimiter=',', encoding="utf-8",quotechar='"')
   df = pd.DataFrame.from_dict(audios_data, orient='columns')
   return df


def get_lyrics(video_id):
   query_result = lyrycs_df.query('youtube_video_id ==  @video_id')
   bow = query_result['bow'].values[0]
   lyric = query_result['lyric'].values[0]

def extract_data(audioset_path, devide_mfcc=True, calculate_gtg=True, samples=-1, calculate_mixed=False):

   model_directory = os.path.dirname(os.path.realpath(__file__))+ "/apnews_dbow/"

   saved_files = [os.path.join(audioset_path,fn) for fn in os.listdir(audioset_path) if fn.endswith('.wav')]



   labels_1 = []
   labels_2 = []
   labels_gtg = []
   lyrics_infered = []
   bow_infered = []
   tokens = []
   x_mfcc_1 = []
   x_mfcc_2 = []
   x_gtg = []
   x_mixed = []
   counter = 0
   errors = {"error":[], "counter":[]}
   twin = 0.3
   thop = 0.3
   channels = 3
   fmin = 20

   doc2vec = d2v.doc2vec_model(model_directory, model_name="doc2vec.bin", lyrics_filename="integral_synced_SongsMoodsFile.csv")
   
   doc2vec_model = doc2vec.load_model()


   lyrycs_df = get_lyrics_dict()
   for audiopath in saved_files[:samples]:

      try:
        # if audiop5000.split("/")[-1].split("_")[0] in ['aggressive', 'happy', 'sad', 'sweet']:
         (rate,sig) = wav.read(audiopath)
         duration = len(sig) / rate
         if duration >= 30:
            #signal, rate, freq = 0.5, numcep=1, winlen = 1, winstep = 0.5 #
            audio_name = audiopath.split("/")[-1]
            mood = audio_name.split("_")[0]
            if mood in MOODS:
               video_id = audio_name[-15:-4]
               query_result = lyrycs_df.query('youtube_video_id ==  @video_id')
               lyc = query_result['lyric'].values[0]
               
               mfcc_1 = get_mfcc(sig, rate, 1, channels, twin, thop)
               print("mfcc_1.shape:", mfcc_1.shape)
               
               bow = query_result['bow'].values[0]
               # Tokenize, its not working. We already have cleaned lyrics.
               # client =  corenlp.CoreNLPClient(annotators="tokenize ssplit sentiment".split())
               #tokenzed_lyc = client.annotate(lyc)
               #tokenzed_bow = client.annotate(bow)
               bow_infered.append(doc2vec_model.infer_vector(bow))
               lyrics_infered.append(doc2vec_model.infer_vector(lyc))


               if calculate_gtg:
                  
                  #gtg = get_gtg(rate, sig, 0.3, thop, 3, fmin)
                  gtg = get_gtg(rate, sig, twin, thop, channels, fmin)
                  
                  print("gtg.shape: ", gtg.shape)
                  labels_gtg.append(int(MOODS.index(mood)))
                  x_gtg.append(np.array(gtg.reshape(1, gtg.size)).astype(np.float32))

                  if calculate_mixed:
                     #gtg_to_mfcc = get_gtg(rate, sig, twin, thop, 4, fmin)
                     mfcc_gtg = get_mfcc(gtg, rate, 0.3, 26, 1, .5)
                     print("mfcc_gtg.shape", mfcc_gtg.shape)
                     x_mixed.append(np.array(mfcc_gtg.reshape(1, mfcc_gtg.size)).astype(np.float32))
               

               

               if devide_mfcc:
                  for seq in range(len(mfcc_1)):
                     x_mfcc_1.append(np.array(mfcc_1[seq].reshape(1, mfcc_1[seq].size)).astype(np.float32))
                     labels_1.append(int(MOODS.index(mood)))
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
   bow_infered = np.array(bow_infered).astype(np.float32)
   lyrics_infered = np.array(lyrics_infered).astype(np.float32)
   x_mfcc_np1 = np.array(x_mfcc_1).astype(np.float32)



   #x_mfcc_np2 = np.array(x_mfcc_2).astype(np.float32)
   labels_onehot1 = (np.arange(19) == labels_np1[:, None]).astype(np.float32)

   #If already finished training a model (=no more updates, only querying, reduce memory usage
   doc2vec_model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

   if calculate_gtg:
      labels_gtg = np.array(labels_gtg).astype(dtype=np.uint8)
      x_gtg_np = np.array(x_gtg).astype(np.float32)
      x_mixed = np.array(x_mixed).astype(np.float32)
      labels_onehotgtg = (np.arange(19) == labels_gtg[:, None]).astype(np.float32)
      return labels_onehot1, x_mfcc_np1, labels_onehotgtg, x_gtg_np, bow_infered, lyrics_infered, x_mixed
   #labels_onehot2 = (np.arange(19) == labels_np2[:, None]).astype(np.float32)
   else:
      return labels_onehot1, x_mfcc_np1, bow_infered, lyrics_infered



   

def test_model(X_test, y_test, model):
   
   correct = 0
   yhat = model.predict_classes(X_test, verbose = 0)
   #labels_np = np.array(yhat).astype(dtype=np.uint8)
   #labels_onehot = (np.arange(19) == labels_np[:, None]).astype(np.float32)
   scores = model.evaluate(X_test, y_test, verbose = 0)
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
   history = model.fit(X, y, epochs=100, verbose=0, batch_size=20, callbacks=[tbCallBack], shuffle=True)

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

def audio_dnn(X, y):
   
   print(X.shape)
   print(y.shape)
   model = Sequential()
   model.add(Dense(1024, input_shape=(X.squeeze().shape[1],), activation="sigmoid" ))
   # now the model will take as input arrays of shape (*, 16)
   # and output arrays of shape (*, 32)

   #model.add(Activation('sigmoid'))
   model.add(Dense(1024, activation="sigmoid"))
   #model.add(Activation('sigmoid'))
   model.add(Dense(y.shape[1], activation='linear'))

   model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])  
   model.fit(X.squeeze(), y, epochs=10, verbose=0, batch_size=20, callbacks=[tbCallBack], shuffle=True)



   return model


# Function to create model, required for KerasClassifier
def create_BLSTM_model(hidden_size, dropout, size):
   # create model
   #hidden_size=50
   #input_size=116

   
   model = Sequential()
   model.add(Bidirectional(LSTM(hidden_size, stateful=False), input_shape=(1, size) ))
   model.add(Dropout(dropout))
   model.add(Dense(19, activation='softmax'))
   # Compile model
   model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])  
   
   return model



def train_LSTM_model(X, y, hidden_size, epochs): 
   loss = list()
   model = Sequential()
   print(X.shape)
   print(y.shape)
   
   model.add(Bidirectional(LSTM(hidden_size, stateful=False), input_shape=(1, X.shape[2])))
   model.add(Dropout(0.5))
   model.add(Dense(19, activation='sigmoid'))
   model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])  
   history = model.fit(X, y, epochs=epochs, verbose=2, batch_size=20, callbacks=[tbCallBack], shuffle=True, validation_split=0.2)
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
   return history, model

def train_RNNLSTM_model(X, y, hidden_size, epochs): 
   loss = list()
   model = Sequential()
   print(X.shape)
   print(y.shape)
   
   model.add(SimpleRNN(hidden_size))
   model.add(Dropout(0.5))
   model.add(Dense(19, activation='relu'))
   model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])  
   history = model.fit(X, y, epochs=epochs, verbose=2, batch_size=20, callbacks=[tbCallBack], shuffle=True, validation_split=0.2)
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
   return history, model


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


def plot_results(hist, epochs):
   xc = range(epochs)

   train_loss = hist.history['loss']
   val_loss = hist.history['val_loss']
   train_acc = hist.history['acc']
   val_acc = hist.history['val_acc']

   #First image
   plt.figure(1, figsize=(7,5))
   plt.plot(xc, train_loss)
   plt.plot(xc, val_loss)
   plt.xlabel("Num. Epochs")
   plt.ylabel("Loss")
   plt.title("train_loss vs val_loss")
   plt.grid(True)
   plt.legend(["train", "val"])
   plt.style.use(['classic'])

   #Second image
   plt.figure(2, figsize=(7,5))
   plt.plot(xc, train_acc)
   plt.plot(xc, val_acc)
   plt.xlabel("Num. Epochs")
   plt.ylabel("Loss")
   plt.title("train_acc vs val_acc")
   plt.grid(True)
   plt.legend(["train", "val"], loc=4)
   plt.style.use(['classic'])
   plt.show()


def constant_loss(y_true, y_pred):
    return y_pred

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


def svm_classify(data, C):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """
    train_data, _, train_label = data[0]
    valid_data, _, valid_label = data[1]
    test_data, _, test_label = data[2]

    print('training SVM...')
    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(train_data, train_label.ravel())

    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)
    p = clf.predict(valid_data)
    valid_acc = accuracy_score(valid_label, p)

    return [test_acc, valid_acc]


def main(argv):
   inputfile = ''
   samples = -1
   gtg = True
   try:
      opts, args = getopt.getopt(argv,"hi:s:",["ifile=", "samples="])
   except getopt.GetoptError:
      print( 'keras_build.py -i <inputfile> -s <samples=-1> ')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print( 'test.py -i <inputfile> ')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg.strip()
   
      elif opt in ("-s","--samples"):
         samples = int(arg.strip())
   start_process(inputfile, samples, gtg)

if __name__ == "__main__":
   main(sys.argv[1:])
   

