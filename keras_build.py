#!/usr/bin/python

import os
import sys, getopt
import random
import numpy as np
import ipdb
import pandas as pd
import scipy.io.wavfile as wav
from scipy.stats import rankdata
import pickle
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
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn import pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics.scorer import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn import svm
from numpy import argmax
from python_speech_features import mfcc
from gammatone import gtgram
from matplotlib import pyplot as plt
import skfuzzy as fuzz
import corenlp
import doc2vec_lyrics as d2v
from cca_layer import CCA
import json

from DeepCCA.utils import load_data, svm_classify
from DeepCCA.linear_cca import linear_cca
from DeepCCA.models import build_BLSTM_model, build_DRNN_model, build_GRU_model, create_CNN_RNN
from DeepCCA.DeepCCA import train_model, test_model as test_dcca, train_single_branch_model
from DeepCCA.losses import cca_loss
from DeepCCA.metrics import Metrics

import librosa
import librosa.display 

#from keras import backend as K
#K.set_image_dim_ordering('th')

#tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
filesystem_path = (os.path.abspath(os.path.join(os.path.dirname("__file__"), '../')) + '/spotify/')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess) 
MOODS = ['aggressive', 'angry', 'bittersweet', 'calm', 'depressing', 'dreamy', 'fun', 'gay', 'happy', 'heavy', 'intense', 'melancholy', 'playful', 'quiet', 'quirky', 'sad', 'sentimental', 'sleepy', 'soothing', 'sweet']

SUBMOODS = []
   

def build_settings(config):

   test_size = [0.1]
   twin = [ 1.024]
   thop = [0.256]
   channels =[16]
   epochs =  [5]
   
   settings = {'test_size':test_size[config[0]], 'twin':twin[config[1]],
    'thop':thop[config[2]],'channels':channels[config[3]],
     'epochs':epochs[config[4]]}
   
   return settings

def get_configs(config):

   learning_rate =  [0.01]
   reg_par = [1e-5]

   outdim_size = [20]
   neurons_lstm =  [20]
   neurons1 =  [ 64, 32]
   neurons2 =  [ 64, 32]
   neurons3 =  [ 32, 16]
   
   dropout =  [ 0.5]
   validation_split =  [0.1]
   
   activation_model1 =  ['softmax' ]
   activation_model2 = ['softmax' ]
   activation_model3 = ['softmax' ]
   
   dense_size = [20]
   activation_lstm = ['softmax' ]

   settings = {'learning_rate':learning_rate[config[0]], 'reg_par':reg_par[config[1]],
    'outdim_size':outdim_size[config[2]],'neurons_lstm':neurons_lstm[config[3]], 
    'neurons1':neurons1[config[4]], 'neurons2':neurons2[config[5]], 'neurons3':neurons3[config[6]],
     'dropout':dropout[config[7]], 'validation_split':validation_split[config[8]]
    , 'activation_model1':activation_model1[config[9]], 'activation_model2':activation_model2[config[10]],
     'activation_model3':activation_model3[config[11]], 'dense_size':dense_size[config[12]], 
     'activation_lstm': activation_lstm[config[13]]}
   
   return settings

def start_process(samples, network, random_moods = 0,  label = "" , epoch_input=3):
   results = pd.DataFrame()

   #datadirs = ["datasets/cuts/30s_cuts1", "datasets/cuts/30s_cuts2" , "datasets/cuts/30s_cuts3"]
   datadirs = ["datasets/cuts/30s_cuts2"]
   overall_results = {'gtg': [], 'mfcc': [], 'dataset_config':[], 'dataset': []}

   calculate_submoods(random_moods)

   for directory in range(len(datadirs)):
      overall_results = {'gtg': [], 'mfcc': [], 'dataset_config':[], 'dataset': []}
 
      dataset_settings = [[0, 0,0, 0, 0]]

      for setting in range(len(dataset_settings)):
         print(" ")
         print("Settings available: ", len(dataset_settings))
         print(" ")
         print("Directory No." + str(directory) +"  Settings No"  +str(setting))
         print(" ")
       
         dataset_config = build_settings(dataset_settings[setting])
         test_size = dataset_config['test_size'] 
         twin = dataset_config['twin']
         thop = dataset_config['thop']
         channels = dataset_config['channels']
         epochs =  dataset_config['epochs']

         divisions = 8
         """
         print("x_mfcc1[0].shape", x_mfcc1[0].shape)
         """

         #pickle_name = 'gamatones1D_2D_lyrics_labels_'+str(divisions)+'div_'+str(twin)+'_'+str(thop)+'_'+str(channels)+'_std_minmax.pickle'
         pickle_name = 'gamatones1D_2D_lyrics_labels_4div_1.024_0.256_24_std_minmax.pickle'
         #pickle_name = 'gamatones1D_2D_lyrics_labels_8div_1.024_0.256_16ch_std_minmax.pickle'
         #pickle_name = 'gamatones1D_2D_lyrics_labels_4div_1.024_0.256_24_std_minmax.pickle'
         if not np.DataSource().exists(pickle_name):
            
            (labels1, labels_gtg, x_gtg, bow_infered, lyrics_infered, x_gtg_2d, x_gtgstd, x_gtgminmax) =  extract_data(datadirs[directory], samples, divisions= divisions,  twin=twin, thop=thop, channels=channels) 
            dataset = [x_gtg, x_gtg_2d, lyrics_infered, labels1, x_gtgstd, x_gtgminmax ]
            with open(pickle_name, 'wb') as output:
         
               pickle.dump(dataset, output)
         
         else:
            with open(pickle_name, 'rb') as data:
               dataset = pickle.load(data)


               

         x_gtg = dataset[0]
         x_gtg_2d = dataset[1]
         lyrics_infered = dataset[2]
         labels1 = dataset[3] 
         x_gtgstd = dataset[4]
         x_gtgminmax = dataset[5]
         
         try:
            if network == "CNN":
               values = np.expand_dims(x_gtgminmax, axis=3)
               
            else:
               values = x_gtg

            print("shape", values.shape)  
            print("GTG Start")

            results = train_deepcca(values, labels1, lyrics_infered, labels1, network, epochs= epoch_input,  batch_size=1500, test_size = test_size, label=label)

         except Exception as e:
            ipdb.set_trace()
         else:
            pass
         finally:
            pass
         
         
         #results = train_deepcca(x_gtg, labels1, bow_infered.squeeze(), labels1, network, epochs,  batch_size=1500, test_size = test_size, label=label)
         overall_results['gtg'].append(results)
         print("GTG Finish")

         print(" ")
         print(" ")



         print(overall_results)

         overall_results['dataset_config'].append([dataset_config])
         overall_results['dataset'].append([datadirs[directory]])

         output_file_name = datadirs[directory]+"_" +str(label)+"_"+str(setting)+"_"+str(divisions)+"_"+str(epochs) + "_"+str(twin)+ "_"+str(thop)+ "_"+str(channels)+".json"
         with open(output_file_name, 'w') as outfile:
            json.dump(overall_results, outfile)

   

def train_deepcca(X1, y1, X2, y2, network = 0, epochs=20, batch_size=20, test_size = 0.3, label="" ):

   y1 = to_categorical(y1, 20) # One-hot encode the labels
   
   num_classes = np.unique(y1).shape[0] # there are 10 image classes
   #y1_valid = to_categorical(y1_test, num_classes) # One-hot encode the labels
   #y1_test = to_categorical(y1_test, num_classes) # One-hot encode the labels

   X1_train, X1_test, y1_train, y1_test, X2_train, X2_test, y2_train, y2_test = train_test_split(X1, y1, X2, y2, test_size=test_size)
   X1_test, X1_valid, y1_test, y1_valid, X2_test, X2_valid, y2_test, y2_valid = train_test_split(X1_test, y1_test, X2_test, y2_test, test_size=0.5)



   data1 = []
   data2 = []


   num_classes = 20

   data1.append([X1_train, y1_train])
   data1.append([X1_valid, y1_valid])
   data1.append([X1_test, y1_test])

   data2.append([X2_train, y2_train])
   data2.append([X2_valid, y2_valid])
   data2.append([X2_test, y2_test])

   # size of the input for view 1 and view 2
   #input_shape1 = X1_train.shape[2]
   #input_shape2 = X2_train.shape[1]

   
   # the parameters for training the network
   epoch_num = epochs
   batch_size = batch_size
   
   # the regularization parameter of the network
   # seems necessary to avoid the gradient exploding especially when non-saturating activations are used
   

   # specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
   # if one option does not work for a network or dataset, try the other one
   use_all_singular_values = True

   # if a linear CCA should get applied on the learned features extracted from the networks
   # it does not affect the performance on noisy MNIST significantly
   apply_linear_cca = True
   best_accuracy = 0
   if len(X1.shape) > 3:
      num_train, height, width, depth = X1.shape
   


   input_size1 =  X1_train.shape[-1]

   input_size2 =  X2_train.shape[1]


   input_shape1_1 = X1_train.shape[1]
   input_shape1_2 = X1_train.shape[-1]
   
   """
   learning_rate =  [0.01]
   reg_par = [1e-5]

   outdim_size = [20]
   neurons_lstm =  [20]
   neurons1 =  [ 64, 32]
   neurons2 =  [ 64, 32]
   neurons3 =  [ 32, 16]
   
   dropout =  [ 0.5]
   validation_split =  [0.1]
   
   activation_model1 =  ['softmax' ]
   activation_model2 = ['softmax' ]
   activation_model3 = ['softmax' ]
   
   dense_size = [20]
   activation_lstm = ['softmax' ]

   """

   best_results = {'config':[], 'best_accuracy': []} 

   configs = [[0,0, 0,0 ,0,0,0 ,0,0, 0,0,0, 0,0],
              #[0,0, 0,0 ,0,1,1 ,0,0, 0,0,0, 0,0],
              #[0,0, 2,0 ,1,1,1 ,1,0, 2,2,2, 1,2],
              ]

   test_config = {}

   for config in range(len(configs)):

      test_config = get_configs(configs[config])
      print(" ")
      print("Settings available: ", len(configs))
      print(" ")
      print("Next Network config No: "  +str(config)+"/"+ str(len(configs)))
      print(" ")
      try:
         model = None


         if network == 'BLSTM':
            model = build_BLSTM_model(input_size1, input_size2, test_config['dense_size'],
                          test_config['learning_rate'], test_config['reg_par'], test_config['outdim_size'], 
                          test_config['activation_lstm'], use_all_singular_values, test_config['neurons_lstm'], test_config['dropout'],
                          test_config['activation_model1'], test_config['activation_model2'], test_config['activation_model3'],
                          test_config['neurons1'], test_config['neurons2'], test_config['neurons3'])
         elif network == 'DRNN':
            model = build_DRNN_model(input_size1, input_size2, test_config['dense_size'],
                          test_config['learning_rate'], test_config['reg_par'], test_config['outdim_size'], 
                          test_config['activation_lstm'], use_all_singular_values, test_config['neurons_lstm'], test_config['dropout'],
                          test_config['activation_model1'], test_config['activation_model2'], test_config['activation_model3'],
                          test_config['neurons1'], test_config['neurons2'], test_config['neurons3'])
         elif network == 'GRU':
            model = build_GRU_model(input_size1, input_size2, test_config['dense_size'],
                          test_config['learning_rate'], test_config['reg_par'], test_config['outdim_size'], 
                          test_config['activation_lstm'], use_all_singular_values, test_config['neurons_lstm'], test_config['dropout'],
                          test_config['activation_model1'], test_config['activation_model2'], test_config['activation_model3'],
                          test_config['neurons1'], test_config['neurons2'], test_config['neurons3'])

         elif network == 'CNN':
            model = create_CNN_RNN(height, width, depth, input_size2, test_config['dense_size'],
                          test_config['learning_rate'], test_config['reg_par'], test_config['outdim_size'], 
                          test_config['activation_lstm'], use_all_singular_values, test_config['neurons_lstm'], test_config['dropout'],
                          test_config['activation_model1'], test_config['activation_model2'], test_config['activation_model3'],
                          test_config['neurons1'], test_config['neurons2'], test_config['neurons3'])

            

         plot_model(model, to_file=network+'_moldel.png')

         model, hist = train_single_branch_model(model, data1, data2, epoch_num, batch_size, label)

         model.summary()
         
         

         new_data = test_dcca(model, data1, None, 20, False)


         
         # Training and testing of SVM with linear kernel on the view 1 with new features
         
         [test_acc, valid_acc] = single_svm_classify(new_data, C=0.01, axis=0, dual=False)
         if test_acc >= best_accuracy:
            best_accuracy = test_acc
            best_results['config'].append(test_config)
            best_results['best_accuracy'].append([best_accuracy])
         print("Accuracy on view 1 (validation data) is:", valid_acc * 100.0)
         print("Accuracy on view 1 (test data) is:", test_acc*100.0)
         plot_results(hist)
      
      except Exception as e:
         print(e)
         best_results['config'].append(test_config)
         




   return best_results

   


def ranking_average_prescision(y, y_pred_probs):
   cntr, uc, u0c, dc, jm, pcd, fpc = fuzz.cluster.cmeans(X.T, 19, 2, error=0.005, maxiter=1000, init=None)
   u0p, up, dp, jm, pp, fpc =  fuzz.cluster.cmeans_predict(up, cntr, 4, error=0.0005, maxiter=1000)
   ranking_score = label_ranking_average_precision_score(y, u0p.T)


   
def get_lyrics_dict():

   files = [os.path.join(filesystem_path,fn) for fn in os.listdir(filesystem_path) if fn.endswith("integral_synced_SongsMoodsFile.csv")]
   audios_data = pd.read_csv(files[0], delimiter=',', encoding="utf-8",quotechar='"')
   df = pd.DataFrame.from_dict(audios_data, orient='columns')
   return df


def get_lyrics(video_id):
   query_result = lyrycs_df.query('youtube_video_id ==  @video_id_no_div')
   bow = query_result['bow'].values[0]
   lyric = query_result['lyric'].values[0]

def extract_data(audioset_path, samples, divisions = 1, twin=0.5, thop=0.250, channels=1):

   model_directory = os.path.dirname(os.path.realpath(__file__))+ "/apnews_dbow/"

   saved_files = [os.path.join(audioset_path,fn) for fn in os.listdir(audioset_path) if fn.endswith('.wav')]



   labels_1 = []
   labels_2 = []
   labels_gtg = []
   lyrics_infered = []
   bow_infered = []
   x_mfcc_1 = []
   x_gtg_2d = []
   gtg_minmax = []
   gtg_std = []
   x_gtg = []
   counter = 0
   errors = {"error":[], "counter":[]}
   
   fmin = 20

   doc2vec = d2v.doc2vec_model(model_directory, model_name="doc2vec.bin", lyrics_filename="integral_synced_SongsMoodsFile.csv")
   
   doc2vec_model = doc2vec.load_model()
   samplerate = 161

   lyrycs_df = get_lyrics_dict()
   scaler_gtg = preprocessing.StandardScaler()
   minmax_scaler = preprocessing.MinMaxScaler()

   for audiopath in saved_files[:samples]:
      try:
        # if audiop5000.split("/")[-1].split("_")[0] in ['aggressive', 'happy', 'sad', 'sweet']:
         #(rate,sig) = wav.read(audiopath)

         sig, rate = librosa.load(audiopath, res_type='scipy')
         
         duration = len(sig) / rate
         if duration >= 30:
            #signal, rate, freq = 0.5, numcep=1, winlen = 1, winstep = 0.5 #
            audio_name = audiopath.split("/")[-1]
               
            if len(SUBMOODS) > 0:
               MOODS = SUBMOODS

            mood = audio_name.split("_")[0]

            if mood in MOODS:
               video_id = audio_name[-15:-4]
               query_result = lyrycs_df.query('youtube_video_id ==  @video_id')
               lyc = query_result['lyric'].values[0]
               bow = query_result['bow'].values[0]
               # Tokenize, its not working. We already have cleaned lyrics.
               # client =  corenlp.CoreNLPClient(annotators="tokenize ssplit sentiment".split())
               #tokenzed_lyc = client.annotate(lyc)
               #tokenzed_bow = client.annotate(bow)
               bow_inf = doc2vec_model.infer_vector(bow)
               lyr_inf = doc2vec_model.infer_vector(lyc)

               sig = np.array(sig).astype(np.float32)
               sig[sig == 0] = 0.000001
               
               fifth = int(len(sig)/divisions)

               for slice_sig in range(divisions):
                  
                  sig_slice = sig[fifth*slice_sig:fifth*(slice_sig+1)]
                  
                  #mfcc_1 = get_mfcc(sig_slice, rate, channels, twin, thop)
                               
                  #mfcc_1_scaled = scaler_mfcc.fit_transform(mfcc_1)


                  #mfcc_1 = np.array(mfcc_1.reshape(1, mfcc_1_scaled.size)).astype(np.float32)
                  #mfcc_1 = preprocessing.normalize(np.array(mfcc_1).astype(np.float32))
                  #print("mfcc_1.shape:", mfcc_1.shape)
                  
                  gtg = get_gtg(rate, sig_slice, twin, thop, channels, fmin)
                  gtg_std_scaled = scaler_gtg.fit_transform(gtg) 
                  gtg_minmax_scaled = minmax_scaler.fit_transform(gtg)

                  

                  #gtg = preprocessing.normalize(np.array(gtg).astype(np.float32))
                  #print("gtg.shape: ", gtg.shape)
                  
                  labels_gtg.append(int(MOODS.index(mood)))
                  labels_1.append(int(MOODS.index(mood)))
                  
                  
                  bow_infered.append(bow_inf)
                  lyrics_infered.append(lyr_inf)
                  
                  x_gtg.append(np.array(gtg.reshape(1, gtg.size)).astype(np.float32))

                  #x_mfcc_1.append(np.array(mfcc_1.reshape(1, mfcc_1.size)).astype(np.float32))
                  x_gtg_2d.append(np.array(gtg).astype(np.float32))

                  gtg_std.append(np.array(gtg_std_scaled).astype(np.float32))
                  gtg_minmax.append(np.array(gtg_minmax_scaled).astype(np.float32))

               """               
               mfcc_gtg = get_mfcc(gtg, rate, 0.3, 26, 1, .5)
               print("mfcc_gtg.shape"
               librosa, mfcc_gtg.shape)
               mfcc_gtg = preprocessing.normalize(np.array(mfcc_gtg).astype(np.float32) )
               x_mixed.append(np.array(mfcc_gtg.reshape(1, mfcc_gtg.size)).astype(np.float32))
               """   

            
      except Exception as e:
         errors['error'].append(e)
         errors['counter'].append(e)
         ipdb.set_trace()
         print(e)
      print(counter)
      counter+=1   
      
   
   labels_np1 = np.array(labels_1).astype(dtype=np.uint8)
   
   #labels_np2 = np.array(labels_2).astype(dtype=np.uint8)
   bow_infered = np.array(bow_infered).astype(np.float32)
   lyrics_infered = np.array(lyrics_infered).astype(np.float32)
   #x_mfcc_np1 = np.array(x_mfcc_1).astype(np.float32)


   #x_mfcc_np2 = np.array(x_mfcc_2).astype(np.float32)
   #labels_onehot1 = (np.arange(19) == labels_np1[:, None]).astype(np.float32)

   #If already finished training a model (=no more updates, only querying, reduce memory usage
   doc2vec_model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

   
   x_gtg_np = np.array(x_gtg).astype(np.float32)
   x_gtg_2d_np = np.array(x_gtg_2d).astype(np.float32)
   x_gtg_std = np.array(gtg_std).astype(np.float32)
   x_gtg_minmax = np.array(gtg_minmax).astype(np.float32)
   
   #x_mixed = np.array(x_mixed).astype(np.float32)
   labels_onehotgtg = (np.arange(20) == labels_np1[:, None]).astype(np.float32)
   return labels_np1, labels_onehotgtg, x_gtg_np, bow_infered, lyrics_infered, x_gtg_2d_np, x_gtg_std, x_gtg_minmax
   


def get_gtg(rate, signal, twin, thop, channels, fmin):
   gtg = gtgram.gtgram(signal, rate, twin, thop, channels, fmin)
   Z = np.flipud(20 * np.log10(gtg))
   return Z


def get_mfcc(signal, rate, numcep=1, twin = 1, thop = 0.5 ):
   mfcc = librosa.feature.mfcc(signal, sr=rate, n_mfcc=numcep).T
   #mfcc(signal, samplerate=rate, numcep=numcep, winlen=twin,winstep=thop )
   return mfcc


def split_data(labels, X, size):
   X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=size )
   return X_train, X_test, y_train, y_test


def plot_results(hist):
   xc = range(len(hist.history['loss']))

   train_loss = hist.history['loss']
   val_loss = hist.history['val_loss']
   
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

   try:
      train_acc = hist.history['acc']
      val_acc = hist.history['val_acc']


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
      
   except Exception as e:
      print("Error: ", e)

   plt.show()
   


def constant_loss(y_true, y_pred):
    return y_pred

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


def fuzzy_svms_classify(data ):
   cntr, uc, u0c, dc, jm, pc, fpc = fuzz.cluster.cmeans(mfcc2_X_trainT, 19, 2, error=0.005, maxiter=1000, init=None)
   u0p, up, dp, jm, pp, fpc =  fuzz.cluster.cmeans_predict(mfcc2_X_testT, cntr, 4, error=0.005, maxiter=1000)

def single_svm_classify(data, C, axis = 0, dual=False):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """
    train_data, train_label = data[0]
    valid_data, valid_label = data[1]
    test_data, test_label = data[2]

    print('training SVM...')
    clf = svm.LinearSVC(C=C, dual=False)
    
    clf.fit(train_data, np.argmax(train_label, axis=1))

    preds = clf.predict(test_data)
    test_acc = accuracy_score(np.argmax(test_label, axis=1), preds)

    preds = clf.predict(valid_data)
    valid_acc = accuracy_score(np.argmax(valid_label, axis=1), preds)
 
    return [test_acc, valid_acc]


def svm_classify(data, C, axis = 0, dual=False):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """
    train_data, train2_data, train_label = data[0]
    valid_data, valid2_data, valid_label = data[1]
    test_data, test2_data, test_label = data[2]




    print('training SVM...')
    clf = svm.LinearSVC(C=C, dual=False)
    
    if axis == 0:
       clf.fit(train_data, np.argmax(train_label, axis=1))

       preds = clf.predict(test_data)
       test_acc = accuracy_score(np.argmax(test_label, axis=1), preds)

       preds = clf.predict(valid_data)
       valid_acc = accuracy_score(np.argmax(valid_label, axis=1), preds)
    else:
       clf.fit(train2_data, np.argmax(train_label, axis=1))

       preds = clf.predict(test2_data)
       test_acc = accuracy_score(np.argmax(test_label, axis=1), preds)

       preds = clf.predict(valid2_data)
       valid_acc = accuracy_score(valid_label, preds)




    return [test_acc, valid_acc]

def calculate_submoods(random_moods):

   if random_moods != 0 and random_moods <= 20:

      secure_random = random.SystemRandom()

      while len(SUBMOODS) < random_moods:
         rand_select = secure_random.choice(MOODS)
         if rand_select not in SUBMOODS:
            SUBMOODS.append(rand_select)
            MOODS.remove(rand_select)

      print("")
      print("")
      print("Selected moods: ", SUBMOODS)
      print("")
      print("")

def main(argv):
   
   samples = -1
   label = ""
   network = ""
   epochs = 1
   
   try:
      opts, args = getopt.getopt(argv,"hi:se:l:n:m:e:",["ifile=", "samples="])
   except getopt.GetoptError:
      print( 'keras_build.py -s <samples=-1>  -l <label=""> -n <network="GRU/DRNN/BLSTM"> -m <moods="int">')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print( 'keras_build.py -s <samples=-1>  -l <label=""> -n <network="GRU/DRNN/BLSTM">')
         sys.exit()
      elif opt in ("-s","--samples"):
         samples = int(arg.strip())
      elif opt in ("-l","--label"):
         label = str(arg.strip())
      elif opt in ("-n", "--network"):
         network = str(arg.strip())
      elif opt in ("-m", "--moods"):
         random_moods = int(arg.strip())
      elif opt in ("-e", "--epochs"):
         epochs = int(arg.strip())

         
   start_process(samples, network, random_moods=20, label=label, epoch_input=epochs)

if __name__ == "__main__":
   main(sys.argv[1:])
   

