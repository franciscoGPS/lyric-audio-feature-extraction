import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
import os
import sys
import os.path as path
filesystem_path = (path.abspath(path.join(path.dirname("__file__"), '../')) + '/spotify/')
sys.path.append(filesystem_path)
import ipdb
import dataset_verification as dataset
from pydub import AudioSegment
from pydub.playback import play
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from gammatone import filters
from gammatone import gtgram
from gammatone import plot

#https://github.com/detly/gammatone/blob/master/gammatone/plot.py
#Utilities for analysing sound using perceptual models of human hearing.

AUDIOS_PATH = "/home/frisco/Documents/Nube/Projects/python/cross-modal/rnn-cross-modal/datasets/"

def get_spectrogram(audio_file):
	sample_rate, X = wav.read(audio_file)
	print (sample_rate, X.shape )
	a,b,c,d = plt.specgram(X, Fs=sample_rate, xextent=(0,30))
	return a,b,c,d,plt





def start_process():
	files = [os.path.join(AUDIOS_PATH+"complete/",fn) for fn in os.listdir(AUDIOS_PATH+"complete/") if fn.endswith('.mp3')]
	for file in files:
		filename = file.split("/")[-1].split(".")[:-1][0]

		# Set up the plot	
		fig = matplotlib.pyplot.figure()
		axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
			
		new_file_name_path = AUDIOS_PATH+"cuts/30s_cuts/"+filename+".wav"
		dataset.cut_30s_from_file(filename, file, AUDIOS_PATH+"cuts/")
		#track_30s = AudioSegment.from_wav(new_file_name_path)
		#play(track_30s)
		#aa,bb,cc,dd, plt = get_spectrogram(new_file_name_path)
		#matplotlib.pyplot.show()
		#ipdb.set_trace()

		(rate,sig) = wav.read(new_file_name_path)
		
		
		#fbank_feat = fbank(sig,samplerate=rate)

			# Average the stereo signal
		duration = False
		if duration:
			nframes = duration * rate
			sig = sig[0:nframes, :]

		#signal = sig.mean()
	 
		# Default gammatone-based spectrogram parameters	
		twin = 0.250
		thop = twin/2
		channels = 8
		fmin = 20


		formatter = plot.ERBFormatter(fmin, rate/2, unit='Hz', places=0)
		axes.yaxis.set_major_formatter(formatter)

		# Figure out time axis scaling
		duration = len(sig) / rate

		# Calculate 1:1 aspect ratio
		aspect_ratio = duration/scipy.constants.golden

		gtg = gtgram.gtgram(sig, rate, twin, thop, channels, fmin)

		Z = np.flipud(20 * np.log10(gtg))
		z_reshaped = Z.reshape(Z.size, 1)
		img = axes.imshow(Z, extent=[0, duration, 1, 0], aspect=aspect_ratio)



		matplotlib.pyplot.show()
		
		
		ipdb.set_trace()




		fig = matplotlib.pyplot.figure()
		axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
		# Default gammatone-based spectrogram parameters	
		twin = 0.250
		thop = twin/2
		channels = 16
		fmin = 20


		formatter = plot.ERBFormatter(fmin, rate*3/4, unit='Hz', places=0)
		axes.yaxis.set_major_formatter(formatter)

		# Figure out time axis scaling
		duration = len(sig) / rate

		# Calculate 1:1 aspect ratio
		aspect_ratio = duration/scipy.constants.golden

		gtg = gtgram.gtgram(sig, rate, twin, thop, channels, fmin)

		Z = np.flipud(20 * np.log10(gtg))


		img = axes.imshow(Z, extent=[0, duration, 1, 0], aspect=aspect_ratio)


		matplotlib.pyplot.show()
		
		
		ipdb.set_trace()

		mfcc_feat = mfcc(sig,samplerate=rate, winlen=twin,winstep=thop)