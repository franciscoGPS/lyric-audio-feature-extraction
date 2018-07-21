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
from gammatone import plot as gammaplot



AUDIOS_PATH = "/home/frisco/Documents/Nube/Projects/python/cross-modal/rnn-cross-modal/datasets/"

def get_spectrogram(audio_file):
	sample_rate, X = wav.read(audio_file)
	print (sample_rate, X.shape )
	a,b,c,d = plt.specgram(X, Fs=sample_rate, xextent=(0,30))
	return a,b,c,d,plt








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
	aa,bb,cc,dd, plt = get_spectrogram(new_file_name_path)
	matplotlib.pyplot.show()

	fd = 2048
	fs = 1024

	f_size = fd * fs

	(rate,sig) = wav.read(new_file_name_path)
	
	
	#fbank_feat = fbank(sig,samplerate=rate)
	
	
	
	#gammatone.gtgram.gtgram(wave, fs, window_time, hop_time, channels, f_min)
	gtgram_function = gtgram.gtgram(sig, rate, .250, .125, 1, 20)
	fig = matplotlib.pyplot.figure()
	axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])	
	axes.set_xlabel("Time (s)")
	axes.set_ylabel("Frequency")
	print("gtgram_function.shape", gtgram_function.shape)	
	print("gtgram_function.T", gtgram_function.T.shape)
	matplotlib.pyplot.plot(gtgram_function.T)
	axes.set_title("gtgram_function.T 1ch. " + os.path.basename(new_file_name_path))
	matplotlib.pyplot.show()


	fig = matplotlib.pyplot.figure()
	axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
	axes.set_xlabel("Time (s)")
	axes.set_ylabel("Frequency")
	axes.set_title("gtgram_function. 1ch. " + os.path.basename(new_file_name_path))
	print("gtgram_function.shape", gtgram_function.shape)
	matplotlib.pyplot.plot(gtgram_function)
	matplotlib.pyplot.show()

	
	
	
	
	#ssc = ssc(sig,samplerate=rate)

	#print(logfbank_feat[1:3,:])

	#gammatone.filters.centre_freqs(fs, num_freqs, cutoff)
	#centre_freqs = filters.#centre_freqs(rate, sig.shape[0], 100)

	
	fig = matplotlib.pyplot.figure()
	axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
	axes.set_xlabel("Time (s)")
	axes.set_ylabel("Frequency")
	
	mfcc_feat = mfcc(sig,samplerate=rate)#(2992, 13)
	print("mfcc_feat.shape:", mfcc_feat.shape )
	
	matplotlib.pyplot.plot(mfcc_feat)
	axes.set_title("mfcc_feat " + os.path.basename(new_file_name_path))
	matplotlib.pyplot.show()


	mfcc_one_line = mfcc_feat.reshape(38896, 1)
	print("mfcc_one_line.T.shape", mfcc_one_line.T.shape)

	matplotlib.pyplot.plot(mfcc_one_line.T)
	fig = matplotlib.pyplot.figure()
	axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
	axes.set_title("mfcc_feat " + os.path.basename(new_file_name_path))
	axes.set_xlabel("Time (s)")
	axes.set_ylabel("Frequency")
	
	axes.set_title(" " + os.path.basename(new_file_name_path))
	axes.set_xlabel("Time (s)")
	axes.set_ylabel("Frequency")
	matplotlib.pyplot.show()

	fig = matplotlib.pyplot.figure()
	axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
	
	logfbank_feat = logfbank(sig, samplerate=rate)
	matplotlib.pyplot.plot(logfbank_feat)
	axes.set_title(" " + os.path.basename(new_file_name_path))
	axes.set_xlabel("Time (s)")
	axes.set_ylabel("Frequency")
	matplotlib.pyplot.show()

	
	fig = matplotlib.pyplot.figure()
	axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
	
	d_mfcc_feat = delta(mfcc_feat, 2)
	print("d_mfcc_feat.shape", d_mfcc_feat.shape)
	matplotlib.pyplot.plot(d_mfcc_feat)
	axes.set_title(" " + os.path.basename(new_file_name_path))
	axes.set_xlabel("Time (s)")
	axes.set_ylabel("Frequency")
	matplotlib.pyplot.show()
	





	"""
	Renders the given ``duration`` of audio from the audio file at ``path``
	using the gammatone spectrogram function ``function``.
	"""
 
	# Average the stereo signal
	duration = False
	if duration:
		nframes = duration * rate
		sig = sig[0:nframes, :]

	#signal = sig.mean()
 
	# Default gammatone-based spectrogram parameters	
	twin = 0.250
	thop = twin/2
	channels = 16
	fmin = 20
 

	
	



	gammaplot.gtgram_plot(
		gtgram.gtgram,
		axes,
		sig,
		rate,
		twin, thop, channels, fmin)
	
	ipdb.set_trace()
	axes.set_title(" " + os.path.basename(new_file_name_path))
	axes.set_xlabel("Time (s)")
	axes.set_ylabel("Frequency")
	matplotlib.pyplot.show()
	
	





