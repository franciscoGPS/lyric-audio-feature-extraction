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
from python_speech_features import fbank

from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from gammatone import filters
from gammatone import gtgram
from gammatone import plot as gammaplot
import librosa 
import librosa.display

AUDIOS_PATH = "/home/frisco/Documents/Nube/Projects/python/cross-modal/rnn-cross-modal/datasets/"

def get_spectrogram(audio_file):
	sample_rate, X = wav.read(audio_file)
	print (sample_rate, X.shape )
	a,b,c,d = plt.specgram(X, Fs=sample_rate, xextent=(0,30))
	return a,b,c,d,plt








files = [os.path.join(AUDIOS_PATH+"complete/",fn) for fn in os.listdir(AUDIOS_PATH+"complete/") if fn.endswith('.mp3')]



for file in files:
	filename = file.split("/")[-1].split(".")[:-1][0]
		
	new_file_name_path = AUDIOS_PATH+"cuts/30s_cuts/"+filename+".wav"
	#dataset.cut_30s_from_file(filename, file, AUDIOS_PATH+"cuts/")
	#track_30s = AudioSegment.from_wav(new_file_name_path)
	#play(track_30s)
	#aa,bb,cc,dd, plt = get_spectrogram(new_file_name_path)
		

	fd = 2048
	fs = 1024

	f_size = fd * fs

	(rate,sig) = wav.read(new_file_name_path)
	x_brahms, sr_brahms = librosa.load(file, duration=30, offset= 30)

	mfcc_feat = mfcc(sig,samplerate=rate)#(2992, 13)
	
	ipdb.set_trace()

	#mfcc_one_line = mfcc_feat.reshape(38896, 1)
	fbank_feat = fbank(sig,samplerate=rate)
	logfbank_feat = logfbank(sig, samplerate=rate)
	d_mfcc_feat = delta(mfcc_feat, 2)
	#gammatone.gtgram.gtgram(wave, fs, window_time, hop_time, channels, f_min)
	
	gtgram_function = gtgram.gtgram(sig, rate, .250, .125, 1, 20)


	print("mfcc_feat.shape:", mfcc_feat.shape )
	print("mfcc_one_line.shape", mfcc_one_line.shape)
	print("logfbank_feat.shape", logfbank_feat.shape)
	print("d_mfcc_feat.shape", d_mfcc_feat.shape)
	print("gtgram_function.shape", gtgram_function.shape)	
	print("gtgram_function.shape.T", gtgram_function.T.shape)
	#ssc = ssc(sig,samplerate=rate)

	#print(logfbank_feat[1:3,:])

	#gammatone.filters.centre_freqs(fs, num_freqs, cutoff)
	#centre_freqs = filters.#centre_freqs(rate, sig.shape[0], 100)

	

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
	channels = 1
	fmin = 20
 
	# Set up the plot	
	fig = matplotlib.pyplot.figure()
	axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
	
	
	

	matplotlib.pyplot.plot(mfcc_feat)
	matplotlib.pyplot.plot(mfcc_one_line)
	matplotlib.pyplot.plot(logfbank_feat)
	matplotlib.pyplot.plot(d_mfcc_feat)
	gammatone = gammaplot.gtgram_plot(
		gtgram.gtgram,
		axes,
		sig,
		rate,
		twin, thop, channels, fmin)
	
	ipdb.set_trace()
	axes.set_title(os.path.basename(new_file_name_path))
	axes.set_xlabel("Time (s)")
	axes.set_ylabel("Frequency")
 	
	matplotlib.pyplot.show()
	





