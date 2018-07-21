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
	channels = 2
	fmin = 20


	formatter = plot.ERBFormatter(fmin, rate*3/4, unit='Hz', places=0)
	axes.yaxis.set_major_formatter(formatter)

	# Figure out time axis scaling
	duration = len(sig) / rate

	# Calculate 1:1 aspect ratio
	aspect_ratio = duration/scipy.constants.golden

	gtg = gtgram.gtgram(sig, rate, twin, thop, channels, fmin)
	Z = np.flipud(20 * np.log10(gtg))


	ipdb.set_trace()

	img = axes.imshow(Z, extent=[0, duration, 1, 0], aspect=aspect_ratio)


	matplotlib.pyplot.show()
	
	
	ipdb.set_trace()
	
	
	
	#gammatone.gtgram.gtgram(wave, fs, window_time, hop_time, channels, f_min)
	"""
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
	ipdb.set_trace()
	"""
	
	
	
	#ssc = ssc(sig,samplerate=rate)

	#print(logfbank_feat[1:3,:])

	#filters.centre_freqs(fs, num_freqs, cutoff)
	centre_freqs = filters.centre_freqs(rate, sig.shape[0], 100)

	axes.set_title("centre_freqs"+ str(centre_freqs.shape)+" " + os.path.basename(new_file_name_path))
	axes.set_xlabel("Time (s)")
	axes.set_ylabel("Frequency")
	print("centre_freqs.shape", centre_freqs.shape)
	matplotlib.pyplot.plot(centre_freqs)
	matplotlib.pyplot.show()
	ipdb.set_trace()


	erb_filters = filters.make_erb_filters(rate, centre_freqs, width=1.0)
	axes.set_title("erb_filters"+ str(erb_filters.shape)+" " + os.path.basename(new_file_name_path))
	axes.set_xlabel("Time (s)")
	axes.set_ylabel("Frequency")
	print("erb_filters.shape", erb_filters.shape)
	matplotlib.pyplot.plot(erb_filters)
	matplotlib.pyplot.show()
	ipdb.set_trace()





	
	fig = matplotlib.pyplot.figure()
	axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
	axes.set_xlabel("Time (s)")
	axes.set_ylabel("Frequency")
	
	mfcc_feat = mfcc(sig,samplerate=rate)#(2992, 13)
	print("mfcc_feat.shape:", mfcc_feat.shape )
	
	matplotlib.pyplot.plot(mfcc_feat)
	axes.set_title("mfcc_feat " + str(mfcc_feat.shape)+" "+   os.path.basename(new_file_name_path))
	matplotlib.pyplot.show()
	ipdb.set_trace()


	mfcc_one_line = mfcc_feat.reshape(38896, 1)
	print("mfcc_one_line.T.shape", mfcc_one_line.T.shape)



	fig = matplotlib.pyplot.figure()
	axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
	
	logfbank_feat = logfbank(sig, samplerate=rate)
	
	axes.set_title("logfbank_feat"+ str(logfbank_feat.shape)+" " + os.path.basename(new_file_name_path))
	axes.set_xlabel("Time (s)")
	axes.set_ylabel("Frequency")
	print("logfbank_feat.shape", logfbank_feat.shape)
	matplotlib.pyplot.plot(logfbank_feat)
	matplotlib.pyplot.show()
	ipdb.set_trace()

	
	fig = matplotlib.pyplot.figure()
	axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
	
	d_mfcc_feat = delta(mfcc_feat, 2)
	print("d_mfcc_feat.shape", d_mfcc_feat.shape)
	axes.set_title("d_mfcc_feat"+ str(d_mfcc_feat.shape) +" "+ os.path.basename(new_file_name_path))
	axes.set_xlabel("Time (s)")
	axes.set_ylabel("Frequency")
	print("d_mfcc_feat.shape", d_mfcc_feat.shape)
	matplotlib.pyplot.plot(d_mfcc_feat)

	matplotlib.pyplot.show()
	ipdb.set_trace()
	





	"""
	Renders the given ``duration`` of audio from the audio file at ``path``
	using the gammatone spectrogram function ``function``.
	"""
 

	


	fig = matplotlib.pyplot.figure()
	axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
	
	erb_filterbank = filters.erb_filterbank(sig, )
	
	axes.set_title("erb_filterbank"+ str(erb_filterbank.shape)+" " + os.path.basename(new_file_name_path))
	axes.set_xlabel("Time (s)")
	axes.set_ylabel("Frequency")
	print("erb_filterbank.shape", erb_filterbank.shape)
	matplotlib.pyplot.plot(erb_filterbank)
	matplotlib.pyplot.show()
	ipdb.set_trace()





