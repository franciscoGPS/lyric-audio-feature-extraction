import scipy
import matplotlib.pyplot as plt

def get_spectrogram(audio_file):
	sample_rate, X = scipy.io.wavfile.read(audio_file)
	print (sample_rate, X.shape )
	plt.specgram(X, Fs=sample_rate, xextent=(0,30))
	return plt