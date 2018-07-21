import pandas as pd
import os
import ipdb
from sentences import Sentences 
from gensim.models.doc2vec import Doc2Vec
import nltk


#file_location="/media/frisco/FAT/cross-modal/synced_audios/"
file_location="/home/frisco/Documents/Nube/Projects/python/cross-modal/spotify/"
file_name="Synced_SongsMoodsFile.csv"
files = [os.path.join(file_location,fn) for fn in os.listdir(file_location) if fn.endswith('Synced_SongsMoodsFile.csv')]

for file in files:
	songs_lyrics_ids_df = pd.read_csv(file, delimiter=',', encoding="utf-8")
	
	
	songs_lyrics_ids_df.head()

	group = ['lyric','mood' ]
	lyrics_by_song = songs_lyrics_ids_df.sort_values(group)\
       .groupby(group).lyric\
       .apply(' '.join)\
       #.reset_index(name='lyric')
	lyrics_by_song.head(1)
	file_name = file

filename=file_name
nltk.download('wordnet')
sentences = Sentences(filename=filename, column="lyric")
df_train = pd.read_csv(filename,
        index_col=[0],
        usecols=[0,3],
        header=0,
        names=["mood", "lyric"])
ipdb.set_trace()
model = Doc2Vec(alpha=0.025,min_alpha=0.025,workers=15, min_count=2,window=10,size=300,iter=20,sample=0.001,negative=5)




model.build_vocab(sentences)
epochs = 10

for epoch in range(epochs):
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay


    model.save('lyrics.doc2vec')

	

#model = Doc2Vec.load('lyrics.doc2vec')
