import csv
from nltk.stem import WordNetLemmatizer
from gensim.models.doc2vec import TaggedDocument
word_net=WordNetLemmatizer()

class Sentences(object):
    
    def __init__(self, filename, column):
        self.filename = filename
        self.column = column
        
    @staticmethod
    def get_tokens(text):
        """Helper function for tokenizing data"""
        return [word_net.lemmatize(r.lower()) for r in text.split()]
 
    def __iter__(self):
        reader = csv.DictReader(open(self.filename, 'r' ))
        for row in reader:
            words = self.get_tokens(row[self.column])
            #tags = ['%s|%s' % (row['artist'], row['song_id'])]
            tags = ['%s' % (row['mood'])]
            yield TaggedDocument(words=words, tags=tags)