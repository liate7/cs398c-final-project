#Ali Muhammed
from collections import defaultdict
from gensim import corpora
from nltk.corpus import stopwords
from gensim import models
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer




def indexprompt():
    if ask("Generate the index? "):
        return not (os.path.exists(INDEX) and not ask("Index exists.... "))
    else:
        return not os.path.exists(INDEX) and ask("Index does not exist...make?")



def get_index():
    if indexprompt():
        return index()
def index():
     termtodocumentMatrix()
    
    

  
def termtodocumentMatrix():
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features= 1000, max_df = 0.5, smooth_idf=True)
    X = vectorizer.fit_transform(news_df['clean_doc'])
    X.shape #sees the shape of matrix

def preprocess_data(doc_set):
    #tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    # create English stop words list
    en_stop = set(stopwords.words('english'))
    
    # list for tokenized documents in loop
    texts = []
    # loop through document list
    for i in doc_set:
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        # add tokens to list
        texts.append(stemmed_tokens)
    return texts
def preProcess(sentence):
#another proprocessing function
    
    sentence=sentence.lower()        
    sentence=cleanPunct(sentence)    
    sentence=word_tokenize(sentence)
    stop_words = stopwords.words('english')
    return sentence

def main():
    #pickle dump
    index = get_index()
    file = open('Desktop', 'Ali') #this can be whatever
    pickle.dump(index, file)
   
    
    
    
   


