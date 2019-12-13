import numpy as np
from nltk import corpus
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec

# This is CJ Monteclaro's code,
# adapted to work with everything else by Andrew Patterson

def lemmasAndCleaned(files, text):    
    lemmatizer = WordNetLemmatizer()
    stoplist = set(stopwords.words('english'))
    cleanedText = []
    text = text.lower()
    text = cleanPunct(text)
    text = word_tokenize(text)
    for word in text:
        lemma = lemmatizer.lemmatize(word)
        if(word not in stoplist):
            cleanedText.append(lemma)
    files.append(cleanedText)
    return files
        
def cleanPunct(strSent):    
    punct = '!()-[]{};:"\,<>./?@|#$%^&*_~'
    cleanSent=""    
    for char in strSent:
        if char not in punct:
            cleanSent+=char            
    return cleanSent

def query(query, index):
    doc2vec = index["vectors"]
    # getting the query and making sure it should be executed
    frequency = []
    # cleaning the query and putting each word into a list
    cleanedQuery = lemmasAndCleaned([], query)[0]
    # getting document vector for the query
    docVector = get_doc_vec(cleanedQuery, index["word2vec"])
    # finding the cosine distance between the query and each document
    cosSims = {}
    for x in range(0, len(doc2vec)):
        dotProd = np.dot(docVector, doc2vec[x])
        magFreq = np.linalg.norm(docVector)
        magFile = np.linalg.norm(doc2vec[x])
        cosSim = (dotProd) / (magFreq * magFile)
        cosSims[x] = cosSim
    return sorted(cosSims, key=lambda x: cosSims[x], reverse=True)[:5] 

def train_word2vec(docs):
    '''
    Training the word2vec model
    Basically copied from the word embedding examples for the class
    '''
    sentences = list(corpus.brown.sents()) # the brown corpus
    # for fileid in corpus.gutenberg.fileids():
        # sentences.extend(corpus.gutenberg.sents(fileid)) 
    for fileid in corpus.reuters.fileids():
        sentences.extend(corpus.reuters.sents(fileid))
    for fileid in corpus.webtext.fileids():
        sentences.extend(corpus.webtext.sents(fileid))
    # for fileid in corpus.inaugural.fileids():
        # sentences.extend(corpus.inaugural.sents(fileid))
    for doc in docs:
        sentences.extend(doc)
    return Word2Vec(sentences, size=100, # size of the vector
            window=10,
            min_count=2,
            workers=10,
            iter=10)

def index(docs):
    files = []
    lemmas = []
    doc2VecModels = {}
    # going through each file and adding all new and unique lemmas to the list
    #  while also adding all of the lemmas from the text to the lemmas list
    for doc in docs:
        files = lemmasAndCleaned(files, doc)
    word2vec = train_word2vec(files)
    for x in range(0, len(files)):
        doc2VecModels[x] = get_doc_vec(files[x], word2vec.wv)
    return { "vectors":  doc2VecModels,
             "word2vec": word2vec.wv}

def get_doc_vec(doc, keyedvectors):
    lstDoc = [word for word in doc if word in keyedvectors]
    if len(lstDoc) != 0:
        return np.mean(keyedvectors[lstDoc], axis=0)
    else:
        return np.zeros(keyedvectors.vector_size)
