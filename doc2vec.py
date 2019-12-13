import numpy as np
import nltk
from collections import defaultdict
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import math
import gensim
from gensim import models
import os
from scipy.spatial.distance import jaccard
import pickle as p
from gensim.models import doc2vec, Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# read documents in folder called os.listdir(path)
#  lstFiles = os.listdir('corpus')

def main():
    menu()

def menu():
    action = ''
    while action != '5':
        action = input('index(1), query(2), doc2vec index(3), word2vec index(4) or quit(5)? ')
        if action == '1':
            index()
        elif action == '2':
            query()
        elif action == '3':
            doc2vec()
        elif action == '4':
            word2Vec()

def index():
    # setting the answer to weather or not they would like to create a new index to yes
    answer = '1'
    # creating the frequency dictionary
    frequency = defaultdict(int)
    if(os.stat('index.p').st_size > 0):
        answer = input('an index already exists. Would you like to replace it(1)? Or not(2)? ')
    if(answer == '1'):
        listFiles = os.listdir('corpus')
        files = []
        lemmas = []
        cleanedText = []
        termToDoc = []
        frequency = []
        # going through each file and adding all new and unique lemmas to the list
        #  while also adding all of the lemmas from the text to the lemmas list
        for file in listFiles:
            lemmas, files = lemmasAndCleaned(lemmas, files, file)
        # comparing each word with each document to find the frequency of each word
        #  and create a a term to document matrix
        termToDoc = termsToDoc(lemmas, termToDoc, files)
        # creating the term to document matrix from the list of each word's list of
        #  frequency in each document
        termDocMatrix = np.matrix(termToDoc)
        # getting the svd and setting t and s to 10
        t, s, v = np.linalg.svd(termDocMatrix, full_matrices=True)
        t2 = t[:,:10]
        s2 = np.diag(s[:10])
        try:
            index = open('index.p', 'wb')
        except:
            print("could not open index.p")
        # making svd dictionary to be dumped with the list of words and the document vectors
        svd = {'t2':t2,
               's2':s2
               }
        words = {'lemmas':lemmas
                 }
        docVectors = {}
        # creating all of the document vectors from the t and s
        for x in range(0, len(files)):
            docNum = x + 1
            name = "doc{}".format(docNum)
            docMatrix = []
            for y in range(0, len(lemmas)):
                docMatrix.append(termDocMatrix[y, x])
            vd = np.matrix([docMatrix])
            toDump = vd * t2 * s2
            toDump = np.array(toDump).flatten()
            docVectors[name] = toDump
        # making a final dictionary that holds a dictionary for the words, document vectors, and svd
        finalDump = {'svd':svd,
                     'documents':docVectors,
                     'lemmas':words
                     }
        # dumping the final dictionary
        p.dump(finalDump, index)
    elif(answer == '2'):
        print('New index NOT created.')

def termsToDoc(lemmas, termToDoc, files):
    for word in lemmas:
        frequency = []
        for file in files:
            count = 0
            for lemma in file:
                if(word == lemma):
                    count += 1
            frequency.append(count)
        termToDoc.append(frequency)
    return termToDoc

def lemmasAndCleaned(lemmas, files, file):    
    lemmatizer = WordNetLemmatizer()
    stoplist = set(stopwords.words('english'))
    cleanedText = []
    try:
        text = open('corpus//' + file).read()
    except:
        print("could not open file")
    text = text.lower()
    text = cleanPunct(text)
    text = word_tokenize(text)
    for word in text:
        lemma = lemmatizer.lemmatize(word)
        if(word not in stoplist):
            cleanedText.append(lemma)
            if(word not in lemmas):
                lemmas.append(lemma)
    files.append(cleanedText)
    return lemmas, files
        
def cleanPunct(strSent):    
    punct = '!()-[]{};:"\,<>./?@|#$%^&*_~'
    cleanSent=""    
    for char in strSent:
        if char not in punct:
            cleanSent+=char            
    return cleanSent

def query():
    # making sure there is an index
    if(os.stat('index.p').st_size == 0):
        print('an index does not exist. Please create one.')
    elif(os.stat('word2vec.p').st_size == 0):
        print('a word2vec index does not exist. Please create one.')
    elif(os.stat('doc2vec.p').st_size == 0):
        print('a doc2vec index does not exist. Please create one.')
    else:    
        lemmatizer = WordNetLemmatizer()
        stoplist = set(stopwords.words('english'))
        try:
            file = open('index.p', 'rb')
        except:
            print("could not open index.p")
        index = p.load(file)
        try:
            file = open('doc2vec.p', 'rb')
        except:
            print("could not open doc2vec.p")
        doc2vec = p.load(file)
        try:
            file = open('word2vec.p', 'rb')
        except:
            print("could not open word2vec.p")
        word2vec = p.load(file)
        query = ''
        # getting the query and making sure it should be executed
        while(query != '-1'):
            query = input('Query (enter "-1" to quit): ')
            frequency = []
            if(query != '-1'):
                # cleaning the query and putting each word into a list
                cleanedQuery = []
                newQuery = query.lower()
                newQuery = cleanPunct(newQuery)
                newQuery = word_tokenize(newQuery)
                for word in newQuery:
                    lemma = lemmatizer.lemmatize(word)
                    if(word not in stoplist):
                        cleanedQuery.append(lemma)
                # checking to see if all of the words in the query appear in the corpus
                #  if none, tell all documents are equal distant
                #  if at least one, calculate distance and print
                noShow = 0
                for word in cleanedQuery:
                    if(word not in index['lemmas']['lemmas']):
                        noShow += 1
                if(noShow == len(cleanedQuery)):
                    print('This word is not in the corpus therefore all documents are equal distance')
                else:
                    for word in index['lemmas']['lemmas']:
                        count = 0
                        for lemma in cleanedQuery:
                            if(word == lemma):
                                count += 1
                        frequency.append(count)
                # making the frequency of each word in the query into a matrix
                #  and then multiplying by s and t to get the vector
                    frequencyVector = np.array(frequency)
                    frequencyMatrix = np.matrix(frequencyVector)
                    freqVector = frequencyMatrix * (index['svd']['t2']) * (index['svd']['s2'])
                    freqVector = np.array(freqVector).flatten()
                answer = input('LSA query(1), word2vec query(2), or doc2vec query(3)? ')
                if(answer == '1'):
                # making a dictionary of cosine distances
                    cosDists = {}
                # finding the cosine distance between the query and each document
                    for x in range(0, len(index['documents'])):
                        docNum = x + 1
                        name = "doc{}".format(docNum)
                        dotProd = np.dot(freqVector, index['documents'][name])
                        magFreq = np.linalg.norm(freqVector)
                        magFile = np.linalg.norm(index['documents'][name])
                        cosDist = (dotProd) / (magFreq * magFile)
                        cosDists[name] = cosDist
                # printing all of the distances
                    print("For query of '{}':".format(query)) 
                    for doc,dist in cosDists.items():
                        print(doc, " has a cosine distance of ", dist)
                elif(answer == '2'):
                    listFiles = os.listdir('corpus')
                    files = []
                    lemmas = []
                    cleanedText = []
                    termToDoc = []
                    frequency = []
        # going through each file and adding all new and unique lemmas to the list
        #  while also adding all of the lemmas from the text to the lemmas list
                    for file in listFiles:
                        lemmas, files = lemmasAndCleaned(lemmas, files, file)
                    wordavg = 0
                    count = 0
                    totalavg = {}
                    for x in range(0, len(word2vec['word2Vec'])):
                        docNum = x + 1
                        name = "doc{}".format(docNum)
                        for word in files[x]:
                            for lemma in cleanedQuery:
                                if word2vec['word2Vec'][name].wv.__contains__(lemma.lower()):
                                    wordavg += word2vec['word2Vec'][name].wv.similarity(word, lemma)
                                count += 1
                        avg = wordavg / count
                        totalavg[name] = avg
                        wordavg = 0
                        count = 0
                    for x in range(0, len(totalavg)):
                        docNum = x + 1
                        name = "doc{}".format(docNum)
                        print(name, " has a distance of ", totalavg[name])
                elif(answer == '3'):
                # finding the cosine distance between the query and each document
                    cosDists = {}
                    for x in range(0, len(doc2vec)):
                        docNum = x + 1
                        name = "doc{}".format(docNum)
                        dotProd = np.dot(freqVector, doc2vec[name])
                        magFreq = np.linalg.norm(freqVector)
                        magFile = np.linalg.norm(doc2vec[name])
                        cosDist = (dotProd) / (magFreq * magFile)
                        cosDists[name] = cosDist
def word2Vec():
    # setting the answer to weather or not they would like to create a new index to yes
    answer = '1'
    # creating the frequency dictionary
    frequency = defaultdict(int)
    if(os.stat('word2vec.p').st_size > 0):
        answer = input('an index already exists. Would you like to replace it(1)? Or not(2)? ')
    if(answer == '1'):
        listFiles = os.listdir('corpus')
        files = []
        lemmas = []
        cleanedText = []
        termToDoc = []
        frequency = []
        word2VecModels = {}
        # going through each file and adding all new and unique lemmas to the list
        #  while also adding all of the lemmas from the text to the lemmas list
        for file in listFiles:
            lemmas, files = lemmasAndCleaned(lemmas, files, file)
        for x in range(0, len(files)):
            model = Word2Vec(files[x])
            docNum = x + 1
            name = "doc{}".format(docNum)
            word2VecModels[name] = model
        word2Vec = {'word2Vec': word2VecModels}
        p.dump(word2Vec, open('word2vec.p', 'wb'))
    elif(answer == '2'):
        print('New index NOT created.')

def doc2vec():
     # setting the answer to whether or not they would like to create a new index to yes
    answer = '1'
    # creating the frequency dictionary
    frequency = defaultdict(int)
    if(os.stat('doc2vec.p').st_size > 0):
        answer = input('an index already exists. Would you like to replace it(1)? Or not(2)? ')
    if(answer == '1'):
        listFiles = os.listdir('corpus')
        files = []
        lemmas = []
        cleanedText = []
        termToDoc = []
        frequency = []
        doc2VecModels = {}
        # going through each file and adding all new and unique lemmas to the list
        #  while also adding all of the lemmas from the text to the lemmas list
        for file in listFiles:
            lemmas, files = lemmasAndCleaned(lemmas, files, file)
        word2vec = p.load(open('word2vec.p', 'rb'))['word2Vec']
        for x in range(0, len(files)):
            docNum = x + 1
            name = "doc{}".format(docNum)
            lstDoc = [word for word in files[x] if word2vec[name].wv.__contains__(word)]
            doc2VecModels[name] = lstDoc
        p.dump(doc2VecModels, open('doc2vec.p', 'wb'))

main()
