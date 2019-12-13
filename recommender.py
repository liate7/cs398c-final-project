from nltk.corpus import sentiwordnet
import numpy as np
import pandas
import nltk
# This file is all Andrew's code (except for the one bit of matrix math to do
# the cosine distances of all the vectors at once, which has attribution

def index(moviesDF):
    """
    Generate the index for the recommender
    Takes in a pandas thing of moviesDF.csv
    """
    tagged = tag_docs(moviesDF['reviewText'])
    lemmas, docs = lemmatize(tagged)
    matrix = make_matrix(list(lemmas), docs)
    idfs = np.log(matrix.shape[1] / np.count_nonzero(matrix, 0))
    tfidf = np.apply_along_axis(lambda row: (row / np.sum(row)) * idfs, 1, matrix)
    # list of (average) sentiments per lemma, tag
    lemma_sentiments = np.array( [tag_tuple(x) for x in lemmas] )
    # doc -> weight array!
    doc_sentiments = np.apply_along_axis(lambda row: np.sum(row * lemma_sentiments),
            1, tfidf)
    # matrix is 0-based, while dataframe is 1-based, so indices are ID-1, not ID
    scores_matrix = doclist_to_matrix(moviesDF, doc_sentiments)
    similarities = cosine_sim_table(np.nan_to_num(scores_matrix))
    # Gets a matrix vectors [ sum(sentiment_i * weight_i) foreach movie ]
    weights = np.nan_to_num(scores_matrix).dot(similarities)
    indices_sorted = np.fliplr(np.argsort(weights))
    # Finally, return a list of recommendations for each user
    return recommended_matrix(indices_sorted, scores_matrix)

def query(user, index):
    return index[user-1]

def tag_docs(dociter):
    """
    Take in an iterable of documents
    Returns an iterable of the documents POS-tagged with wordnet tags
    """
    for doc in dociter:
        tagged = []
        if isinstance(doc, str):
            tagged = nltk.pos_tag(nltk.word_tokenize(doc.lower()))
        yield list([(word[0], upenn_to_wn_tag(word[1])) for word in tagged])

def lemmatize(dociter):
    """
    Takes in an iterable of tagged documents
    Returns a set of all the lemmas and a list of the documents lemmatized
    """
    lemmas = set()
    documents = list()
    wn = nltk.WordNetLemmatizer()
    for doc in dociter:
        lemmatized = []
        for word, tag in doc:
            lemma = ( wn.lemmatize(word, tag), tag )
            lemmatized.append(lemma)
            lemmas.add(lemma)
        documents.append(lemmatized)
    return lemmas, documents

def make_matrix(lemmas, docs):
    """
    Takes an list of lemmas and a list of lemmatized docs
    Returns a num(docs) by num(lemmas) matrix of word counts
    """
    matrix = np.zeros((len(docs), len(lemmas)))
    doclen = len(docs)
    lemmalen = len(lemmas)
    for doc in range(doclen):
        counts = dict()
        for lemma in docs[doc]:
            counts[lemma] = counts.get(lemma, 0) + 1
        for lemma in range(lemmalen):
            matrix[doc][lemma] = counts.get(lemmas[lemma], 0)
    return matrix

def doclist_to_matrix(moviesDF, doc_sent):
    """
    Takes in a pandas dataframe with reviewerID and movieID columns and a list
    of sentiments
    Returns a matrix(reviewerID, movieID) of sentiments
    """
    ret = np.full((moviesDF['reviewerID'].max(), moviesDF['movieID'].max()), np.nan)
    for i in range(len(doc_sent)):
        ret[moviesDF['reviewerID'][i] - 1][moviesDF['movieID'][i] - 1] = doc_sent[i]
    return np.apply_along_axis(variance, 1, ret)

def variance(arr):
    return arr - np.average(np.nan_to_num(arr))

def upenn_to_wn_tag(tag):
    transl = { 'JJ' : 'a', 'RB' : 'r', 'NN' : 'n', 'VB' : 'v' }
    if len(tag) >= 2:
        return transl.get(tag[:2], 'n')
    else:
        return 'n'

def tag_tuple(tupl):
    def weight(synset):
        return synset.pos_score() - synset.neg_score()
    word, tag = tupl
    synsets = sentiwordnet.senti_synsets(word, upenn_to_wn_tag(tag))
    return avg([weight(s) for s in synsets])

def avg(it):
    i = 0
    acc = 0
    for x in it:
        acc += x
        i += 1
    if i == 0:
        return 0
    else:
        return acc / i

def cosine_sim_table(matrix):
    '''
    Takes in a matrix of shape (x,y)
    Returns a matrix of shape (y,y) of the cosine similarities in a table
    '''
    # This code is from <https://stackoverflow.com/a/37929303>
    # It only halfway makes sense to me, I don't really have the linear algebra
    # knowledge to get it, but it should be faster than a looping version
    #
    # cos(v1,v2) = (v1 `dot` v2) / (norm(v1) * norm(v2))
    # matrix.T `dot` matrix -> matrix of dot products
    # norm is vector of normalized arrays
    norm = np.linalg.norm(matrix, axis=0)[None, :]
    return matrix.T.dot(matrix) / norm.T.dot(norm)

def recommended_matrix(sorted_indices, sentiments_matrix):
    '''
    Take in a matrix of indices sorted in most recommended-least recommended, and a matrix of sentiments
    Returns the indices of the first five that are nan (not reviewed) in the sentiments matrix
    '''
    ret = np.full((sorted_indices.shape[0], 5), np.nan)
    for user in range(sorted_indices.shape[0]):
        i = 0
        for index in sorted_indices[user]:
            if np.isnan(sentiments_matrix[user][index]):
                ret[user][i] = index
                i += 1
                if i == 5:
                    break
    return ret
