import gensim
import nltk
import pandas
import pickle
import os
import functools
from nltk.corpus import sentiwordnet
import collections
import numpy
import math
#- UI
#   - Like previous assignments
#   - Ask if indexing (LSA (3 versions???), doc2vec, maybe repr for recommender part)
#       - should_index() + index()
#   - Loop which of [below + quit] should do
#- keyword search
#   - Title search
#       - Like word embedding assignment
#   - Title search in genre
#       - Like prev, except filter by genre?
#           - Before, after?
#- Recommender
#   - Get ratings based on sentiments of reviews (NOT reviews in ratings.csv)
#   - Rank all unwatched movies with $r_ij = \sum(k, sim(u_i, u_k) * (r_kj - \avg(r_k))) / ratings of j$,
#       where r is ratings[uid][movie] and u is users[uid]
#   - Then print reverse-sorted list of unwatched movies by ratings

## TODO: work out index format

INDEX = 'index.pkl'

def main():
    index = get_index()
    quit = False
    while not quit:
        quit = dispatch()

def should_index():
    """ Asks the user if the index should be generated, with confirmation for unusual cases """
    if ask("Generate the index? "):
        return not (os.path.exists(INDEX) and not ask("Index exists.  Regenerate the index? "))
    else:
        return not os.path.exists(INDEX) and ask("Index does not exist.  Generate the index? ")

def get_index():
    if should_index():
        return index()
    else:
        return read_index()

## Call all your indexing functions in here.
def index():
    """ Function to do all the indexing, returns the index after writing it """
    recommender_index()

def dispatch():
    PROMPT = "[Search] by title, search by title in [genre], [recommend] a movie, or [quit]? "
    inp = input(PROMPT)
    inp = inp.strip().lower()
    if inp == '':
        print('Respond with a prefix of one of search, genre, recommend, or quit')
        return False
    if 'search'.startswith(inp):
        ## Search in all the movie titles
        return False
    elif 'genre'.startswith(inp):
        ## Do a genre search
        return False
    elif 'recommend'.startswith(inp):
        ## Get the user number and get their recommendations
        return False
    elif 'quit'.startswith(inp):
        # Tell the outer loop to finish
        return True
    else:
        print('Respond with a prefix of one of search, genre, recommend, or quit')
        return False

def recommender_index(moviesDF):
    """
    Generate the index for the recommender
    Takes in a pandas thing of moviesDF.csv
    """
    # Get TF-IDF scores
    weights = tfidf_scores(moviesDF['reviewText'])
    # Get the scores
    ratings = ratings_matrix(moviesDF, weights)
    #for reviewer in m['reviewerID'].unique():
    return ratings

def tfidf_scores(docs_iter):
    """
    Takes in an iterator of documents
    Returns a dict{word, tf-idf score}
    """
    term_freqs = dict()
    doc_freqs = dict()
    num_docs = 0
    for doc in docs_iter:
        words = score_doc(doc, term_freqs)
        for word in words:
            doc_freqs[word] = doc_freqs.get(word, 0) + 1
        num_docs += 1
    scores = term_freqs
    for key in scores:
        scores[key] /= math.log(num_docs / doc_freqs[key])
    return scores

def score_doc(doc, freq_dict):
    """
    Take in a string and a dict to put word frequencies in
    Returns a set of words in the document
    """
    ret = set()
    if isinstance(doc, str):
        for word in nltk.word_tokenize(doc):
            ret.add(word)
            freq_dict[word] = freq_dict.get(word, 0) + 1
    return ret

def ratings_matrix(moviesDF, weights):
    """
    Takes in a pandas thing with reviewerID, movieID, and reviewText parts
    Returns a [max(reviewerIDs), max(movieID)] array of reviewTexts
    """
    ret = [[""] * moviesDF['movieID'].max()] * moviesDF['reviewerID'].max()
    def closure(row):
        """
        Is a named function because python doesn't allow assignment in lambdas
        """
        if isinstance(row['reviewText'], str):
            ret[row['reviewerID'] - 1][row['movieID'] - 1] = process_text(row['reviewText'], weights)
        else:
            ret[row['reviewerID'] - 1][row['movieID'] - 1] = 0
    moviesDF.apply(closure, axis=1)
    return ret


def process_text(text, weights):
    sents = map(nltk.word_tokenize, nltk.sent_tokenize(text))
    tagged = nltk.pos_tag_sents(sents)
    ret = avg([ avg([tag_tuple(x) * weights.get(x[0], 0) for x in sent if x[0] not in nltk.corpus.stopwords.words('english') ])
            for sent in tagged])
    # return list(flatten(ret))
    return ret

def flatten(it):
    for x in it:
        if isinstance(x, str):
            yield x
        elif isinstance(x, collections.Iterable):
            for y in flatten(x):
                yield y
        else:
            yield y

def upenn_to_wn_tag(tag):
    transl = { 'JJ' : 'a', 'RB' : 'r', 'NN' : 'n', 'VB' : 'v' }
    if len(tag) >= 2:
        return transl.get(tag[:2], None)
    elif len(tag) == 1:
        tag

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

def ask(prompt):
    res = input(prompt)
    if res.lower() in ['y', 'yes']:
        return True
    return False

if __name__ == '__main__': # So the program only starts if it's not getting imported
    main()
