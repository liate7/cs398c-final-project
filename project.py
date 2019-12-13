import gensim
import nltk
import pandas
import pickle
import os
import collections
import numpy

import recommender
import doc2vec
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
        quit = dispatch(index)

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
    index = {}
    movies = pandas.read_csv('movies.csv')
    moviesDF = pandas.read_csv('moviesDF.csv')
    index['movies'] = movies
    index['doc2vec'] = doc2vec.index(movies['titles'])
    index['recommender'] = recommender.index(moviesDF)
    pickle.dump(index, open(INDEX, 'wb'))
    return index

def read_index():
    return pickle.load(open(INDEX, 'rb'))

def dispatch(index):
    PROMPT = "[Search] by title, search by title in [genre], [recommend] a movie, or [quit]? "
    inp = input(PROMPT)
    inp = inp.strip().lower()
    if inp == '':
        print('Respond with a prefix of one of search, genre, recommend, or quit')
        return False
    if 'search'.startswith(inp):
        search(index)
        return False
    elif 'genre'.startswith(inp):
        ## Do a genre search
        return False
    elif 'recommend'.startswith(inp):
        recommend(index)
        return False
    elif 'quit'.startswith(inp):
        # Tell the outer loop to finish
        return True
    else:
        print('Respond with a prefix of one of search, genre, recommend, or quit')
        return False

def search(index):
    query = input("Searching for: ")
    i = index['doc2vec']
    for result in doc2vec.query(query, i):
        print(index['movies']['movieId'][result], index['movies']['title'][result])
    
def recommend(index):
    query = int(input("User ID (number): "))
    i = index['recommender']
    for result in recommender.query(query, i):
        if numpy.isnan(result):
            break
        print(index['movies']['movieId'][result], index['movies']['title'][result])

def ask(prompt):
    res = input(prompt)
    if res.lower() in ['y', 'yes']:
        return True
    return False

if __name__ == '__main__': # So the program only starts if it's not getting imported
    main()
