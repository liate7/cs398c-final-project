#Ali Muhammed, heavily edited by Andrew Patterson to work with everything else
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def index(docs, size_frac):
    vectorizer = TfidfVectorizer(stop_words='english', max_df = 0.5, smooth_idf=True)
    matrix = vectorizer.fit_transform(docs)
    svder = TruncatedSVD(n_components = int(matrix.get_shape()[1] * size_frac))
    return vectorizer, svder, svder.fit_transform(matrix)

def query(query, index):
    vectorizer, svder, matrix = index
    query = vectorizer.fit_transform(query)


