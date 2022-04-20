from gensim.models import Word2Vec
from gensim.test.utils import common_texts
if __name__ == '__main__':
    model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
    model.save("word2vec.model")
