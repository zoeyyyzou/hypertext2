import os
from gensim.models import Word2Vec
import time
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import pickle


def train_word2vec(input_dir, output_dir, size: int = 100, window: int = 3, min_count: int = 1,
                   workers: int = 4, sg: int = 1):
    """
    size: The number of dimensions of the embeddings and the default is 100.
    window: The maximum distance between a target word and words around the target word. The default window is 5.
    min_count: The minimum count of words to consider when training the model; words with occurrence less than this
                count will be ignored. The default for min_count is 5.
    workers: The number of partitions during training and the default workers is 3.
    sg: The training algorithm, either CBOW(0) or skip gram(1). The default training algorithm is CBOW.
    """
    word2vec_model_file = f"{output_dir}{os.sep}word2vec_{str(size)}.model"

    data_df = pd.concat([pd.read_csv(f"{input_dir}{os.sep}train.csv"), pd.read_csv(f"{input_dir}{os.sep}test.csv")])

    start_time = time.time()
    stemmed_tokens = pd.Series(data_df['stemmed_tokens']).values
    # Train the Word2Vec Model
    w2v_model = Word2Vec(stemmed_tokens, min_count=min_count, vector_size=size, workers=workers, window=window, sg=sg)
    print("Time taken to train word2vec model: " + str(time.time() - start_time))
    w2v_model.save(word2vec_model_file)


def generate_word2vec_vectors(input_dir, output_dir, size: int = 100):
    sg_w2v_model = Word2Vec.load(f"{input_dir}{os.sep}word2vec_{str(size)}.model")
    word2vec_filename = f"{output_dir}{os.sep}train_review_word2vec.csv"
    X_train = pd.read_csv(f"{input_dir}{os.sep}train.csv")

    with open(word2vec_filename, 'w+') as word2vec_file:
        for index, row in X_train.iterrows():
            model_vector = (np.mean([sg_w2v_model.wv[token] for token in row['stemmed_tokens']], axis=0)).tolist()
            if index == 0:
                header = ",".join(str(ele) for ele in range(size))
                word2vec_file.write(header)
                word2vec_file.write("\n")
            # Check if the line exists else it is vector of zeros
            if type(model_vector) is list:
                line1 = ",".join([str(vector_element) for vector_element in model_vector])
            else:
                line1 = ",".join([str(0) for i in range(size)])
            word2vec_file.write(line1)
            word2vec_file.write('\n')


def train_decision_tree(input_dir, output_dir):
    word2vec_filename = f"{output_dir}{os.sep}train_review_word2vec.csv"
    Y_train = pd.read_csv(f"{input_dir}{os.sep}train.csv")

    # Load from the filename
    word2vec_df = pd.read_csv(word2vec_filename)
    # Initialize the model
    clf_decision_word2vec = DecisionTreeClassifier()

    start_time = time.time()
    # Fit the model
    clf_decision_word2vec.fit(word2vec_df, Y_train['sentiment'])
    print("Time taken to fit the model with word2vec vectors: " + str(time.time() - start_time))

    with open(f'{output_dir}/model.pkl', 'wb') as f:
        pickle.dump(clf_decision_word2vec, f)


def test_decision_tree(input_dir, size: int = 100):
    with open(f'{input_dir}/model.pkl', 'rb') as f:
        clf_decision_word2vec = pickle.load(f)
    X_test = pd.read_csv(f"{input_dir}{os.sep}test.csv")
    sg_w2v_model = Word2Vec.load(f"{input_dir}{os.sep}word2vec_{str(size)}.model")
    test_features_word2vec = []
    for index, row in X_test.iterrows():
        model_vector = np.mean([sg_w2v_model.wv[token] for token in row['stemmed_tokens']], axis=0)
        if type(model_vector) is list:
            test_features_word2vec.append(model_vector)
        else:
            test_features_word2vec.append(np.array([0 for i in range(size)]))
    test_predictions_word2vec = clf_decision_word2vec.predict(test_features_word2vec)
    print(X_test['sentiment'].value_counts())
    print(classification_report(X_test['sentiment'], test_predictions_word2vec))


if __name__ == '__main__':
    """
    size: The number of dimensions of the embeddings and the default is 100.
    window: The maximum distance between a target word and words around the target word. The default window is 5.
    min_count: The minimum count of words to consider when training the model; words with occurrence less than this 
                count will be ignored. The default for min_count is 5.
    workers: The number of partitions during training and the default workers is 3.
    sg: The training algorithm, either CBOW(0) or skip gram(1). The default training algorithm is CBOW.
    """

    dir = "datasets/yelp_word2vec_1w"
    size = 100

    # 1. train word2vec
    print("1. Start train word2vec")
    train_word2vec(dir, dir, size=size)
    #
    # # 2. generate word2vec vectors
    print("2. Start  generate word2vec vectors")
    generate_word2vec_vectors(dir, dir, size=size)

    # 3. train decision tree
    print("3. Start train decision tree")
    train_decision_tree(dir, dir)

    # 4. test decision tree
    print("4. Start test decision tree")
    test_decision_tree(dir, size=size)
