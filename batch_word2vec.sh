mkdir -p train_and_test_result
python train_and_test_word2vec.py 0 > train_and_test_result/yelp_word2vec_1w.log &
python train_and_test_word2vec.py 1 > train_and_test_result/yelp_word2vec_10w.log &
python train_and_test_word2vec.py 2 > train_and_test_result/yelp_word2vec_20w.log &
python train_and_test_word2vec.py 3 > train_and_test_result/yelp_word2vec_40w.log &