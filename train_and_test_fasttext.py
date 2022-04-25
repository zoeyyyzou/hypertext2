import time

import fasttext
from data_preprocessor.config import *
from sklearn.metrics import classification_report
from tqdm import tqdm
from models.hypertext_model.utils import get_time_dif

if __name__ == '__main__':
    # fasttext_records = []
    for k, v in ds_yelp_fasttext_config.items():
        dir = f"datasets{os.sep}{v['dirname']}"
        print(f"\n======================= {dir} =======================")
        start_time = time.time()
        model = fasttext.train_supervised(f"{dir}{os.sep}train.txt")
        # N, p, r = model.test(f"datasets{os.sep}{v['dirname']}{os.sep}test.txt")
        fasttext_train_time = get_time_dif(start_time)
        print(f"fasttext train time: {fasttext_train_time}")
        predicted = []
        y_test = []
        with open(f"{dir}{os.sep}test.txt") as f:
            for line in tqdm(f.readlines()):
                line = line.strip()
                y_test.append(int(line[-1]))
                # # tuple => ( ('__label__0'), array([0.7]) )
                # ttt = model.predict(line[:-11])
                # # ('__label__0')
                # ttt[0]
                # # __label__0
                # ttt[0][0]
                # # 0
                # ttt[0][0][-1]
                predicted.append(int(model.predict(line[:-11])[0][0][-1]))

        print(classification_report(y_test, predicted, target_names=["class 0", "class 1"]))
