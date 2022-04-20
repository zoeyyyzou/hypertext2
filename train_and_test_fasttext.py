import fasttext
from data_preprocessor.config import *
from tabulate import tabulate


class FastTextRecord:
    def __init__(self, name, N, p, r):
        self.name = name
        self.N = N
        self.p = p
        self.r = r


fasttext_records = []
for k, v in ds_yelp_fasttext_config.items():
    model = fasttext.train_supervised(f"datasets{os.sep}{v['dirname']}{os.sep}train.txt")
    N, p, r = model.test(f"datasets{os.sep}{v['dirname']}{os.sep}test.txt")
    fasttext_records.append(FastTextRecord(v['path'], N, p, r))
    # model = fasttext.train_supervised(f"{v['path_after_data_cleaning']}{os.sep}train.txt")
    # N, p, r = model.test(f"{v['path_after_data_cleaning']}{os.sep}test.txt")
    # fasttext_records.append(FastTextRecord(v['path_after_data_cleaning'], N, p, r))

d = []
for record in fasttext_records:
    d.append([record.name, record.N, record.p, record.r])

print(tabulate(d, headers=["Name", "Number of Sample", "Precision", "Recall"]))
