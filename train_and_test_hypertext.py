import sys
import threading
import time
import os
import torch
import numpy as np
import random
from train import train
from models.Config import Config
from utils import build_dataset, get_time_dif, build_dataloader
from models import HyperText
from config import *


def print_config(config):
    print("hyperparamper config:")
    for k, v in config.__dict__.items():
        print("%s=%s" % (str(k), str(v)))


def train_and_test(dataset_dir: str, output_dir: str, embedding: str = "random",
                   model_name: str = "HyperText", dropout: float = 0.0, require_improvement: int = 6000,
                   num_epochs: int = 2, batch_size: int = 50, max_length: int = 1000,
                   learning_rate: float = 1.1e-2, embed_dim: int = 20, bucket=int(1500000), wordNgrams: int = 2,
                   lr_decay_rate: float = 0.96, use_word_segment: bool = True, min_freq: int = 1,
                   eval_per_batchs: int = 50,
                   ):
    print(f"\n\n============================== Start {dataset_dir} ==============================")
    config = Config(dataset_dir, output_dir, embedding)

    # reset config
    config.model_name = model_name
    config.save_path = os.path.join(output_dir, model_name + '.ckpt')
    config.log_path = os.path.join(output_dir, model_name + '.log')
    config.dropout = dropout
    config.require_improvement = require_improvement
    config.num_epochs = num_epochs
    config.batch_size = batch_size
    config.max_length = max_length
    config.learning_rate = learning_rate
    config.embed = embed_dim
    config.bucket = bucket
    config.wordNgrams = wordNgrams
    config.lr_decay_rate = lr_decay_rate

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, use_word_segment,
                                                           min_freq=int(min_freq))
    time_dif = get_time_dif(start_time)
    print("Finished Data loaded...")
    print("Time usage:", time_dif)

    config.n_vocab = len(vocab)
    model = HyperText.Model(config).to(config.device)

    print_config(config)
    print(model.parameters)
    train(config, model, train_data, dev_data, test_data, eval_per_batchs)


if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    os.system("mkdir -p output")

    skip = 0
    if len(sys.argv) >= 2:
        skip = int(sys.argv[1])
    for k, v in ds_yelp_hypertext_config.items():
        if skip > 0:
            skip -= 1
            continue
        train_and_test(v["path_after_data_cleaning"], f"output/20_yelp_data_cleaning_{v['count']}", num_epochs=2)
        break


