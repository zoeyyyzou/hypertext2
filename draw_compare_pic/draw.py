import os
import matplotlib.pyplot as plt
import numpy as np


def drawOnePic(label1: str, label2: str, label3: str, label4: str,
               label_list: [str], y1: [float], y2: [float], y3: [float], y4: [float],
               savePath, xlabel="Time(s)", ylabel='Bitrate(Mbps)', title: str = "", ylimLow=None, ylimHigh=None):
    x = range(len(label_list))
    l1 = plt.plot(x, y1, "r", label=label1, linestyle='dotted', marker='o',
                  markerfacecolor='none', markersize=6)
    l2 = plt.plot(x, y2, "g", label=label2, linestyle='dotted', marker='^',
                  markerfacecolor='none', markersize=6)
    l3 = plt.plot(x, y3, "b", label=label3, linestyle='dotted', marker='*',
                  markerfacecolor='none', markersize=6)
    l4 = plt.plot(x, y4, "#5E2612", label=label4, linestyle='dotted', marker='X',
                  markerfacecolor='none', markersize=6)
    print()
    print(y1)
    print(y2)
    print(y3)
    print(y4)
    print()
    if title:
        plt.title(title)
    plt.xticks([index for index in x], label_list)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylimLow is not None and ylimHigh is not None:
        plt.ylim(ylimLow, ylimHigh)
    elif ylimHigh is not None:
        plt.ylim(ylimHigh)
    plt.legend(loc=0)
    # save pictures
    full_path = os.getcwd() + "/" + savePath  # 将图片保存到当前目录，记得斜杠；可更改文件格式（.tif），不写的话默认“.png ”
    plt.savefig(full_path, bbox_inches='tight', pad_inches=0)  # 存储路径+设置图片分辨率dpi=960
    plt.close()


if __name__ == '__main__':
    plt.rc('font', family='Times New Roman', size=16)

    drawOnePic("word2vec-tree", "word2vec-svm", "fasttext", "hypertext",
               ["1w", "10w", "20w", "40w"],
               [0.56, 0.58, 0.58, 0.59],
               [0.63, 0.66, 0.66, 0.68],
               [0.84, 0.88, 0.89, 0.91],
               [0.8745, 0.8942, 0.9055, 0.9251],
               "../doc/accuracy.svg",
               xlabel="Datasets size",
               ylabel="Accuracy"
               )
    drawOnePic("word2vec-tree", "word2vec-svm", "fasttext", "hypertext",
               ["1w", "10w", "20w", "40w"],
               [0.56, 0.58, 0.58, 0.59],
               [0.62, 0.66, 0.66, 0.68],
               [0.84, 0.88, 0.89, 0.91],
               [0.8745, 0.8941, 0.9055, 0.9251],
               "../doc/f1score.svg",
               xlabel="Datasets size",
               ylabel="F1-score"
               )
    drawOnePic("word2vec-tree", "word2vec-svm", "fasttext", "hypertext",
               ["1w", "10w", "20w", "40w"],
               [14, 1 * 60 + 46, 3 * 60 + 38, 7 * 60 + 13],
               [15, 7 * 60 + 5, 26 * 60 + 59, 2 * 60 * 24 + 52 * 60 + 40],
               [0, 2, 3, 7],
               [20 * 60 + 43, 4 * 60 * 24 + 22 * 60 + 55, 6 * 60 * 24 + 45 * 60 + 10, 12 * 60 * 24 + 55 * 60 + 43],
               "../doc/train-time.svg",
               xlabel="Datasets size",
               ylabel="Train time(s)"
               )
