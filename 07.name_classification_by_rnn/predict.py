#encoding utf-8

from model import *
from data import *

import sys

rnn = torch.load("char-rnn-classification.pt")

def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i],hidden)

    return output

def predict(line, n_predictions=3):
    output = evaluate(lineToTensor(line))

    topv, topi = output.data.topk(n_predictions,1,True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print(value, all_categories[category_index])
        predictions.append([value, all_categories[category_index]])

if __name__ == "__main__":
    predict(input())
