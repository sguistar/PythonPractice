from model.Transformer import EncoderComponent,DecoderComponent,Transformer,ScaleDotProductAttention,MultiHeadAttention
import torch
from torch import nn as nn
from torch import optim as optim
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import d2l
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#@save
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

#@save
def read_data_nmt():
    """载入“英语－法语”数据集"""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
             encoding='utf-8') as f:
        return f.read()

raw_text = read_data_nmt()
print(raw_text[:75])