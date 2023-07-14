# 引入必要的python库
import numpy as np
from PIL import Image
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertConfig, BertModel
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class txtModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.txt_model = BertModel.from_pretrained('bert-base-uncased')
        self.t_linear = nn.Linear(768, 256)
        self.fc = nn.Linear(256, 3)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, image):
        # print('text_only')
        txt_out = self.txt_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_out = txt_out.last_hidden_state[:,0,:]
        txt_out.view(txt_out.shape[0], -1)
        txt_out = self.t_linear(txt_out)
        txt_out = self.relu(txt_out)
        last_out = self.fc(txt_out)
        return last_out



class imgModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_model = torchvision.models.resnet18(pretrained=True)
        self.i_linear = nn.Linear(1000, 256)
        self.fc = nn.Linear(256, 3)
        self.relu = nn.ReLU()


    def forward(self, input_ids, attention_mask, image):
        # print('image_only')
        img_out = self.img_model(image)
        img_out = self.i_linear(img_out)
        img_out = self.relu(img_out)
        last_out = self.fc(img_out)
        return last_out