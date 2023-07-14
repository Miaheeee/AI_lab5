# 引入必要的python库
from cgitb import reset
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
import argparse
from model.bert_resnet_simple import simpleModel
from model.bert_resnet_weight import weightModel
from model.bert_densenet_weight import densenetweightModel
from model.txt_or_img import txtModel
from model.txt_or_img import imgModel


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)


# 将文本转换为词嵌入表示
def txt_embed(txt, token):
    result = token.batch_encode_plus(batch_text_or_text_pairs=txt, truncation=True, padding='max_length', max_length=32,
                                     return_tensors='pt')
    input_ids = result['input_ids']
    attention_mask = result['attention_mask']
    return input_ids, attention_mask


class NewDataset():
    def __init__(self, images, descriptions, tags, token):
        self.images = images
        self.descriptions = descriptions
        self.tags = tags
        self.input_ids, self.attention_masks = txt_embed(self.descriptions, token)

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        # print('index', idx)
        img = self.images[idx]
        des = self.descriptions[idx]
        tag = self.tags[idx]
        input_id = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        return img, des, tag, input_id, attention_mask


def train_process(model, epoch_num, optimizer, train_dataloader, valid_dataloader, train_count, valid_count):
    Loss_C = nn.CrossEntropyLoss()
    train_acc = []
    valid_acc = []
    for epoch in range(epoch_num):
        loss = 0.0
        train_cor_count = 0
        valid_cor_count = 0
        for b_idx, (img, des, target, idx, mask) in enumerate(train_dataloader):
            img, mask, idx, target = img.to(device), mask.to(device), idx.to(device), target.to(device)
            output = model(idx, mask, img)
            optimizer.zero_grad()
            loss = Loss_C(output, target)
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1)
            train_cor_count += int(pred.eq(target).sum())
        train_acc.append(train_cor_count / train_count)
        for img, des, target, idx, mask in valid_dataloader:
            img, mask, idx, target = img.to(device), mask.to(device), idx.to(device), target.to(device)
            output = model(idx, mask, img)
            pred = output.argmax(dim=1)
            valid_cor_count += int(pred.eq(target).sum())
        valid_acc.append(valid_cor_count / valid_count)
        print('Train Epoch: {}, Train_Loss: {:.4f}, Train Accuracy: {:.4f}, Valid Accuracy: {:.4f}'.format(epoch + 1,
                                                                                                           loss.item(),
                                                                                                           train_cor_count / train_count,
                                                                                                           valid_cor_count / valid_count))
    plt.plot(train_acc, label="train_accuracy")
    plt.plot(valid_acc, label="valid_accuracy")
    plt.title(model.__class__.__name__)
    plt.xlabel("Epoch")
    plt.xticks(range(epoch_num), range(1, epoch_num + 1))
    plt.ylabel("Accuracy")
    plt.ylim(ymin=0, ymax=1)
    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--model', default='bert_resnet_simple')
    parser.add_argument('--image_only', action='store_true')
    parser.add_argument('--text_only', action='store_true')
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--epoch_num', default=10, type=int)
    args = parser.parse_args()
    if args.image_only:
        model = imgModel().to(device)
    if args.text_only:
        model = txtModel().to(device)
    else:
        if (args.model == 'bert_resnet_simple'):
            model = simpleModel().to(device)
        elif (args.model == 'bert_resnet_weight'):
            model = weightModel().to(device)
        elif (args.model == 'bert_densenet_weight'):
            model = densenetweightModel().to(device)
    lr = args.lr
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    epoch_num = args.epoch_num
    images = []
    descriptions = []
    tags = []
    emo_tag = {"neutral": 0, "negative": 1, "positive": 2}
    train_dataframe = pd.read_csv("./train.txt")
    pre_trained_model = "bert-base-uncased"
    token = BertTokenizer.from_pretrained(pre_trained_model, mirror='bfsu')
    for i in range(train_dataframe.shape[0]):
        guid = train_dataframe.iloc[i]['guid']
        tag = train_dataframe.iloc[i]['tag']
        img = Image.open('./data/' + str(guid) + '.jpg')
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img = np.asarray(img, dtype='float32')
        with open('./data/' + str(guid) + '.txt', encoding='gb18030') as f:
            des = f.read()
        images.append(img.transpose(2, 0, 1))
        descriptions.append(des)
        tags.append(emo_tag[tag])
    for i in range(len(descriptions)):
        des = descriptions[i]
        word_list = des.replace("#", "").split(" ")
        words_result = []
        for word in word_list:
            if len(word) < 1:
                continue
            elif (len(word)>=4 and 'http' in word) or word[0]=='@':
                continue
            else:
                words_result.append(word)
        descriptions[i] = " ".join(words_result)
    img_txt_pairs = [(images[i], descriptions[i]) for i in range(len(descriptions))]
    # print('len(img_txt_pairs)',len(img_txt_pairs))
    # 划分训练集和验证集
    X_train, X_valid, tag_train, tag_valid = train_test_split(img_txt_pairs, tags, test_size=0.2, random_state=1458, shuffle=True)
    image_train, txt_train = [X_train[i][0] for i in range(len(X_train))], [X_train[i][1] for i in range(len(X_train))]
    image_valid, txt_valid = [X_valid[i][0] for i in range(len(X_valid))], [X_valid[i][1] for i in range(len(X_valid))]
    train_dataset = NewDataset(image_train, txt_train, tag_train, token)
    # print('len(train_dataset) ',len(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    # print('len(train_dataloader):', len(train_dataloader))
    valid_dataset = NewDataset(image_valid, txt_valid, tag_valid, token)
    # print('len(valid_dataset) ',len(valid_dataset))
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=True)
    # print('len(valid_dataloader):', len(valid_dataloader))

    train_process(model, epoch_num, optimizer, train_dataloader, valid_dataloader, len(X_train), len(X_valid))
    # 对测试集进行预测
    emo_list = ["neutral", "negative", "positive"]
    test_df = pd.read_csv("./test_without_label.txt")
    guid_list = test_df['guid'].tolist()
    tag_pre_list = []
    for idx in guid_list:
        img = Image.open('./data/' + str(idx) + '.jpg')
        img = img.resize((224,224), Image.Resampling.LANCZOS)
        image = np.asarray(img, dtype = 'float32')
        image = image.transpose(2,0,1)
        with open('./data/' + str(idx) + '.txt', encoding='gb18030') as fp:
            description = fp.read()
        input_id, mask = txt_embed([description],token)
        image = image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
        y_pred = model(input_id.to(device), mask.to(device), torch.Tensor(image).to(device))
        tag_pre_list.append(emo_list[y_pred[0].argmax(dim=-1).item()])
    
    result_df = pd.DataFrame({'guid':guid_list, 'tag':tag_pre_list})
    result_df.to_csv('./my_test.txt',sep=',',index=False)


if __name__ == '__main__':
    main()