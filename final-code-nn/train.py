# 训练代码，核心是神经网络
import torch
import torch.nn as nn
import numpy
import nltk
import json
from torch.utils.data import Dataset, DataLoader


class ChatNN(nn.Module):
    # 需要自己写初始化和forward函数
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatNN, self).__init__()   # 默认要加这一行，调用父类的初始化操作
        self.linear0 = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, input_x):
        # 拼写检查要注意
        output_mid = self.linear0(input_x)
        output_mid = self.relu(output_mid)
        output_mid = self.linear1(output_mid)
        output_mid = self.relu(output_mid)
        output_mid = self.linear1(output_mid)
        output_mid = self.relu(output_mid)
        output_mid = self.linear1(output_mid)
        output_mid = self.relu(output_mid)
        output_final = self.linear2(output_mid)
        return output_final


with open('traindata.json', 'r') as f:
    TrainArray = json.load(f)


def trans_to_words(sen):
    # 将句子拆分成单词，忽略部分标点符号，处理一些不文明单词
    # 返回一个word数组
    words_tmp = []
    word_array = nltk.word_tokenize(sen)
    # print(word_array)
    for s_word in word_array:
        s_word = s_word.lower()
        if s_word == ',' or s_word == ';' or s_word == ':' or s_word == '!' or s_word == '?':
            # 无需考虑 ，但是要考虑 .?!
            continue
        if s_word == 'fuck' or s_word == 'bitch' or s_word == 'sb':
            # 处理 dirty words
            continue
        words_tmp.append(s_word)
    # print(words_tmp)
    return words_tmp


tags = []
words = []
sentence_tag = []  # 句子与标签的一一对应，是个元组的数组，但是其中的句子实际上是单词数组
train_input = []
train_output = []

for i in TrainArray['traindata']:
    tag = i['tag']
    tags.append(tag)
    for input_sentence in i['input_sen']:
        # print(input_sentence)
        tmp = trans_to_words(input_sentence)
        # print(tmp)
        words.extend(tmp)
        sentence_tag.append((tmp, tag))

words = sorted(set(words))


def trans_to_num(sen):
    # 这里的sentence已经是处理后的单词数组
    num_tmp = numpy.zeros(shape=len(words), dtype=numpy.float32)
    # numpy的数据格式有严格要求，此处先初始化
    for index in range(len(words)):
        if words[index] in sen:
            num_tmp[index] = 1
    return num_tmp


# 将输入输出转换为数字
for (sentence, tag) in sentence_tag:
    # 这里的sentence实际上已经是处理后得到的单词数组
    sen_num = trans_to_num(sentence)
    train_input.append(sen_num)
    tag_num = tags.index(tag)
    train_output.append(tag_num)

train_input = numpy.array(train_input)
train_output = numpy.array(train_output)

epoch = 4000


class ChatDataset(Dataset):
    def __init__(self):
        self.x = train_input
        self.y = train_output
        self.num = len(train_input)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.num


chatdataset = ChatDataset()
chatdataloader = DataLoader(dataset=chatdataset, batch_size=8, shuffle=True)
# 创建dataset

device = torch.device('cpu')

chatnn = ChatNN(len(words), 16, len(tags)).to(device)
# 实例化一个神经网络的对象

criterion = nn.CrossEntropyLoss()       # 交叉熵
optimizer = torch.optim.Adam(chatnn.parameters(), lr=0.001)     # 优化器

for cnt in range(epoch):
    for (sen_tensor, tag) in chatdataloader:

        sen_tensor = sen_tensor.to(device)
        tag = tag.to(dtype=torch.long).to(device)

        results = chatnn(sen_tensor)

        loss = criterion(results, tag)
        # 实际输出与与其输出的差别

        optimizer.zero_grad()
        # 必须先用0填充，防止逐次累加

        loss.backward()
        optimizer.step()

    if (cnt+1) % 100 == 0:
        print("epoch:", cnt+1)



data = {
    "model_state": chatnn.state_dict(),
    "input_size": len(words),
    "hidden_size": 16,
    "output_size": len(tags),
    "words": words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)
