import torch
import random
import json
import nltk
import numpy
import torch.nn as nn

device = torch.device('cpu')

with open('traindata.json', 'r') as f:
    TrainArray = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
words = data['words']
tags = data['tags']
model_state = data["model_state"]


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


chatnn = ChatNN(input_size, hidden_size, output_size).to(device)
chatnn.load_state_dict(model_state)
chatnn.eval()


def trans_to_num(sen):
    # 这里的sentence已经是处理后的单词数组
    num_tmp = numpy.zeros(shape=len(words), dtype=numpy.float32)
    # numpy的数据格式有严格要求，此处先初始化
    for index in range(len(words)):
        if words[index] in sen:
            num_tmp[index] = 1
    return num_tmp


print("Alice: Hello!")

while True:
    your_sentence = input()

    sentence = nltk.word_tokenize(your_sentence)

    X = trans_to_num(sentence)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = chatnn(X)
    tmp, index = torch.max(output, dim=1)
    index = index.item()

    tag = tags[index]

    ProbsVector = torch.softmax(output, dim=1)
    # print(output)
    # print(probs)
    # print(probs[0])
    prob = ProbsVector[0][index].item() # 先取[0]

    if prob > 0.80:
        for i in TrainArray['traindata']:
            if tag == i["tag"]:
                print("Alice: ", random.choice(i['responses']))
    else:
        print("Alice: I do not understand...")
