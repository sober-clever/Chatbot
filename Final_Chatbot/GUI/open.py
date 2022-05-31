import os


def search_open(key):
    flag = 0
    file = r"C:\Users\86138\Desktop\Chatbot\GUI\exe.txt"
    key_temp = key.lower()
    keywords = ["Typora", "vmware", "QQ", "VS Code" , "msedge" , "pycharm64" , "WeChat" , "YoudaoDict" , "steam"]
    for keyword in keywords:
        keyword_temp = keyword.lower()
        if keyword_temp in key_temp:
            key = keyword
    f = open(file)
    for line in f:
        line = line.strip('\n')
        if key in line:
            # app_dir = r'line'
            # print(line)
            # os.system(line)
            flag = 1
            os.startfile(line)
    return flag


if __name__ == '__main__':
    key = input("input:")
    search_open(key)
