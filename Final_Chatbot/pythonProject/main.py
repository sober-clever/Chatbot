import speech_recognition as sr
import pyttsx3
from selenium import webdriver
from PIL import Image
import ipinfo
from selenium.webdriver.edge.options import Options
from PIL import Image
import matplotlib.pyplot as plt
import os


def speech_recognize():
    r = sr.Recognizer()
    harvard = sr.AudioFile(r"C:\Users\86138\Desktop\Chatbot\Audio\output.wav")
    with harvard as source:
        audio = r.record(source)
# recognize speech using Sphinx
    text = r.recognize_sphinx(audio)
    try:
        print(text)
    except sr.UnknownValueError:
        print("I'm sorry, I didn't catch what you said")
    except sr.RequestError as e:
        print("Sphinx error; {0}".format(e))
    return text


def speak(text, rate, volume):
    # 模块初始化
    engine = pyttsx3.init()

    # 设置发音速率，默认值为200
    engine.setProperty('rate', rate - 50)

    # 设置发音大小，范围为0.0-1.0
    engine.setProperty('volume', volume)

    # 设置默认的声音：voices[0].id代表男生，voices[1].id代表女生
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    # 添加朗读文本
    engine.say(text)
    # 等待语音播报完毕
    engine.runAndWait()


# txt = speech_recognize()


# speak(txt)


def get_ip():
    access_token = 'ca84b181a6ad94'
    handler = ipinfo.getHandler(access_token)
    details = handler.getDetails()
    # 获取当前城市
    city = details.city
    return city


def weather():
    city = get_ip()
    options = Options()
    options.add_argument("headless")
    # print(options)
    driver = webdriver.Edge(r'C:\Users\86138\Desktop\Chatbot\pythonProject\msedgedriver.exe', options=options)
    print(4)
    url = 'https://weathernew.pae.baidu.com/weathernew/pc?query='+city+'天气&srcid=4982'

    driver.get(url)
    width = driver.execute_script("return document.documentElement.scrollWidth")
    height = driver.execute_script("return document.documentElement.scrollHeight")
    driver.set_window_size(width, height)  # 修改浏览器窗口大小

# 获取整个网页截图
    driver.get_screenshot_as_file(r'C:\Users\86138\Desktop\Chatbot\pythonProject\weather_org.png')
# print("整个网页尺寸:height={},width={}".format(height, width))
    # im = Image.open('webpage.png')
# print("截图尺寸:height={},width={}".format(im.size[1], im.size[0]))
    pic_path = r'C:\Users\86138\Desktop\Chatbot\pythonProject\weather_org.png'
    pic_save_dir_path = r'C:\Users\86138\Desktop\Chatbot\pythonProject\weather.png'
    left, upper, right, lower = 0, 50, 200, 125
    show_cut(pic_path, left, upper, right, lower)
    image_cut_save(pic_path, left, upper, right, lower, pic_save_dir_path)


def mytime():
    options = Options()
    options.add_argument("headless")
    driver = webdriver.Edge(r'C:\Users\86138\Desktop\Chatbot\pythonProject\msedgedriver.exe', options=options)

    url = 'http://time.tianqi.com/'

    driver.get(url)
    width = driver.execute_script("return document.documentElement.scrollWidth")
    height = driver.execute_script("return document.documentElement.scrollHeight")
    driver.set_window_size(width, height)  # 修改浏览器窗口大小

    # 获取整个网页截图
    driver.get_screenshot_as_file(r'C:\Users\86138\Desktop\Chatbot\pythonProject\time.png')
    pic_path = r'C:\Users\86138\Desktop\Chatbot\pythonProject\time.png'
    pic_save_dir_path = r'C:\Users\86138\Desktop\Chatbot\pythonProject\time_out.png'
    left, upper, right, lower = 20, 420, 350, 650
    show_cut(pic_path, left, upper, right, lower)
    image_cut_save(pic_path, left, upper, right, lower, pic_save_dir_path)


def show_cut(path, left, upper, right, lower):
    """
        原图与所截区域相比较
    :param path: 图片路径
    :param left: 区块左上角位置的像素点离图片左边界的距离
    :param upper：区块左上角位置的像素点离图片上边界的距离
    :param right：区块右下角位置的像素点离图片左边界的距离
    :param lower：区块右下角位置的像素点离图片上边界的距离
     故需满足：lower > upper、right > left
    """

    img = Image.open(path)

    print("This image's size: {}".format(img.size))  # (W, H)

    plt.figure("Image Contrast")

    plt.subplot(1, 2, 1)
    plt.title('origin')
    plt.imshow(img)
    plt.axis('off')

    box = (left, upper, right, lower)
    roi = img.crop(box)

    plt.subplot(1, 2, 2)
    plt.title('roi')
    plt.imshow(roi)
    plt.axis('off')
    # plt.show()


def image_cut_save(path, left, upper, right, lower, save_path):
    """
        所截区域图片保存
    :param path: 图片路径
    :param left: 区块左上角位置的像素点离图片左边界的距离
    :param upper：区块左上角位置的像素点离图片上边界的距离
    :param right：区块右下角位置的像素点离图片左边界的距离
    :param lower：区块右下角位置的像素点离图片上边界的距离
     故需满足：lower > upper、right > left
    :param save_path: 所截图片保存位置
    """
    img = Image.open(path)  # 打开图像
    box = (left, upper, right, lower)
    roi = img.crop(box)

    # 保存截取的图片
    roi.save(save_path)


if __name__ == '__main__':
    speech_recognize()



