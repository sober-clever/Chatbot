# Demo of the final GUI of the chatbot.
import sys, PyQt5, os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QToolTip,
                             QHBoxLayout, QVBoxLayout, QGridLayout, QComboBox,
                             QDesktopWidget, QLineEdit, QSplitter, QAction, QFrame,
                             QWidget, QTextBrowser, QTextEdit, QLabel, QScrollArea,
                             QMessageBox, qApp, QTextEdit, QStackedLayout, QFileDialog)
from PyQt5.QtGui import QIcon, QFont, QFontMetrics, QPixmap, QMovie
from PyQt5.QtCore import Qt, QEvent, QPropertyAnimation, QRect, QThread, pyqtSignal
from qt_material import apply_stylesheet
from finalnn.main import GetReply
from Audio.main import recording, noise_reduce
import time
from GUI.history import HistoryUI
from pythonProject1.xunfei_test import ws, wsParam, wsUrl, on_message, on_error, on_open
from pythonProject2.main import voc, evaluateInput, encoder, decoder, searcher
import torch.nn as nn
import websocket, ssl
from Speech.main import speak
from GUI.open import search_open
from pythonProject.main import weather, mytime
from threading import Thread
from final_code_nn.nn_main import getreply


class Worker_Record(QThread):
    sinOut = pyqtSignal()
    
    def __init__(self):
        super(Worker_Record, self).__init__()

    def run(self):
        recording(r'C:\Users\86138\Desktop\Chatbot\Audio\input.wav')
        # recording(r'C:\Users\Tanjf\Desktop\Chatbot\Audio\noise.wav')
        noise_reduce()

        websocket.enableTrace(False)
        wsUrl = wsParam.create_url()
        ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error)

        ws.on_open = on_open
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        # 语音识别
        self.sinOut.emit()

class Worker_FindTime(QThread):
    sinOut = pyqtSignal()

    def __init__(self):
        super(Worker_FindTime, self).__init__()

    def run(self):

        self.sinOut.emit()

class Worker_Speak(QThread):
    # sinOut = pyqtSignal()
    reply = ""
    def __init__(self):
        super(Worker_Speak, self).__init__()

    def run(self):
        speak(self.reply, 200, 0.5)
        # self.sinOut.emit()

class MainUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    # 初始化 UI 界面
    def initUI(self):

        year = time.strftime('%Y', time.localtime(time.time()))
        month = time.strftime('%m', time.localtime(time.time()))
        day = time.strftime('%d', time.localtime(time.time()))
        t = time.strftime('%m%d %H%M%S', time.localtime(time.time()))

        fileYear = 'C:\\Users\\86138\\Desktop\\Chatbot\\History\\' + year
        fileMonth = fileYear + '/' + month
        # fileDay = fileMonth + '/' + day

        # self.filename = fileDay

        if not os.path.exists(fileYear):
            os.mkdir(fileYear)
            os.mkdir(fileMonth)
            # os.mkdir(fileDay)
        else:
            if not os.path.exists(fileMonth):
                os.mkdir(fileMonth)
                # os.mkdir(fileDay)
            # else:
            # if not os.path.exists(fileDay):
            # os.mkdir(fileDay)

        # 创建一个文件，以‘timeFile_’+具体时间为文件名称

        self.record_thread = Worker_Record()

        self.histo_file = fileMonth + '/timeFile_' + str(year) + '_' + str(month) + '_' + str(day) + '.txt'

        QToolTip.setFont(QFont('SansSerif', 20))

        self.entry_cnt = 0  # 用于记录聊天记录的总条数
        self.setFont(QFont('SanSerif', 10))
        self.setToolTip('Chatarea')

        exitAct = QAction(QIcon('img/Ubuntu.jpeg'), '&Exit', self)
        # QAction 是菜单栏、工具栏或者快捷键的动作的组合
        # 创建了一个图标、一个exit的标签
        exitAct.setShortcut('Ctrl+W')
        # 创建快捷键
        exitAct.setStatusTip('Exit application.')
        # 创建状态栏，当鼠标悬停在菜单栏时，能显示当前状态
        exitAct.triggered.connect(self.close)
        # 当执行这个指定的动作时，就触发了一个事件
        # 这个事件跟 QApplication 的 quit() 行为相关联

        sendAct = QAction(QIcon('img/Ubuntu.jpeg'), "&Send", self)
        sendAct.setShortcut(Qt.CTRL + Qt.Key_S)
        # sendAct.setVisible(False)
        sendAct.setStatusTip('Send Message.')
        sendAct.triggered.connect(self.DoAnim)
        self.reply_thread = Worker_FindTime()
        self.reply_thread.sinOut.connect(self.Reply)
        self.speak_thread = Worker_Speak()

        avatarAct = QAction(QIcon('img/Ubuntu.jpeg'), "&UserAvatar", self)
        avatarAct.setShortcut('Ctrl+U')
        avatarAct.setStatusTip('Edit your avatar')
        avatarAct.triggered.connect(self.FileOpen)

        roboAvatarAct = QAction(QIcon('img/Ubuntu.jpeg'), "&RobotAvatar", self)
        roboAvatarAct.setShortcut('Ctrl+R')
        roboAvatarAct.setStatusTip('Edit robot avatar')
        roboAvatarAct.triggered.connect(self.FileOpen2)

        bar = self.menuBar()
        themes = bar.addMenu("Themes")
        themes.addAction("dark_amber")
        themes.addAction("dark_blue")
        themes.addAction("dark_pink")
        themes.addAction("light_cyan")
        themes.addAction("light_purple")
        themes.addAction("light_lightgreen")
        themes.triggered[QAction].connect(self.onActivated)

        exitMenu = bar.addMenu("ShortCuts")
        exitMenu.addAction(exitAct)
        exitMenu.addAction(sendAct)
        exitMenu.addAction(avatarAct)
        exitMenu.addAction(roboAvatarAct)

        histoAct = QAction("History", self)  # 切换到历史记录界面
        histoAct.setStatusTip('Check History.')
        histoAct.triggered.connect(self.SwitchToHisto)
        mainAct = QAction("Main", self)  # 切换到主界面
        mainAct.setStatusTip('Switch to the main interface.')
        mainAct.triggered.connect(self.SwitchToMain)

        SwitchMenu = bar.addMenu("Interfaces")
        SwitchMenu.addAction(histoAct)
        SwitchMenu.addAction(mainAct)

        # self.main_layout.setSpacing(0)
        # self.center()
        # self.setWindowFlags(Qt.FramelessWindowHint)
        self.setGeometry(300, 300, 800, 800)

        # btn.setFont(QFont('YouYuan'))
        self.setWindowOpacity(0.9)

        self.lay = QStackedLayout()
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.lay)

        self.main_widget = QWidget()
        main_layout = QVBoxLayout()
        splitter = QSplitter(Qt.Vertical)
        # 创建按垂直方向分布的 QSplitter

        # self.main_widget.setCursor(Qt.PointingHandCursor)
        # 改变鼠标样式

        self.top_frame = QFrame(self)
        self.top_frame.setFrameShape(QFrame.StyledPanel)
        self.top_frame.resize(self.top_frame.width(), 430)

        self.bottom_frame = QFrame(self)
        self.bottom_frame.setFrameShape(QFrame.StyledPanel)
        self.bottom_frame.resize(self.top_frame.width(), 300)

        self.btn = QPushButton("SEND", self.bottom_frame)
        self.btn.resize(150, 80)
        self.btn.clicked.connect(self.DoAnim)

        # 用于用户输入
        self.input = QTextEdit(self.bottom_frame)
        self.input.setToolTip('Type to input here.')
        # self.input.isClearButtonEnabled()
        # self.input.setPlainText("Input.")

        self.audio_btn = QPushButton("Record", self.bottom_frame)
        self.audio_btn.resize(150, 80)
        self.audio_btn.clicked.connect(self.Audio_Cope)
        self.record_thread.sinOut.connect(self.Audio_Send)

        # 用于下一次按下录音按钮时是开始录音还是结束录音
        self.audio_start_flag = True

        # 用于显示录音时的动态效果
        self.audio_anim1 = QLabel(self.bottom_frame)
        self.wavegif1 = QMovie('img/siri1.gif')
        # self.wavegif1.resize(50, 50)
        self.audio_anim1.setMovie(self.wavegif1)
        self.audio_anim1.setScaledContents(True)
        self.audio_anim1.resize(148, 125)
        self.audio_anim1.setVisible(True)
        self.wavegif1.start()

        # 用于显示录音时的动态效果
        self.audio_anim2 = QLabel(self.bottom_frame)
        self.wavegif2 = QMovie('img/siri2.gif')
        # self.wavegif1.resize(50, 50)
        self.audio_anim2.setMovie(self.wavegif2)
        self.audio_anim2.setScaledContents(True)
        self.audio_anim2.resize(148, 125)
        self.audio_anim2.setVisible(False)
        self.wavegif2.start()

        self.chat_area = QWidget(self.top_frame)
        self.chat_area.resize(self.top_frame.rect().width() - 5, self.top_frame.rect().height())

        self.Labels = []
        for i in range(1000):
            new_label = QLabel("Text", self.chat_area)
            new_label.setVisible(False)
            new_label.setAlignment(Qt.AlignCenter)
            self.Labels.append(new_label)

        self.roboAvatars = []
        for i in range(1000):
            new_label = QLabel(self.chat_area)
            self.Image = QPixmap()
            self.Image.load('img/Walle.jpeg')
            self.Image = self.Image.scaled(35, 35)
            new_label.setPixmap(self.Image)
            new_label.setVisible(False)
            self.roboAvatars.append(new_label)

        self.userAvatars = []
        for i in range(1000):
            new_label = QLabel(self.chat_area)
            self.Image = QPixmap()
            self.Image.load('img/User.jpeg')
            self.Image = self.Image.scaled(35, 35)
            new_label.setPixmap(self.Image)
            new_label.setVisible(False)
            self.userAvatars.append(new_label)

        self.reply_Labels = []
        for i in range(1000):
            new_label = QLabel("Text", self.chat_area)
            new_label.setVisible(False)
            new_label.setAlignment(Qt.AlignCenter)
            self.reply_Labels.append(new_label)

        self.message_pos_y = 0

        self.scroll = QScrollArea(self.top_frame)
        self.scroll.setWidget(self.chat_area)
        self.scrollBar = self.scroll.verticalScrollBar()

        splitter.addWidget(self.top_frame)
        splitter.addWidget(self.bottom_frame)
        main_layout.addWidget(splitter)
        self.main_widget.setLayout(main_layout)
        self.setCentralWidget(self.central_widget)

        self.setWindowTitle('Chatbot')
        self.setWindowIcon(QIcon('img/Jarvis.jpeg'))

        self.histo_widget = HistoryUI()
        self.lay.addWidget(self.main_widget)
        self.lay.addWidget(self.histo_widget)

        self.message = ""
        self.show()

    def FileOpen(self):
        path, _ = QFileDialog.getOpenFileName(self, '打开文件', r'C:\Users\86138\Desktop',
                                              '图片文件 (*.jpeg)')

        # print(type(path))
        # print(path)

        for i in range(1000):
            Image = QPixmap()
            Image.load(path)
            Image = Image.scaled(35, 35)
            self.userAvatars[i].setPixmap(Image)

    def FileOpen2(self):
        path, _ = QFileDialog.getOpenFileName(self, '打开文件', r'C:\Users\86138\Desktop',
                                              '图片文件 (*.jpeg)')

        for i in range(1000):
            Image = QPixmap()
            Image.load(path)
            Image = Image.scaled(35, 35)
            self.roboAvatars[i].setPixmap(Image)
        # print(type(path))
        # print(path)

    def SwitchToHisto(self):
        '''
        width = self.rect().width()
        height = self.rect().height()
        print(width, height)
        self.histo_widget.resize(self.rect().width(), self.rect().height())
        self.histo_widget.input_widget_input.resize(self.rect().width(), self.histo_widget.input_widget_input.rect().height())
        print(self.histo_widget.input_widget_input.rect().width(), self.histo_widget.input_widget_input.rect().height())
        '''
        self.lay.setCurrentIndex(1)

    def SwitchToMain(self):
        self.lay.setCurrentIndex(0)

    # 用于切换主题
    def onActivated(self, q):
        global app
        apply_stylesheet(app, q.text() + ".xml")
        stylesheet = app.styleSheet()
        if 'light' in q.text():
            with open('light_custom.css') as file:
                app.setStyleSheet(stylesheet + file.read().format(**os.environ))
        else:
            with open('dark_custom.css') as file:
                app.setStyleSheet(stylesheet + file.read().format(**os.environ))
        self.show()

    # 发送消息的动画
    def DoAnim(self):
        self.entry_cnt += 1

        self.message = self.input.toPlainText()

        with open(self.histo_file, 'a') as f:
            wirte_in = "User: " + self.message + "\n"
            t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            f.write(str(t) + "\n")
            f.write(wirte_in)

        f.close()

        img_flag = False
        if 'time' in self.message.lower() or 'weather' in self.message.lower():
            img_flag = True

        self.Labels[self.entry_cnt - 1].setText(self.message)
        self.Labels[self.entry_cnt - 1].setVisible(True)
        self.Labels[self.entry_cnt - 1].adjustSize()
        self.Labels[self.entry_cnt - 1].resize(self.Labels[self.entry_cnt - 1].rect().width() + 25, \
                                               self.Labels[self.entry_cnt - 1].rect().height() + 25)

        # lines = self.input.document().lineCount()
        self.input.clear()

        self.anim = QPropertyAnimation(self.Labels[self.entry_cnt - 1], b"geometry")
        self.anim.setDuration(100)

        self.message_pos_y += (self.Labels[self.entry_cnt - 1].rect().height() + 10)

        if self.message_pos_y > self.chat_area.rect().height():
            self.chat_area.resize(self.chat_area.rect().width(), self.chat_area.rect().height() \
                                  + self.Labels[self.entry_cnt - 1].rect().height() + 10)

        width = self.chat_area.rect().width()
        height = self.chat_area.rect().height()

        self.userAvatars[self.entry_cnt - 1].setVisible(True)
        self.userAvatars[self.entry_cnt - 1].setGeometry(width - 45,
                                                         self.message_pos_y - self.Labels[
                                                             self.entry_cnt - 1].rect().height() - 5, \
                                                         35, 35)
        if not img_flag:
            self.anim.setStartValue(QRect(width, height, 0, 0))

            self.anim.setEndValue(QRect(width - self.Labels[self.entry_cnt - 1].rect().width() - 50, \
                                        self.message_pos_y - self.Labels[self.entry_cnt - 1].rect().height(), \
                                        self.Labels[self.entry_cnt - 1].rect().width(), \
                                        self.Labels[self.entry_cnt - 1].rect().height()))

            self.anim.start()
        else:
            self.Labels[self.entry_cnt - 1].setGeometry(width - self.Labels[self.entry_cnt - 1].rect().width() - 50, \
                                        self.message_pos_y - self.Labels[self.entry_cnt - 1].rect().height(), \
                                        self.Labels[self.entry_cnt - 1].rect().width(), \
                                        self.Labels[self.entry_cnt - 1].rect().height())
            self.Labels[self.entry_cnt - 1].setVisible(True)
        self.scrollBar.setValue(self.chat_area.rect().height())
        self.reply_thread.start()
        # print(1)
        # self.show()

    def Reply(self):

        img_flag = False # 表示机器人的回复是不是一张图片
        reply = "Image"

        if self.message != "" and search_open(self.message):
            reply = 'Opening for you'
        elif 'weather' in self.message.lower():
            print(2)
            weather()
            print(3)
            img_flag = True
            Image1 = QPixmap()
            Image1.load(r'C:\Users\86138\Desktop\Chatbot\pythonProject\weather.png')
            Image1 = Image1.scaled(200, 75)
            print(1)
            self.reply_Labels[self.entry_cnt - 1].setPixmap(Image1)
            # self.reply_Labels[self.entry_cnt - 1].setVisible(True)
            self.reply_Labels[self.entry_cnt - 1].resize(200, 75)

        elif 'time' in self.message.lower():
            mytime()
            img_flag = True
            Image1 = QPixmap()
            Image1.load(r'C:\Users\86138\Desktop\Chatbot\pythonProject\time_out.png')
            Image1 = Image1.scaled(330, 270)
            print(1)
            self.reply_Labels[self.entry_cnt - 1].setPixmap(Image1)
            self.reply_Labels[self.entry_cnt - 1].resize(330, 270)
        else:
            # reply = evaluateInput(encoder, decoder, searcher, voc, message)
            reply = getreply(self.message)
            print(reply)

        with open(self.histo_file, 'a') as f:
            wirte_in = "Robot: " + reply + "\n\n"
            f.write(wirte_in)

        f.close()

        self.message_pos_y += (self.reply_Labels[self.entry_cnt - 1].rect().height() + 10)

        if self.message_pos_y > self.chat_area.rect().height():
            self.chat_area.resize(self.chat_area.rect().width(), self.chat_area.rect().height() \
                                      + self.reply_Labels[self.entry_cnt - 1].rect().height() + 10)

        width = self.chat_area.rect().width()
        height = self.chat_area.rect().height()

        self.roboAvatars[self.entry_cnt - 1].setVisible(True)
        self.roboAvatars[self.entry_cnt - 1].setGeometry(0,
                                                         self.message_pos_y - self.reply_Labels[self.entry_cnt - 1].rect().height() - 5,
                                                         35, 35)

        if not img_flag:
            print(55)
            self.reply_Labels[self.entry_cnt - 1].setText(reply)
            self.reply_Labels[self.entry_cnt - 1].setVisible(True)
            self.reply_Labels[self.entry_cnt - 1].adjustSize()
            self.reply_Labels[self.entry_cnt - 1].resize(self.reply_Labels[self.entry_cnt - 1].rect().width() + 25, \
                                                         self.reply_Labels[self.entry_cnt - 1].rect().height() + 25)

            # 回复对话框的动画

            self.anim2 = QPropertyAnimation(self.reply_Labels[self.entry_cnt - 1], b"geometry")
            self.anim2.setDuration(100)

            self.anim2.setStartValue(QRect(25, height, 0, 0))

            self.anim2.setEndValue(QRect(40, self.message_pos_y - self.reply_Labels[self.entry_cnt - 1].rect().height(),
                                         self.reply_Labels[self.entry_cnt - 1].rect().width(),
                                         self.reply_Labels[self.entry_cnt - 1].rect().height()))

            self.anim2.start()
        else:
            self.reply_Labels[self.entry_cnt-1].move(40, self.message_pos_y - self.reply_Labels[self.entry_cnt - 1].rect().height())
            self.reply_Labels[self.entry_cnt-1].setVisible(True)
            img_flag = False

        self.show()
        # speak(reply, 200, 0.5)
        self.scrollBar.setValue(self.message_pos_y)
        self.speak_thread.reply = reply
        self.speak_thread.start()

    # 用于拉动 Frame 时按钮等控件的位置跟随
    def event(self, e):
        if e.type() in (QEvent.Show, QEvent.Resize, QEvent.CursorChange, QEvent.MouseMove):
            width = self.top_frame.rect().width()
            height = self.top_frame.rect().height()
            self.chat_area.resize(width, self.chat_area.rect().height())
            self.scroll.resize(width - 4, height - 4)
            self.scroll.move(2, 2)
            width2 = self.bottom_frame.rect().width()
            height2 = self.bottom_frame.rect().height()
            # print(height, height2)
            for i in range(self.entry_cnt):
                self.Labels[i].move(width - self.Labels[i].rect().width() - 50, self.Labels[i].pos().y())

            for i in range(self.entry_cnt):
                self.reply_Labels[i].move(40, self.reply_Labels[i].pos().y())

            for i in range(self.entry_cnt):
                self.userAvatars[i].move(width - 45, self.userAvatars[i].pos().y())

            self.btn.move(width2 - self.btn.width(), height2 - self.btn.height())
            self.audio_btn.move(width2 - self.audio_btn.width(), 0)
            self.audio_anim1.move(width2 - self.audio_btn.width(), \
                                  (self.audio_btn.height() + height2 - self.btn.height()) // 2 - 62)
            self.audio_anim2.move(width2 - self.audio_btn.width(), \
                                  (self.audio_btn.height() + height2 - self.btn.height()) // 2 - 62)
            self.input.resize(width2 - self.audio_btn.width(), height2)
            self.histo_widget.input_widget_input.resize(self.rect().width() - 250, 50)
            self.histo_widget.input_button.move(self.rect().width() - 175, 50)
            self.histo_widget.output_widget_show.resize(self.rect().width() - 100, self.rect().height() - 200)
        return QMainWindow.event(self, e)

    def Audio_Send(self):
        with open(r'C:\Users\86138\Desktop\Chatbot\temp.txt', 'r') as f:
            message = f.read()
            self.input.setText(message)
            self.DoAnim()
        f.close()

        with open(r'C:\Users\86138\Desktop\Chatbot\temp.txt', 'w') as f:
            f.write("")
        f.close()

        self.audio_anim1.setVisible(True)
        self.audio_anim2.setVisible(False)
        self.audio_btn.setText('Record')

    # 用于录音按钮按下后的动画
    def Audio_Cope(self):

        self.audio_btn.setText('Recording\n...')

        self.audio_anim1.setVisible(False)
        self.audio_anim2.setVisible(True)
        # time.sleep(0.01)

        self.record_thread.start()

        # self.audio_anim1.setVisible(True)
        # self.audio_anim2.setVisible(False)
        # self.audio_btn.setText('Record')
        '''

        if self.audio_start_flag:
            self.audio_btn.setText("Recording\n...")
            self.audio_start_flag = False
            self.audio_anim1.setVisible(False)
            self.audio_anim2.setVisible(True)

        else:
            self.audio_btn.setText("Record")
            self.audio_start_flag = True
            self.audio_anim2.setVisible(False)
            self.audio_anim1.setVisible(True)
        '''

    # 用于退出时弹出确认框
    def closeEvent(self, event):
        # 如果关闭 QMainWindow，就会产生一个QCloseEvent，
        # 并且把它传入到closeEvent函数的event参数中

        reply = QMessageBox.question(self, 'Message',
                                     'Are you sure to quit?',
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        # 创建了一个消息框，上面有俩按钮：Yes和No.
        # 第一个字符串显示在消息框的标题栏，第二个字符串显示在对话框，
        # 第三个参数是消息框的两个按钮，最后一个参数是默认按钮，这个按钮是默认选中的。
        # 返回值在变量reply里

        if reply == QMessageBox.Yes:
            '''
            localtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            year = time.strftime('%Y', time.localtime(time.time()))
            month = time.strftime('%m', time.localtime(time.time()))
            day = time.strftime('%d', time.localtime(time.time()))
            t = time.strftime('%m%d %H%M%S', time.localtime(time.time()))

            fileYear = 'C:\\Users\\Tanjf\\Desktop\\Chatbot\\History\\' + year
            fileMonth = fileYear + '/' + month
            # fileDay = fileMonth + '/' + day

            if not os.path.exists(fileYear):
                os.mkdir(fileYear)
                os.mkdir(fileMonth)
                # os.mkdir(fileDay)
            else:
                if not os.path.exists(fileMonth):
                    os.mkdir(fileMonth)
                    # os.mkdir(fileDay)
                # else:
                    # if not os.path.exists(fileDay):
                        # os.mkdir(fileDay)

            # 创建一个文件，以‘timeFile_’+具体时间为文件名称
            fileDir = fileMonth + '/timeFile_' + str(year) + '_' + str(month) + '_' + str(day) + '.txt'

            out = open(fileDir, 'w')
            for num in range(self.entry_cnt):
                text = self.Labels[num].text()
                out.write('User: ' + text)
                out.write('\n')
                text = self.reply_Labels[num].text()
                out.write('Robot: '+ text)
                out.write('\n\n')
            out.close()
            '''

            event.accept()
            # 接受退出事件，关闭组件和应用
        else:
            event.ignore()
            # 忽略关闭事件

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # PyQt5.QtGui.QFontDatabase.addApplicationFont("Sounso-Victoria.ttf")

    apply_stylesheet(app, "dark_amber.xml")
    stylesheet = app.styleSheet()
    with open('dark_custom.css') as file:
        app.setStyleSheet(stylesheet + file.read().format(**os.environ))
    ex = MainUI()
    sys.exit(app.exec_())
    # 如 果app.exec_() 运行结束，则程序退出
