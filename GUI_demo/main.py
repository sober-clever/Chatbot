# Demo of the final GUI of the chatbot.
import sys, PyQt5, os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QToolTip,
                             QHBoxLayout, QVBoxLayout, QGridLayout, QComboBox,
                             QDesktopWidget, QLineEdit, QSplitter, QAction, QFrame,
                             QWidget, QTextBrowser, QTextEdit, QLabel, QScrollArea,
                             QMessageBox, qApp, QTextEdit)
from PyQt5.QtGui import QIcon, QFont, QFontMetrics, QPixmap, QMovie
from PyQt5.QtCore import Qt, QEvent, QPropertyAnimation, QRect
from qt_material import apply_stylesheet
import time


class MainUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    # 初始化 UI 界面
    def initUI(self):
        QToolTip.setFont(QFont('SansSerif', 20))

        self.entry_cnt = 0  # 用于记录聊天记录的总条数
        self.setFont(QFont('SanSerif', 10))
        self.setToolTip('Welcome to Chatbot <b>Jarvis</b>')

        exitAct = QAction(QIcon('Ubuntu.jpeg'), '&Exit', self)
        # QAction 是菜单栏、工具栏或者快捷键的动作的组合
        # 创建了一个图标、一个exit的标签
        exitAct.setShortcut('Ctrl+Q')
        # 创建快捷键
        exitAct.setStatusTip('Exit application.')
        # 创建状态栏，当鼠标悬停在菜单栏时，能显示当前状态
        exitAct.triggered.connect(self.close)
        # 当执行这个指定的动作时，就触发了一个事件
        # 这个事件跟 QApplication 的 quit() 行为相关联

        sendAct = QAction(QIcon('Ubuntu.jpeg'), "&Send", self)
        sendAct.setShortcut(Qt.CTRL + Qt.Key_S)
        # sendAct.setVisible(False)
        sendAct.setStatusTip('Send Message.')
        sendAct.triggered.connect(self.DoAnim)

        bar = self.menuBar()
        themes = bar.addMenu("Themes")
        themes.addAction("dark_amber")
        themes.addAction("dark_blue")
        themes.addAction("light_cyan")
        themes.addAction("light_lightgreen_500")
        themes.triggered[QAction].connect(self.onActivated)

        exitMenu = bar.addMenu("ShortCuts")
        exitMenu.addAction(exitAct)
        exitMenu.addAction(sendAct)

        # self.main_layout.setSpacing(0)
        # self.center()
        # self.setWindowFlags(Qt.FramelessWindowHint)
        self.setGeometry(300, 300, 800, 800)

        # btn.setFont(QFont('YouYuan'))
        self.setWindowOpacity(0.9)

        main_widget = QWidget()
        main_layout = QVBoxLayout()
        splitter = QSplitter(Qt.Vertical)
        # 创建按垂直方向分布的 QSplitter

        # main_widget.setCursor(Qt.PointingHandCursor)
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

        # 用于下一次按下录音按钮时是开始录音还是结束录音
        self.audio_start_flag = True

        # 用于显示录音时的动态效果
        self.audio_anim1 = QLabel(self.bottom_frame)
        self.wavegif1 = QMovie('siri1.gif')
        # self.wavegif1.resize(50, 50)
        self.audio_anim1.setMovie(self.wavegif1)
        self.audio_anim1.setScaledContents(True)
        self.audio_anim1.resize(148, 125)
        self.audio_anim1.setVisible(True)
        self.wavegif1.start()

        # 用于显示录音时的动态效果
        self.audio_anim2 = QLabel(self.bottom_frame)
        self.wavegif2 = QMovie('siri2.gif')
        # self.wavegif1.resize(50, 50)
        self.audio_anim2.setMovie(self.wavegif2)
        self.audio_anim2.setScaledContents(True)
        self.audio_anim2.resize(148, 125)
        self.audio_anim2.setVisible(False)
        self.wavegif2.start()

        self.chat_area = QWidget(self.top_frame)
        self.chat_area.resize(self.top_frame.rect().width(), self.top_frame.rect().height())

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
            self.Image.load('Jarvis.jpeg')
            self.Image = self.Image.scaled(25, 25)
            new_label.setPixmap(self.Image)
            new_label.setVisible(False)
            self.roboAvatars.append(new_label)


        self.reply_Labels = []
        for i in range(1000):
            new_label = QLabel("Text", self.chat_area)
            new_label.setVisible(False)
            new_label.setAlignment(Qt.AlignCenter)
            self.reply_Labels.append(new_label)

        self.scroll = QScrollArea(self.top_frame)
        self.scroll.setWidget(self.chat_area)
        self.scrollBar = self.scroll.verticalScrollBar()

        splitter.addWidget(self.top_frame)
        splitter.addWidget(self.bottom_frame)
        main_layout.addWidget(splitter)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.setWindowTitle('Chatbot')
        self.setWindowIcon(QIcon('Jarvis.jpeg'))
        self.show()

    # 用于切换主题
    def onActivated(self, q):
        global app
        apply_stylesheet(app, q.text()+".xml")
        stylesheet = app.styleSheet()
        with open('custom.css') as file:
            app.setStyleSheet(stylesheet + file.read().format(**os.environ))
        self.show()

    # 发送消息的动画
    def DoAnim(self):
        self.entry_cnt += 1

        self.Labels[self.entry_cnt-1].setText(self.input.toPlainText())
        self.Labels[self.entry_cnt-1].setVisible(True)
        self.Labels[self.entry_cnt-1].adjustSize()
        self.Labels[self.entry_cnt-1].resize(self.Labels[self.entry_cnt-1].rect().width()+25,\
                                             self.Labels[self.entry_cnt-1].rect().height()+25)

        # lines = self.input.document().lineCount()
        self.input.clear()

        self.anim = QPropertyAnimation(self.Labels[self.entry_cnt-1], b"geometry")
        self.anim.setDuration(100)
        self.chat_area.resize(self.chat_area.rect().width(), self.chat_area.rect().height()\
                              + self.Labels[self.entry_cnt-1].rect().height()+25)
        width = self.chat_area.rect().width()
        height = self.chat_area.rect().height()
        self.anim.setStartValue(QRect(width, height, 0, 0))

        self.anim.setEndValue(QRect(width-self.Labels[self.entry_cnt-1].rect().width()-25,\
                                    height-self.Labels[self.entry_cnt-1].rect().height(),\
                                    self.Labels[self.entry_cnt-1].rect().width(),\
                                    self.Labels[self.entry_cnt-1].rect().height()))

        self.anim.start()
        self.scrollBar.setValue(self.chat_area.rect().height())
        # print(1)
        # self.show()

        self.reply_Labels[self.entry_cnt - 1].setText("Testing...")
        self.reply_Labels[self.entry_cnt - 1].setVisible(True)
        self.reply_Labels[self.entry_cnt - 1].adjustSize()
        self.reply_Labels[self.entry_cnt - 1].resize(self.reply_Labels[self.entry_cnt - 1].rect().width() + 25, \
                                                     self.reply_Labels[self.entry_cnt - 1].rect().height() + 25)

        # time.sleep(0.5)
        self.anim2 = QPropertyAnimation(self.reply_Labels[self.entry_cnt - 1], b"geometry")
        self.anim2.setDuration(100)
        self.chat_area.resize(self.chat_area.rect().width(), self.chat_area.rect().height() \
                              + self.reply_Labels[self.entry_cnt - 1].rect().height() + 25)

        width = self.chat_area.rect().width()
        height = self.chat_area.rect().height()

        self.roboAvatars[self.entry_cnt - 1].setVisible(True)
        self.roboAvatars[self.entry_cnt - 1].setGeometry(0,
                                                         height - self.reply_Labels[self.entry_cnt - 1].rect().height()-5,\
                                                         20, 20)
        self.anim2.setStartValue(QRect(25, height, 0, 0))

        self.anim2.setEndValue(QRect(25, height - self.reply_Labels[self.entry_cnt - 1].rect().height(), \
                                    self.reply_Labels[self.entry_cnt - 1].rect().width(), \
                                    self.reply_Labels[self.entry_cnt - 1].rect().height()))

        self.anim2.start()
        self.scrollBar.setValue(self.chat_area.rect().height())

    # 用于拉动 Frame 时按钮等控件的位置跟随
    def event(self, e):
        if e.type() in (QEvent.Show, QEvent.Resize, QEvent.CursorChange, QEvent.MouseMove):
            width = self.top_frame.rect().width()
            height = self.top_frame.rect().height()
            self.chat_area.resize(width, self.chat_area.rect().height())
            self.scroll.resize(width-4, height-4)
            self.scroll.move(2, 2)
            width2 = self.bottom_frame.rect().width()
            height2 = self.bottom_frame.rect().height()
            print(height, height2)
            for i in range(self.entry_cnt):
                self.Labels[i].move(width-self.Labels[i].rect().width()-25, self.Labels[i].pos().y())

            for i in range(self.entry_cnt):
                self.reply_Labels[i].move(25, self.reply_Labels[i].pos().y())

            self.btn.move(width2-self.btn.width(), height2-self.btn.height())
            self.audio_btn.move(width2 - self.audio_btn.width(), 0)
            self.audio_anim1.move(width2 - self.audio_btn.width(), \
                                  (self.audio_btn.height()+height2-self.btn.height())//2-62)
            self.audio_anim2.move(width2 - self.audio_btn.width(), \
                                 (self.audio_btn.height()+height2-self.btn.height())//2-62)
            self.input.resize(width2 - self.audio_btn.width(), height2)
        return QMainWindow.event(self, e)

    # 用于录音按钮按下后的动画
    def Audio_Cope(self):
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
    with open('custom.css') as file:
        app.setStyleSheet(stylesheet + file.read().format(**os.environ))
    ex = MainUI()
    sys.exit(app.exec_())
    # 如 果app.exec_() 运行结束，则程序退出
