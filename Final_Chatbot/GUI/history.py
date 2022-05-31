import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QFile, QTextStream, QIODevice, QByteArray
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMessageBox,
                             QAction, QFileDialog, QSplitter,
                             QTextEdit)
from PyQt5.QtGui import QTextCursor, QTextCharFormat, QFont


class HistoryUI(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # 设置窗口标题
        # self.resize(800, 800)
        self.setWindowOpacity(0.9)  # 设置窗口透明度

        self.initUi()

    def initUi(self):

        # self.initMenuBar()
        self.main_widget = QtWidgets.QWidget(self)  # 创建窗口主部件
        self.main_layout = QtWidgets.QGridLayout()  # 创建主部件的网格布局
        self.main_widget.setLayout(self.main_layout)  # 设置窗口主部件布局为网格布局
        # self.setCentralWidget(self.main_widget)  # 设置窗口主部件

        # self.input_widget = QtWidgets.QWidget(self)  # 搜索框部件
        # self.input_layout = QtWidgets.QGridLayout()
        # self.input_widget.setLayout(self.input_layout)
        self.input_button = QtWidgets.QPushButton("Search", self)  # 搜索按钮

        self.input_widget_input = QtWidgets.QLineEdit(self)  # 搜索栏
        self.input_widget_input.isClearButtonEnabled()
        self.input_widget_input.setPlaceholderText("Keyword")

        # self.output_widget = QtWidgets.QWidget()
        # self.output_layout = QtWidgets.QGridLayout()
        # self.output_widget.setLayout(self.output_layout)
        self.output_widget_show = QtWidgets.QTextEdit(self)
        self.output_widget_show.setReadOnly(True)

        self.input_widget_input.setGeometry(50, 50, 550, 50)
        self.input_button.setGeometry(625, 50, 125, 50)
        self.output_widget_show.setGeometry(50, 125, 700, 600)



        '''
        self.output_layout.addWidget(self.output_widget_show, 1, 0, 8, 8)
        self.main_layout.addWidget(self.output_widget, 1, 0, 8, 8)
        '''
        self.input_button.clicked.connect(self.onFileSearch)

        '''
        self.input_layout.addWidget(self.input_widget_input, 0, 0, 1, 8)
        self.input_layout.addWidget(self.input_button, 0, 8, 1, 1)
        self.main_layout.addWidget(self.input_widget, 0, 0, 1, 8)
        '''

        # self.show()

    '''
    def initMenuBar(self):
        menuBar = self.menuBar()
        menuFile = menuBar.addMenu('open')

        # 打开文件
        actionFileOpen = QAction('打开(&O)...', self)
        actionFileOpen.setShortcut(Qt.CTRL + Qt.Key_O)
        actionFileOpen.setToolTip('打开一个文本文件')
        actionFileOpen.triggered.connect(self.onFileOpen)

        menuGoback = menuBar.addMenu('Go Back')
        actionFileOpen.setShortcut(Qt.CTRL + Qt.Key_G)
        actionGo = QAction('返回Main', self)
        # actionGo.triggered.connect(QApplication.instance().quit)

        menuExit = menuBar.addMenu('Exit')
        actionFileOpen.setShortcut(Qt.CTRL + Qt.Key_Q)
        actionExit = QAction('退出(&X)', self)
        actionExit.triggered.connect(QApplication.instance().quit)

        menuFile.addAction(actionFileOpen)
        menuFile.addSeparator()
        menuGoback.addAction(actionGo)
        menuExit.addAction(actionExit)
    '''

    def msgCritical(self, strInfo):
        dlg = QMessageBox(self)
        dlg.setIcon(QMessageBox.Critical)
        dlg.setText(strInfo)
        dlg.show()

    def onFileSearch(self):
        self.output_widget_show.clear()
        file_list = []
        count = 0
        key = self.input_widget_input.text()
        cursor = self.output_widget_show.textCursor()
        self.input_widget_input.clear()
        # print(key)
        for filepath, dirnames, filenames in os.walk(r'C:\Users\86138\Desktop\Chatbot\History'):
            for filename in filenames:
                file_list.append(os.path.join(filepath, filename))
                # print(os.path.join(filepath, filename))
        for file in file_list:
            size = os.path.getsize(file)
            if size != 0:
                f = open(file)
                if key in f.read():
                    f.close()
                    f = open(file)
                    temp = "chat history" + str(count) + "------" + file
                    temp = temp.strip('\n')
                    self.output_widget_show.append(temp)
                    # 光标移到结尾
                    cursor.movePosition(QTextCursor.End)
                    self.output_widget_show.setTextCursor(cursor)
                    # 选择当前行
                    cursor.select(QTextCursor.LineUnderCursor)
                    if cursor.block().text() == temp:
                        # 格式
                        format1 = QTextCharFormat()
                        format1.setFontWeight(QFont.Bold)
                        format1.setFontPointSize(9)
                        cursor.mergeCharFormat(format1)
                        cursor.clearSelection()
                        # 重回文本开头
                        cursor.movePosition(QTextCursor.Start)
                        self.output_widget_show.setTextCursor(cursor)

                    for line in f:
                        line = line.strip('\n')
                        # 去掉结尾的换行符，否则下方的光标无法选择文本
                        self.output_widget_show.append(line)
                        if key in line:
                            # 光标移到结尾
                            cursor.movePosition(QTextCursor.End)
                            self.output_widget_show.setTextCursor(cursor)
                            # 选择当前行
                            cursor.select(QTextCursor.LineUnderCursor)
                            # 格式
                            format2 = QTextCharFormat()
                            format2.setBackground(QtGui.QBrush(QtGui.QColor("grey")))
                            cursor.mergeCharFormat(format2)
                            cursor.clearSelection()
                            # 重回文本开头
                            cursor.movePosition(QTextCursor.Start)
                            self.output_widget_show.setTextCursor(cursor)

                    count += 1
                    self.output_widget_show.append('\n')
                    self.output_widget_show.append('\n')
                    f.close()  # 关闭文件
                else:
                    # print("no")
                    f.close()
        if count == 0:
            self.output_widget_show.append("Not found.")

    def onFileOpen(self):
        path, _ = QFileDialog.getOpenFileName(self, '打开文件', r'C:\Users\86138\Desktop\Chatbot\History', '文本文件 (*.txt)')
        if path:
            f = QFile(path)
            if not f.exists():
                self.msgCritical('目标文件不存在')
                return False
            if not f.open(QIODevice.ReadOnly | QIODevice.Text):
                self.msgCritical('打开文件失败')
                return False
            self.path = path

            self.show()
            while not f.atEnd():
                line = QByteArray(f.readLine())
                linebytes = bytes(line)
                linestr = linebytes.decode("utf-8")
                self.output_widget_show.append(linestr)
            f.close()  # 关闭文件


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HistoryUI()
    # window.show()
    sys.exit(app.exec())
