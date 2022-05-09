# PyQt

### 下载

[Anaconda+Pycharm+PyQt安装教程 - 简书 (jianshu.com)](https://www.jianshu.com/p/8b992e47a0e4)

注：如果不使用 conda，而是使用本地的 python，那么创建 Qt Designer 那一步，designed.exe 的路径不对，大概是在这个位置 `python\Lib\site-packages\qt5_applications\Qt\bin`



### 控件（Widget）

#### QApplication

这个函数提供了整个图形界面程序的底层管理功能，比如：初始化、程序入口参数的处理、用户事件分发给各个对应的控件等等。

在任何界面控件对象创建前，先运行这个函数



#### QMainWindow、QPlainTextEdit、QPushButton

分别是界面的主窗口、文本框和按钮，都是控件基类对象 QWidget 的子类。

要在界面上创建一个控件（widget），就需要在程序代码中创建这个控件对应类的一个实例对象。



#### 控件的操作

##### 在 Qt 系统中，控件是**层层嵌套**的，除了最顶层的控件，其他的控件都有父控件。

QPlainTextEdit、QPushButton 实例化时，都有一个参数window，如下

```python
QPlainTextEdit(window)
QPushButton('统计', window)
```

就是指定它的父控件对象 是 window 对应的QMainWindow 主窗口。而实例化 QMainWindow 主窗口时，却没有指定 父控件， 因为它就是最上层的控件了

##### 控件对象的 move 方法决定了这个控件显示的位置

`window.move(300, 310)` 

决定了主窗口的左上角坐标在相对屏幕的左上角的 X 横坐标 300 像素，Y 纵坐标 310 像素这个位置。

`textEdit.move(10,25)` 

决定了文本框的左上角坐标在相对父窗口的左上角的 X 横坐标 10 像素,  Y 纵坐标 25 像素这个位置。

##### 控件对象的 resize 方法决定了这个控件显示的大小。

`window.resize(500, 400)` 

就决定了 主窗口的宽度为500像素，高度为400像素。

`textEdit.resize(300,350)` 

决定了文本框的 宽度为300像素，高度为350像素。

##### 注

放在主窗口的控件，要能全部显示在界面上， 必须加上下面这行代码

```py
window.show()
```

最后 ，通过下面这行代码

```py
app.exec_()
```

进入QApplication的事件处理循环，接收用户的输入事件（），并且分配给相应的对象去处理。



### 界面动作处理（signal & slot）

在 Qt 系统中， 当界面上一个控件被操作时，比如被点击、被输入文本、被鼠标拖拽等， 就会发出信号 **signal**。就是表明一个事件（比如被点击、被输入文本）发生了。

我们可以预先在代码中指定处理这个 signal 的函数，这个函数叫做 **slot**

 比如，我们可以像下面这样定义一个函数

```python
def handleCalc():
    print('按钮被点击了')
```

然后， 如果发生了制定 button 被按下的事情，需要让 `handleCalc` 来处理，像这样

```python
button.clicked.connect(handleCalc)
```

用 QT 的术语来解释上面这行代码，就是：把 button 被点击（clicked）的信号（signal），连接（connect）到了 handleCalc 这样的一个 slot 上



#### 封装到类中

通常应该把一个窗口和其包含的控件，对应的代码全部封装到类中，方便阅读，并且防止全局变量过多造成冲突

如：

```python
from PySide2.QtWidgets import QApplication,
from PySide2.QtWidgets import QMainWindow, QPushButton,  QPlainTextEdit,QMessageBox

class Stats():
    def __init__(self):
        self.window = QMainWindow()
        self.window.resize(500, 400)
        self.window.move(300, 300)
        self.window.setWindowTitle('薪资统计')

        self.textEdit = QPlainTextEdit(self.window)
        self.textEdit.setPlaceholderText("请输入薪资表")
        self.textEdit.move(10, 25)
        self.textEdit.resize(300, 350)

        self.button = QPushButton('统计', self.window)
        self.button.move(380, 80)

        self.button.clicked.connect(self.handleCalc)


    def handleCalc(self):
        info = self.textEdit.toPlainText()

        # 薪资20000 以上 和 以下 的人员名单
        salary_above_20k = ''
        salary_below_20k = ''
        for line in info.splitlines():
            if not line.strip():
                continue
            parts = line.split(' ')
            # 去掉列表中的空字符串内容
            parts = [p for p in parts if p]
            name,salary,age = parts
            if int(salary) >= 20000:
                salary_above_20k += name + '\n'
            else:
                salary_below_20k += name + '\n'

        QMessageBox.about(self.window,
                    '统计结果',
                    f'''薪资20000 以上的有：\n{salary_above_20k}
                    \n薪资20000 以下的有：\n{salary_below_20k}'''
                    )

app = QApplication([])
stats = Stats()
stats.window.show()
app.exec_()
```



### 界面设计师（Qt Designer）

Qt Designer，即为 Qt 界面生成器。通过 Qt Designer ，可实现以拖拽的方式设计界面，最终生成一个 ui 文件



#### 动态加载 UI 文件

使 python 程序直接从 ui 文件中加载 UI 定义，并且动态创建一个相应的窗口对象。

代码如下：

```python
from PyQt5 import uic

class Stats:
    def __init__(self):
        # 从文件中加载UI定义
        self.ui = uic.loadUi("main.ui")
```



#### 转化 UI 文件为 python 代码

执行如下命令：`pyuic5 main.ui > ui_main.py`

然后可在代码文件中使用界面：

```python
from ui_main import Ui_MainWindow

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        # 使用ui文件导入定义界面类
        self.ui = Ui_MainWindow()
        # 初始化界面
        self.ui.setupUi(self)

app = QApplication([])
mainw = MainWindow()
mainw.show()
app.exec_()
```



#### 界面布局的推荐步骤

首先添加各类控件，之后根据控件的大小、位置需要，从小到大、从局部到整体添加 Layout，再调整各类属性，达到预期要求

调整 layout 中控件的**大小比例**，优先使用 layout 的 layoutStrentch 属性来控制



#### 页面布局 Layout

+ QHBoxLayout 水平布局
+ QVBoxLayout  垂直布局
+ QGridLayout   表格布局
+ QFormLayout 表单布局

增加页面布局后，可以保证控件可以随着主窗口的大小变化而变化



#### 调整控件位置和大小

+ 在页面布局后，控件不可直接通过拉伸的方式调整大小，决定控件大小的属性是 `sizePolocy` ，可修改为 fixed

+ 而在已经布局好的页面中，单一控件也无法通过直接拖拽的方式调整位置，比如居中，此时需要通过再加布局的方式。而单一部件无法直接添加布局，此时需要先添加一个空的布局，然后将控件拖拽到“对象查看器”中，新添加入的布局，即可实现居中
+ 调整一个布局中的不同空间之间大小的比例：修改相应控件属性的伸展量
+ 或者更灵活的调整控件大小的方法：为单个控件添加布局，修改布局的 Margin。垂直布局会保证其中的单个控件会垂直居中，水平控件会保证其中的控件会水平居中
+ 增加空间之间的距离：增加 Spacers



### 显示样式

Qt 中定义界面显示样式的方法，称为 Qt Style Sheet，简称 QSS

例如设置一个 button 的属性：

```python
QPushButton { 
    color: red ;
    font-size:15px;
}
```

这部分的语法，由 selector 和 declaration 组成

 `QPushButton` 称之为 selector，大括号之内的，称之为 Properties （样式属性）



#### selector 选择器

常见语法

| Selector            | 示例                        | 说明                                               |
| ------------------- | --------------------------- | -------------------------------------------------- |
| Universal Selector  | `*`                         | 星号匹配所有的界面元素                             |
| Type Selector       | `QPushButton`               | 选择所有 QPushButton 类型 （包括其子类）           |
| Class Selector      | `.QPushButton`              | 选择所有 QPushButton 类型 ，但是不包括其子类       |
| ID Selector         | `QPushButton#okButton`      | 选择所有 `对象名为 okButton` 的QPushButton 类型    |
| Property Selector   | `QPushButton[flat="false"]` | 选择所有 flat 属性值为 false 的 QPushButton 类型。 |
| Descendant Selector | `QDialog QPushButton`       | 选择所有 QDialog `内部` QPushButton 类型           |
| Child Selector      | `QDialog > QPushButton`     | 选择所有 QDialog `直接子节点` QPushButton 类型     |



#### 样式属性

##### 背景

可以指定某些元素的背景色，像这样

```css
QTextEdit { background-color: yellow }
```

颜色可以使用红绿蓝数字，像这样

```css
QTextEdit { background-color: #e7d8d8 }
```

也可以像这样指定背景图片

```css
QTextEdit {
    background-image: url(gg03.png);
}
```

##### 边框

可以像这样指定边框 `border:1px solid #1d649c;`

其中

`1px` 是边框宽度

`solid` 是边框线为实线， 也可以是 `dashed`(虚线) 和 `dotted`（点）

比如

```css
*[myclass=bar2btn]:hover{
	border:1px solid #1d649c;
}
```

边框可以指定为无边框 `border:none`

##### 字体、大小、颜色

可以这样指定元素的 文字字体、大小、颜色

```css
*{	
	font-family:微软雅黑;
	font-size:15px;
	color: #1d649c;
}
```

##### 宽度、高度

可以这样指定元素的 宽度、高度

```css
QPushButton {	
	width:50px;
	height:20px;
}
```



