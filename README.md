初次运行需要安装环境

在你的python环境中

pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple

注意parl必须是1.3.1，pygame必须是1.9.6

*原因：parl后期改了读model的接口，pygame则改了事件部分的接口*

运行完双击run.bat则可以开始游戏

算法控制的为红色paddle

用户控制的为蓝色paddle



## 控制

按空格键，则跳起一次

按ESC键，重新开始游戏



## 修改运行框的大小

打开run.bat

在最后一行

```
call python pvsc.py 1640（宽） 1200（高）
```

里的两个数字改成你需要的宽、高后，再次运行



由于使用opencv放大了窗口，所以需要点一下原来的pygam窗口，让它能够监听键盘事件。

