import sys
import os
from glob import glob
from PySide2 import QtWidgets
from PySide2.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PySide2.QtCore import QDir, QTimer
from PySide2.QtGui import QPixmap, QImage
from ui_mainwindow import Ui_MainWindow
import cv2
from set_thresh import ChangeJDETrackerThresh
import re


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        # 打开文件类型，用于类的定义
        self.f_type = 0

    def window_init(self):
        # 设置控件属性

        # self.label.setText("目标数目")
        # self.label_2.setText("置信度")
        self.label_3.setText("输入")
        self.label_4.setText("请输入0-1范围内的阈值")
        self.label_5.setText("请打开文件")
        self.label_6.setText("阈值")
        self.pushButton.setText("Browse")
        self.pushButton_4.setText("确定")
        # 按钮使能（否）
        self.pushButton.setEnabled(True)
        # self.pushButton_2.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)

        self.pushButton.clicked.connect(input_path_init)
        self.pushButton_3.clicked.connect(show_video)
        self.pushButton_4.clicked.connect(confirm)
        # self.pushButton_3.clicked.connect(loadprogress)
        # self.pushButton_2.clicked.connect(savefile)
        # 菜单按钮 槽连接 到函数
        # self.actionOpen_image.triggered.connect(ImgConfig_init)
        # self.actionOpen_video.triggered.connect(VdoConfig_init)
        # 自适应窗口缩放
        self.label_5.setScaledContents(True)


def open_video():
    # 打开文件对话框
    file_dir, _ = QFileDialog.getOpenFileName(window.pushButton, "上传视频", QDir.currentPath(),
                                              "视频文件(*.mp4 *.avi *.wmv );;所有文件(*)")
    # 判断是否正确打开文件
    if not file_dir:
        QMessageBox.warning(window.pushButton, "警告",
                            "文件错误或打开文件失败！", QMessageBox.Yes)
        return
    if file_dir:
        window.lineEdit_3.setText(file_dir)
    window.lineEdit_3.setEnabled(False)
    # print("读入文件成功")
    # 返回视频路径
    return file_dir


def get_dir():
    file_dir = window.lineEdit_3.text()
    return file_dir


def confirm():
    window.pushButton.setEnabled(False)
    file_dir = get_dir()
    thresh_text = window.lineEdit_5.text()
    thresh = float(thresh_text)
    thresh_change = ChangeJDETrackerThresh(thresh=thresh)
    my_video = window.lineEdit_3.text()
    my_video = re.sub(r"\\", "/", my_video)
    output_path_ = re.search(r"(.*?/.+?)*/", my_video, flags=0)
    output_path = output_path_.group()
    window.label_5.setText("图像正在处理中")
    os.system(
        "CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=fairmot_final.pdparams --video_file={} --output_dir={} --save_videos".format(my_video, output_path))
    # os.system(
    #     "python tools/infer_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o use_gpu=False weights=fairmot_final.pdparams --video_file={} --output_dir={} --save_videos".format(my_video, output_path))
    window.label_5.setText("图像处理完成")
    window.pushButton_2.setEnabled(True)
    window.pushButton_2.setText("结果播放")
    window.pushButton_2.clicked.connect(show_result)


def show_result():
    my_video = window.lineEdit_3.text()
    my_video = re.sub(r"\\", "/", my_video)
    print(my_video)
    output_path_ = re.search(r"(.*?/.+?)*/", my_video, flags=0)
    output_path = output_path_.group()
    print(output_path_)
    print(output_path)
    video_name = re.sub(r"(.*?/.+?)*/", "", my_video)
    video_name = re.sub(r".mp4", "", video_name)
    print(video_name)
    output_video = output_path+"mot_outputs/"+video_name+"_vis.mp4"
    print(output_video)
    window.f_type_3 = VideoConTroller(output_video)

# 获得输入路径


class SetInputPath:
    def __init__(self):
        window.pushButton.setEnabled(True)
        # window.pushButton_2.setEnabled(False)
        window.pushButton_3.setEnabled(False)
        self.file = open_video()
        if not self.file:
            return
        window.pushButton_3.setEnabled(True)
        window.pushButton_3.setText("播放")
        window.label_5.setText("正在读取请稍后...")


class VideoConTroller:
    def __init__(self, file):
        self.file = file
        # 设置时钟
        print(file)
        self.v_timer = QTimer()
        # 读取视频
        self.cap = cv2.VideoCapture(self.file)
        if not self.cap:
            print("打开视频失败")
            return
        # 获取视频FPS
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        # 获取视频总帧数
        self.total_f = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # 获取视频当前帧所在的帧数
        self.current_f = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        # 设置定时器周期，单位毫秒
        self.v_timer.start(int(1000 / self.fps))
        print("FPS:".format(self.fps))

        window.pushButton.setEnabled(True)
        # window.pushButton_2.setEnabled(True)
        window.pushButton_3.setEnabled(True)
        window.pushButton.setText("")
        # window.pushButton_2.setText("保存")
        window.pushButton_3.setText("暂停")

        # 连接定时器周期溢出的槽函数，用于显示一帧视频
        self.v_timer.timeout.connect(self.show_pic)
        # 连接按钮和对应槽函数，lambda表达式用于传参
        window.pushButton_3.clicked.disconnect()
        window.pushButton_3.clicked.connect(self.go_pause)
        window.pushButton_2.clicked.disconnnect()
        window.pushButton_2.clicked.connect(self.go_pause)

        print("init OK")
    # 播放视频

    def show_pic(self):
        # 读取一帧
        success, frame = self.cap.read()
        if success:
            # Mat格式图像转Qt中图像的方法
            show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            showImage = QImage(
                show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            showImage = showImage.scaled(window.label_5.width() - window.label_5.lineWidth() * 2,
                                         window.label_5.height() - window.label_5.lineWidth() * 2)
            window.label_5.setScaledContents(True)
            window.label_5.setPixmap(QPixmap.fromImage(showImage))

            # 状态栏显示信息
            self.current_f = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            current_t, total_t = self.calculate_time(
                self.current_f, self.total_f, self.fps)
            window.statusbar.showMessage(
                "文件名：{}        {}({})".format(self.file, current_t, total_t))

    def go_pause(self):
        if window.pushButton_3.text() == "暂停":
            self.v_timer.stop()
            window.pushButton_3.setText("播放")
        elif window.pushButton_3.text() == "播放":
            self.v_timer.start(int(1000 / self.fps))
            window.pushButton_3.setText("暂停")

    def calculate_time(self, c_f, t_f, fps):
        total_seconds = int(t_f / fps)
        current_sec = int(c_f / fps)
        c_time = "{}:{}:{}".format(
            int(current_sec / 3600), int((current_sec % 3600) / 60), int(current_sec % 60))
        t_time = "{}:{}:{}".format(int(total_seconds / 3600), int((total_seconds % 3600) / 60),
                                   int(total_seconds % 60))
        return c_time, t_time


def input_path_init():
    window.f_type = SetInputPath()


def show_video():
    window.f_type_1 = VideoConTroller(window.lineEdit_3.text())


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.window_init()
    window.show()
    sys.exit(app.exec_())
