from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import os
import cv2
import random
import numpy as np
import CIFAR10.Cifar10_Classification as CIFAR10_Classification
import matplotlib.pyplot as plt


class Ui_Cifar10Classification(object):
    def setupUi(self, Cifar10Classification):
        Cifar10Classification.setObjectName("Cifar10Classification")
        Cifar10Classification.setWindowModality(QtCore.Qt.NonModal)
        Cifar10Classification.resize(435, 346)
        Cifar10Classification.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Cifar10Classification.setWindowIcon(icon)
        Cifar10Classification.setIconSize(QtCore.QSize(250, 250))
        self.centralwidget = QtWidgets.QWidget(Cifar10Classification)
        self.centralwidget.setObjectName("centralwidget")
        self.Title = QtWidgets.QLabel(self.centralwidget)
        self.Title.setGeometry(QtCore.QRect(130, 20, 181, 21))
        self.Title.setObjectName("Title")
        self.TrainButton = QtWidgets.QPushButton(self.centralwidget)
        self.TrainButton.setGeometry(QtCore.QRect(20, 100, 81, 41))
        self.TrainButton.setObjectName("TrainButton")
        self.LoadParamsButton = QtWidgets.QPushButton(self.centralwidget)
        self.LoadParamsButton.setGeometry(QtCore.QRect(20, 170, 91, 41))
        self.LoadParamsButton.setObjectName("LoadParamsButton")
        self.TestButton = QtWidgets.QPushButton(self.centralwidget)
        self.TestButton.setGeometry(QtCore.QRect(20, 240, 81, 41))
        self.TestButton.setObjectName("TestButton")
        self.TrueLabel = QtWidgets.QLabel(self.centralwidget)
        self.TrueLabel.setGeometry(QtCore.QRect(170, 90, 131, 31))
        self.TrueLabel.setObjectName("TrueLabel")
        self.PredLabel = QtWidgets.QLabel(self.centralwidget)
        self.PredLabel.setGeometry(QtCore.QRect(170, 150, 131, 31))
        self.PredLabel.setObjectName("PredLabel")
        self.GetImageButton = QtWidgets.QPushButton(self.centralwidget)
        self.GetImageButton.setGeometry(QtCore.QRect(160, 240, 61, 41))
        self.GetImageButton.setObjectName("GetImageButton")
        self.PredictButton = QtWidgets.QPushButton(self.centralwidget)
        self.PredictButton.setGeometry(QtCore.QRect(240, 240, 61, 41))
        self.PredictButton.setObjectName("PredictButton")
        self.ImageLabel = QtWidgets.QLabel(self.centralwidget)
        self.ImageLabel.setGeometry(QtCore.QRect(320, 90, 96, 96))
        self.ImageLabel.setAutoFillBackground(True)
        self.ImageLabel.setText("")
        self.ImageLabel.setObjectName("ImageLabel")
        Cifar10Classification.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(Cifar10Classification)
        self.statusbar.setObjectName("statusbar")
        Cifar10Classification.setStatusBar(self.statusbar)

        self.retranslateUi(Cifar10Classification)
        QtCore.QMetaObject.connectSlotsByName(Cifar10Classification)
        # 自写代码
        self.flag1 = 0  # 判断是否有模型可用
        self.flag2 = 0  # 判断是否加载图片
        self.text_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']
        self.TrainButton.clicked.connect(self.TrainButton_Click)
        self.LoadParamsButton.clicked.connect(self.LoadParamsButton_Click)
        self.TestButton.clicked.connect(self.TestButton_Click)
        self.GetImageButton.clicked.connect(self.GetImageButton_Click)
        self.PredictButton.clicked.connect(self.PredictButton_Click)

    def TrainButton_Click(self):
        self.net = CIFAR10_Classification.Train(load_Params=False)
        self.flag1 = 1

    def LoadParamsButton_Click(self):
        self.net = CIFAR10_Classification.Train(load_Params=True)
        self.flag1 = 1
        print("模型已加载!!\n模型已加载!!\n模型已加载!!\n模型已加载!!\n模型已加载!!\n")

    def TestButton_Click(self):
        if self.flag1 == 0:
            print("请先训练好模型!\n请先训练好模型!\n请先训练好模型!\n请先训练好模型!\n请先训练好模型!")
        else:
            CIFAR10_Classification.Test(self.net)

    def GetImageButton_Click(self):
        self.image, y_index = CIFAR10_Classification.GetImage()
        self.TrueLabel.setText('True: ' + self.text_labels[y_index])
        self.flag2 = 1
        _, figs = plt.subplots(1, 1, figsize=(0.96, 0.96))
        plt.axis('off')
        figs.imshow(np.transpose(self.image.asnumpy(), [1, 2, 0]))
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig('temp.png')
        pix = QPixmap('temp.png')
        self.ImageLabel.setPixmap(pix)

    def PredictButton_Click(self):
        if self.flag1 == 0:
            print("请先训练好模型!\n请先训练好模型!\n请先训练好模型!\n请先训练好模型!\n请先训练好模型!")
        elif self.flag2 == 0:
            print("请先获取图片\n请先获取图片\n请先获取图片\n请先获取图片\n请先获取图片\n")
        else:
            y_pred = CIFAR10_Classification.Predict(self.net, self.image)
            self.PredLabel.setText("Pred: " + self.text_labels[int(y_pred)])

    # end
    def retranslateUi(self, Cifar10Classification):
        _translate = QtCore.QCoreApplication.translate
        Cifar10Classification.setWindowTitle(_translate("Cifar10Classification", "Cifar10Classification v1.0 --by Chino"))
        self.Title.setText(_translate("Cifar10Classification", "基于深度学习的Cifar10分类任务"))
        self.TrainButton.setText(_translate("Cifar10Classification", "训练模型"))
        self.LoadParamsButton.setText(_translate("Cifar10Classification", "加载已训练参数"))
        self.TestButton.setText(_translate("Cifar10Classification", "测试模型"))
        self.TrueLabel.setText(_translate("Cifar10Classification", "待定"))
        self.PredLabel.setText(_translate("Cifar10Classification", "待定"))
        self.GetImageButton.setText(_translate("Cifar10Classification", "获取图片"))
        self.PredictButton.setText(_translate("Cifar10Classification", "预测"))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_Cifar10Classification()
    ui.setupUi(MainWindow)
    MainWindow.show()
sys.exit(app.exec_())