# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tinf_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 156)
        Dialog.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout_3.addWidget(self.label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_2 = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.toptLabel = QtWidgets.QLabel(Dialog)
        self.toptLabel.setObjectName("toptLabel")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.toptLabel)
        self.topt1 = QtWidgets.QLineEdit(Dialog)
        self.topt1.setReadOnly(True)
        self.topt1.setObjectName("topt1")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.topt1)
        self.tinfLabel = QtWidgets.QLabel(Dialog)
        self.tinfLabel.setObjectName("tinfLabel")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.tinfLabel)
        self.tinf1 = QtWidgets.QLineEdit(Dialog)
        self.tinf1.setReadOnly(True)
        self.tinf1.setObjectName("tinf1")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.tinf1)
        self.verticalLayout_2.addLayout(self.formLayout)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_3 = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setObjectName("formLayout_2")
        self.toptLabel_2 = QtWidgets.QLabel(Dialog)
        self.toptLabel_2.setObjectName("toptLabel_2")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.toptLabel_2)
        self.topt15 = QtWidgets.QLineEdit(Dialog)
        self.topt15.setReadOnly(True)
        self.topt15.setObjectName("topt15")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.topt15)
        self.tinfLabel_2 = QtWidgets.QLabel(Dialog)
        self.tinfLabel_2.setObjectName("tinfLabel_2")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.tinfLabel_2)
        self.tinf15 = QtWidgets.QLineEdit(Dialog)
        self.tinf15.setReadOnly(True)
        self.tinf15.setObjectName("tinf15")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.tinf15)
        self.verticalLayout.addLayout(self.formLayout_2)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        spacerItem = QtWidgets.QSpacerItem(20, 0, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_3.addItem(spacerItem)
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setMinimumSize(QtCore.QSize(128, 32))
        self.pushButton.setStyleSheet("QPushButton\n"
"{\n"
"    border:1px solid rgb(182, 182, 182);\n"
"    color: rgb(0,0,0);\n"
"}\n"
"QPushButton:hover\n"
"{\n"
"    border:1px solid rgb(150, 150, 150);\n"
"    color: rgb(0,0,0);\n"
"    background-color: rgba(180,180,180,80);\n"
"}\n"
"")
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_3.addWidget(self.pushButton)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Dataset:"))
        self.label_2.setText(_translate("Dialog", "MMRT 1.0:"))
        self.toptLabel.setText(_translate("Dialog", "Topt:"))
        self.tinfLabel.setText(_translate("Dialog", "Tinf"))
        self.label_3.setText(_translate("Dialog", "MMRT 1.5:"))
        self.toptLabel_2.setText(_translate("Dialog", "Topt:"))
        self.tinfLabel_2.setText(_translate("Dialog", "Tinf"))
        self.pushButton.setText(_translate("Dialog", "OK"))