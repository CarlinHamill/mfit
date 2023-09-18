# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mmrt1_parameter_value_display_widget.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(223, 236)
        Form.setStyleSheet("QWidget\n"
"{\n"
"    background-color: rgb(255, 255, 255);\n"
"}\n"
"QLineEdit\n"
"{\n"
"    border:1px solid rgb(182, 182, 182);\n"
"}\n"
"\n"
"QWidget#widget\n"
"{\n"
"    border:1px solid rgb(150, 150, 150);\n"
"}\n"
"QWidget#widget_2\n"
"{\n"
"    border:1px solid rgb(150, 150, 150);\n"
"}\n"
"QWidget#widget_3\n"
"{\n"
"    border:1px solid rgb(150, 150, 150);\n"
"}\n"
"QWidget#widget_4\n"
"{\n"
"    border:1px solid rgb(150, 150, 150);\n"
"}\n"
"QWidget#widget_5\n"
"{\n"
"    border:1px solid rgb(150, 150, 150);\n"
"}\n"
"QWidget#widget_6\n"
"{\n"
"    border:1px solid rgb(150, 150, 150);\n"
"}\n"
"QWidget#widget_7\n"
"{\n"
"    border:1px solid rgb(150, 150, 150);\n"
"}")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setSpacing(2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.widget_8 = QtWidgets.QWidget(Form)
        self.widget_8.setObjectName("widget_8")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.widget_8)
        self.horizontalLayout_8.setContentsMargins(3, 0, 3, 0)
        self.horizontalLayout_8.setSpacing(3)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_11 = QtWidgets.QLabel(self.widget_8)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy)
        self.label_11.setMinimumSize(QtCore.QSize(140, 0))
        self.label_11.setMaximumSize(QtCore.QSize(140, 16777215))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_11.setFont(font)
        self.label_11.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeading|QtCore.Qt.AlignmentFlag.AlignLeft|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_8.addWidget(self.label_11)
        self.label_12 = QtWidgets.QLabel(self.widget_8)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_8.addWidget(self.label_12)
        self.verticalLayout_3.addWidget(self.widget_8)
        self.widget = QtWidgets.QWidget(Form)
        self.widget.setObjectName("widget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_2.setContentsMargins(3, 3, 3, 3)
        self.horizontalLayout_2.setSpacing(3)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.enthalpyLabel = QtWidgets.QLabel(self.widget)
        self.enthalpyLabel.setMinimumSize(QtCore.QSize(140, 0))
        self.enthalpyLabel.setMaximumSize(QtCore.QSize(140, 16777215))
        self.enthalpyLabel.setObjectName("enthalpyLabel")
        self.verticalLayout_11.addWidget(self.enthalpyLabel)
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_11.addWidget(self.label_4)
        self.horizontalLayout_2.addLayout(self.verticalLayout_11)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.enthalpyLineEdit = QtWidgets.QLineEdit(self.widget)
        self.enthalpyLineEdit.setMinimumSize(QtCore.QSize(70, 0))
        self.enthalpyLineEdit.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.enthalpyLineEdit.setText("")
        self.enthalpyLineEdit.setReadOnly(True)
        self.enthalpyLineEdit.setObjectName("enthalpyLineEdit")
        self.verticalLayout_7.addWidget(self.enthalpyLineEdit)
        self.lineEdit = QtWidgets.QLineEdit(self.widget)
        self.lineEdit.setMinimumSize(QtCore.QSize(70, 0))
        self.lineEdit.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.lineEdit.setText("")
        self.lineEdit.setReadOnly(True)
        self.lineEdit.setObjectName("lineEdit")
        self.verticalLayout_7.addWidget(self.lineEdit)
        self.horizontalLayout_2.addLayout(self.verticalLayout_7)
        self.verticalLayout_3.addWidget(self.widget)
        self.widget_2 = QtWidgets.QWidget(Form)
        self.widget_2.setObjectName("widget_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget_2)
        self.horizontalLayout_3.setContentsMargins(3, 3, 3, 3)
        self.horizontalLayout_3.setSpacing(3)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout()
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.entropyLabel = QtWidgets.QLabel(self.widget_2)
        self.entropyLabel.setMinimumSize(QtCore.QSize(140, 0))
        self.entropyLabel.setMaximumSize(QtCore.QSize(140, 16777215))
        self.entropyLabel.setObjectName("entropyLabel")
        self.verticalLayout_12.addWidget(self.entropyLabel)
        self.label_5 = QtWidgets.QLabel(self.widget_2)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_12.addWidget(self.label_5)
        self.horizontalLayout_3.addLayout(self.verticalLayout_12)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.entropyLineEdit = QtWidgets.QLineEdit(self.widget_2)
        self.entropyLineEdit.setMinimumSize(QtCore.QSize(70, 0))
        self.entropyLineEdit.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.entropyLineEdit.setText("")
        self.entropyLineEdit.setReadOnly(True)
        self.entropyLineEdit.setObjectName("entropyLineEdit")
        self.verticalLayout_6.addWidget(self.entropyLineEdit)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.widget_2)
        self.lineEdit_2.setMinimumSize(QtCore.QSize(70, 0))
        self.lineEdit_2.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.lineEdit_2.setText("")
        self.lineEdit_2.setReadOnly(True)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.verticalLayout_6.addWidget(self.lineEdit_2)
        self.horizontalLayout_3.addLayout(self.verticalLayout_6)
        self.verticalLayout_3.addWidget(self.widget_2)
        self.widget_6 = QtWidgets.QWidget(Form)
        self.widget_6.setObjectName("widget_6")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.widget_6)
        self.horizontalLayout_5.setContentsMargins(3, 3, 3, 3)
        self.horizontalLayout_5.setSpacing(3)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout()
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.heatCapacityUpperLabel = QtWidgets.QLabel(self.widget_6)
        self.heatCapacityUpperLabel.setMinimumSize(QtCore.QSize(140, 0))
        self.heatCapacityUpperLabel.setMaximumSize(QtCore.QSize(140, 16777215))
        self.heatCapacityUpperLabel.setObjectName("heatCapacityUpperLabel")
        self.verticalLayout_13.addWidget(self.heatCapacityUpperLabel)
        self.label_7 = QtWidgets.QLabel(self.widget_6)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_13.addWidget(self.label_7)
        self.horizontalLayout_5.addLayout(self.verticalLayout_13)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.heatCapacityUpperLineEdit = QtWidgets.QLineEdit(self.widget_6)
        self.heatCapacityUpperLineEdit.setMinimumSize(QtCore.QSize(70, 0))
        self.heatCapacityUpperLineEdit.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.heatCapacityUpperLineEdit.setText("")
        self.heatCapacityUpperLineEdit.setReadOnly(True)
        self.heatCapacityUpperLineEdit.setObjectName("heatCapacityUpperLineEdit")
        self.verticalLayout_2.addWidget(self.heatCapacityUpperLineEdit)
        self.lineEdit_6 = QtWidgets.QLineEdit(self.widget_6)
        self.lineEdit_6.setMinimumSize(QtCore.QSize(70, 0))
        self.lineEdit_6.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.lineEdit_6.setText("")
        self.lineEdit_6.setReadOnly(True)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.verticalLayout_2.addWidget(self.lineEdit_6)
        self.horizontalLayout_5.addLayout(self.verticalLayout_2)
        self.verticalLayout_3.addWidget(self.widget_6)
        self.widget_7 = QtWidgets.QWidget(Form)
        self.widget_7.setObjectName("widget_7")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget_7)
        self.horizontalLayout.setContentsMargins(3, 3, 3, 3)
        self.horizontalLayout.setSpacing(3)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setSpacing(3)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.label_9 = QtWidgets.QLabel(self.widget_7)
        self.label_9.setMinimumSize(QtCore.QSize(140, 0))
        self.label_9.setMaximumSize(QtCore.QSize(140, 16777215))
        self.label_9.setObjectName("label_9")
        self.verticalLayout_8.addWidget(self.label_9)
        self.label = QtWidgets.QLabel(self.widget_7)
        self.label.setObjectName("label")
        self.verticalLayout_8.addWidget(self.label)
        self.horizontalLayout.addLayout(self.verticalLayout_8)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.tnLineEdit = QtWidgets.QLineEdit(self.widget_7)
        self.tnLineEdit.setMinimumSize(QtCore.QSize(70, 0))
        self.tnLineEdit.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.tnLineEdit.setText("")
        self.tnLineEdit.setReadOnly(True)
        self.tnLineEdit.setObjectName("tnLineEdit")
        self.verticalLayout.addWidget(self.tnLineEdit)
        self.lineEdit_7 = QtWidgets.QLineEdit(self.widget_7)
        self.lineEdit_7.setMinimumSize(QtCore.QSize(70, 0))
        self.lineEdit_7.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.lineEdit_7.setText("")
        self.lineEdit_7.setReadOnly(True)
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.verticalLayout.addWidget(self.lineEdit_7)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_3.addWidget(self.widget_7)
        self.enthalpyLabel.setBuddy(self.enthalpyLineEdit)
        self.entropyLabel.setBuddy(self.entropyLineEdit)
        self.heatCapacityUpperLabel.setBuddy(self.heatCapacityUpperLineEdit)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_11.setText(_translate("Form", "MMRT 1.0 Parameters:"))
        self.label_12.setText(_translate("Form", "Value:\n"
"Error:"))
        self.enthalpyLabel.setText(_translate("Form", "Enthalpy (&#8710;H<sub>T&#8320;</sub><sup>&Dagger;</sup>):"))
        self.label_4.setText(_translate("Form", "(J mol<sup>-1</sup>)"))
        self.enthalpyLineEdit.setStatusTip(_translate("Form", "Parameter fit value"))
        self.lineEdit.setStatusTip(_translate("Form", "Parameter fit error"))
        self.entropyLabel.setText(_translate("Form", "Entropy (&#8710;S<sub>T&#8320;</sub><sup>&Dagger;</sup>):"))
        self.label_5.setText(_translate("Form", "(J mol<sup>-1</sup> K<sup>-1</sup>)"))
        self.entropyLineEdit.setStatusTip(_translate("Form", "Parameter fit value"))
        self.lineEdit_2.setStatusTip(_translate("Form", "Parameter fit error"))
        self.heatCapacityUpperLabel.setText(_translate("Form", "Heat Capacity (&#8710;C<sub>P,T&#8320;</sub><sup>&Dagger;</sup>):"))
        self.label_7.setText(_translate("Form", "(J mol<sup>-1</sup> K<sup>-1</sup>)"))
        self.heatCapacityUpperLineEdit.setStatusTip(_translate("Form", "Parameter fit value"))
        self.lineEdit_6.setStatusTip(_translate("Form", "Parameter fit error"))
        self.label_9.setText(_translate("Form", "Reference Temperature (T<sub>0</sub>):"))
        self.label.setText(_translate("Form", "(K)"))
        self.tnLineEdit.setStatusTip(_translate("Form", "Parameter fit value"))
        self.lineEdit_7.setStatusTip(_translate("Form", "Parameter fit error"))
