# Form implementation generated from reading ui file 'application_base.ui'
#
# Created by: PyQt6 UI code generator 6.5.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 535)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"")
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(parent=self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuExport = QtWidgets.QMenu(parent=self.menubar)
        self.menuExport.setObjectName("menuExport")
        self.menuHelp = QtWidgets.QMenu(parent=self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.menuTools = QtWidgets.QMenu(parent=self.menubar)
        self.menuTools.setObjectName("menuTools")
        self.menuRate_Type = QtWidgets.QMenu(parent=self.menuTools)
        self.menuRate_Type.setObjectName("menuRate_Type")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionExit = QtGui.QAction(parent=MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/data/data/cross.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionExit.setIcon(icon)
        self.actionExit.setObjectName("actionExit")
        self.actionEdit_Parameters = QtGui.QAction(parent=MainWindow)
        self.actionEdit_Parameters.setObjectName("actionEdit_Parameters")
        self.actionChange_input_units = QtGui.QAction(parent=MainWindow)
        self.actionChange_input_units.setObjectName("actionChange_input_units")
        self.actionEdit_Dataset_Names = QtGui.QAction(parent=MainWindow)
        self.actionEdit_Dataset_Names.setObjectName("actionEdit_Dataset_Names")
        self.actionSession = QtGui.QAction(parent=MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/data/data/disk.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionSession.setIcon(icon1)
        self.actionSession.setObjectName("actionSession")
        self.actionGraphs = QtGui.QAction(parent=MainWindow)
        self.actionGraphs.setObjectName("actionGraphs")
        self.actionData = QtGui.QAction(parent=MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/data/data/folder_go.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionData.setIcon(icon2)
        self.actionData.setObjectName("actionData")
        self.actionAll = QtGui.QAction(parent=MainWindow)
        self.actionAll.setObjectName("actionAll")
        self.actionRestart = QtGui.QAction(parent=MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/data/data/arrow-circle-225.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionRestart.setIcon(icon3)
        self.actionRestart.setObjectName("actionRestart")
        self.actionAbout = QtGui.QAction(parent=MainWindow)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/data/data/application-detail.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionAbout.setIcon(icon4)
        self.actionAbout.setObjectName("actionAbout")
        self.actionHelp = QtGui.QAction(parent=MainWindow)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/data/data/question-frame.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionHelp.setIcon(icon5)
        self.actionHelp.setObjectName("actionHelp")
        self.actionEdit_Datapoints = QtGui.QAction(parent=MainWindow)
        self.actionEdit_Datapoints.setCheckable(True)
        self.actionEdit_Datapoints.setEnabled(False)
        self.actionEdit_Datapoints.setObjectName("actionEdit_Datapoints")
        self.actionCurve = QtGui.QAction(parent=MainWindow)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/data/data/chart_curve_go.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionCurve.setIcon(icon6)
        self.actionCurve.setObjectName("actionCurve")
        self.actionCalculate_Topt_Tinf = QtGui.QAction(parent=MainWindow)
        self.actionCalculate_Topt_Tinf.setObjectName("actionCalculate_Topt_Tinf")
        self.actionAnalyse_Fit = QtGui.QAction(parent=MainWindow)
        self.actionAnalyse_Fit.setCheckable(True)
        self.actionAnalyse_Fit.setShortcutContext(QtCore.Qt.ShortcutContext.WindowShortcut)
        self.actionAnalyse_Fit.setObjectName("actionAnalyse_Fit")
        self.actionc_d = QtGui.QAction(parent=MainWindow)
        self.actionc_d.setObjectName("actionc_d")
        self.actionDeltaG_vs_T = QtGui.QAction(parent=MainWindow)
        self.actionDeltaG_vs_T.setCheckable(True)
        self.actionDeltaG_vs_T.setChecked(True)
        self.actionDeltaG_vs_T.setObjectName("actionDeltaG_vs_T")
        self.actionLn_rate_vs_T = QtGui.QAction(parent=MainWindow)
        self.actionLn_rate_vs_T.setCheckable(True)
        self.actionLn_rate_vs_T.setObjectName("actionLn_rate_vs_T")
        self.actionLn_rate_vs_1_T = QtGui.QAction(parent=MainWindow)
        self.actionLn_rate_vs_1_T.setCheckable(True)
        self.actionLn_rate_vs_1_T.setObjectName("actionLn_rate_vs_1_T")
        self.actionCSV = QtGui.QAction(parent=MainWindow)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/data/data/table_go.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionCSV.setIcon(icon7)
        self.actionCSV.setObjectName("actionCSV")
        self.actionRate_vs_T = QtGui.QAction(parent=MainWindow)
        self.actionRate_vs_T.setCheckable(True)
        self.actionRate_vs_T.setObjectName("actionRate_vs_T")
        self.action1_Rate_vs_1_T = QtGui.QAction(parent=MainWindow)
        self.action1_Rate_vs_1_T.setCheckable(True)
        self.action1_Rate_vs_1_T.setObjectName("action1_Rate_vs_1_T")
        self.menuFile.addAction(self.actionExit)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionRestart)
        self.menuExport.addAction(self.actionSession)
        self.menuExport.addSeparator()
        self.menuExport.addAction(self.actionData)
        self.menuExport.addAction(self.actionCurve)
        self.menuExport.addAction(self.actionCSV)
        self.menuHelp.addAction(self.actionAbout)
        self.menuHelp.addSeparator()
        self.menuHelp.addAction(self.actionHelp)
        self.menuRate_Type.addAction(self.actionDeltaG_vs_T)
        self.menuRate_Type.addAction(self.actionLn_rate_vs_T)
        self.menuRate_Type.addAction(self.actionLn_rate_vs_1_T)
        self.menuRate_Type.addAction(self.actionRate_vs_T)
        self.menuTools.addAction(self.actionEdit_Datapoints)
        self.menuTools.addAction(self.actionCalculate_Topt_Tinf)
        self.menuTools.addAction(self.actionAnalyse_Fit)
        self.menuTools.addAction(self.menuRate_Type.menuAction())
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuExport.menuAction())
        self.menubar.addAction(self.menuTools.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        self.actionExit.triggered.connect(MainWindow.close) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuExport.setTitle(_translate("MainWindow", "Export"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.menuTools.setTitle(_translate("MainWindow", "Tools"))
        self.menuRate_Type.setStatusTip(_translate("MainWindow", "Change plot type in parameter analysis window"))
        self.menuRate_Type.setTitle(_translate("MainWindow", "Rate Type"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionExit.setStatusTip(_translate("MainWindow", "Exit the application"))
        self.actionExit.setShortcut(_translate("MainWindow", "Ctrl+Q"))
        self.actionEdit_Parameters.setText(_translate("MainWindow", "Edit Parameters"))
        self.actionChange_input_units.setText(_translate("MainWindow", "Change input units"))
        self.actionEdit_Dataset_Names.setText(_translate("MainWindow", "Edit Dataset Names"))
        self.actionSession.setText(_translate("MainWindow", "Session"))
        self.actionSession.setStatusTip(_translate("MainWindow", "Export \'.mmrt\' file of session"))
        self.actionGraphs.setText(_translate("MainWindow", "Graphs"))
        self.actionData.setText(_translate("MainWindow", "Data"))
        self.actionData.setStatusTip(_translate("MainWindow", "Export folder containing plots and csv of fit statistics"))
        self.actionAll.setText(_translate("MainWindow", "All"))
        self.actionRestart.setText(_translate("MainWindow", "Restart"))
        self.actionRestart.setStatusTip(_translate("MainWindow", "Restart the application"))
        self.actionAbout.setText(_translate("MainWindow", "About"))
        self.actionAbout.setStatusTip(_translate("MainWindow", "About the application"))
        self.actionHelp.setText(_translate("MainWindow", "Help"))
        self.actionHelp.setStatusTip(_translate("MainWindow", "Show user manual"))
        self.actionHelp.setShortcut(_translate("MainWindow", "Ctrl+H"))
        self.actionEdit_Datapoints.setText(_translate("MainWindow", "Edit Datapoints"))
        self.actionEdit_Datapoints.setStatusTip(_translate("MainWindow", "Exclude datapoints from fit"))
        self.actionCurve.setText(_translate("MainWindow", "Curve"))
        self.actionCurve.setStatusTip(_translate("MainWindow", "Export csv of curves"))
        self.actionCalculate_Topt_Tinf.setText(_translate("MainWindow", "Calculate Topt/Tinf"))
        self.actionCalculate_Topt_Tinf.setStatusTip(_translate("MainWindow", "Calculate the temperature optimum and inflection point"))
        self.actionAnalyse_Fit.setText(_translate("MainWindow", "Analyse Fit"))
        self.actionAnalyse_Fit.setStatusTip(_translate("MainWindow", "Show temperature profile of fit parameters"))
        self.actionAnalyse_Fit.setShortcut(_translate("MainWindow", "Ctrl+P"))
        self.actionc_d.setText(_translate("MainWindow", "c d"))
        self.actionDeltaG_vs_T.setText(_translate("MainWindow", "DeltaG vs. T"))
        self.actionLn_rate_vs_T.setText(_translate("MainWindow", "Ln(rate) vs. T"))
        self.actionLn_rate_vs_1_T.setText(_translate("MainWindow", "Ln(rate) vs. 1/T"))
        self.actionCSV.setText(_translate("MainWindow", "CSV"))
        self.actionCSV.setStatusTip(_translate("MainWindow", "Export .csv file of fit data"))
        self.actionRate_vs_T.setText(_translate("MainWindow", "Rate vs. T"))
        self.action1_Rate_vs_1_T.setText(_translate("MainWindow", "1/Rate vs. 1/T"))
