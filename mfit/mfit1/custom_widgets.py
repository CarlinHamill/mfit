"""
This file is part of Macromolecular Rate Theory (MMRT) Fitting Tool (MFIT)
Copyright (C), 2023, Carlin Hamill.

MFIT is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import typing
import types
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from mfit.mfit1.main import ApplicationWindow

from PyQt6 import QtWidgets, QtCore, QtGui
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class DropFrame(QtWidgets.QFrame):
    # icon from https://icon-icons.com/icon/import-download/176152
    # by Nicky Lim
    # https://creativecommons.org/licenses/by/4.0/

    file_drop_event = QtCore.pyqtSignal(object)
    file_click_event = QtCore.pyqtSignal()

    def __init__(self, parent:QtWidgets.QWidget=None)->None:
        QtWidgets.QFrame.__init__(self, parent)
        self.setAcceptDrops(True)
    
    @staticmethod
    def has_file_suffix(file:str)->bool:
        acceptable_file_suffix = ['xlsx', 'csv', 'txt', 'xls', 'xlsm', 'tsv', 'tab', 'mmrt']
        if any([file.endswith(suffix) for suffix in acceptable_file_suffix]):
            return True
        return False
    
    def all_has_file_suffix(self, event:QtGui.QDragEnterEvent)->list[str]:
        file_list = [str(url.toLocalFile()) for url in event.mimeData().urls()]
        return all([self.has_file_suffix(file) for file in file_list])

    def dragEnterEvent(self, event:QtGui.QDragEnterEvent)->None:
        if self.all_has_file_suffix(event):
            event.accept()
            self.setStyleSheet(
                "QWidget#frame\n"
                "{\n"
                "    border:5px dashed rgb(182, 182, 182);\n"
                "    background-image: url(:/data/data/import_download_icon_176152.ico);\n"
                "    background-repeat: no-repeat; \n"
                "    background-position: center;\n"
                "    background-color: rgba(180,180,180,80);\n"
                "}"
            )
        else:
            event.ignore()
    
    def dragLeaveEvent(self, event:QtGui.QDragLeaveEvent)->None:
        self.setStyleSheet(
            "QWidget#frame\n"
            "{\n"
            "    border:5px dashed rgb(182, 182, 182);\n"
            "    background-image: url(:/data/data/import_download_icon_176152.ico);\n"
            "    background-repeat: no-repeat; \n"
            "    background-position: center;\n"
            "}"
        )

    def dropEvent(self, event:QtGui.QDropEvent)->None: 
        self.setStyleSheet(
            "QWidget#frame\n"
            "{\n"
            "    border:5px dashed rgb(182, 182, 182);\n"
            "    background-image: url(:/data/data/import_download_icon_176152.ico);\n"
            "    background-repeat: no-repeat; \n"
            "    background-position: center;\n"
            "}"
        )
        file_list = [str(url.toLocalFile()) for url in event.mimeData().urls()]
        self.file_drop_event.emit(file_list)

    def mouseReleaseEvent(self, event:QtGui.QMouseEvent):
        self.file_click_event.emit()


class MPLCanvas(FigureCanvas):
    """
    Base class for figure canvas objects
    Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.).
    https://matplotlib.org/2.0.2/examples/user_interfaces/embedding_in_qt5.html
    """
    def __init__(self, parent:QtWidgets.QWidget=None, width:int=5, height:int=4, dpi:int=100)->None:
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        FigureCanvas.updateGeometry(self)

    def add_plot(self, plot_function:typing.Callable)->None:
        plot_function(self.axes)
        self.fig.canvas.draw()

    def clear_plot(self)->None:
        self.axes.clear()


class PageWidget(QtWidgets.QWidget):
    """creates and adds widget from ui form to main layout. Widget is hidden by default"""
    def __init__(self,  ui_module:types.ModuleType, application:'ApplicationWindow', *args, **kwargs)->None:
        super(PageWidget, self).__init__(*args, **kwargs)
        self.application = application
        self.ui = ui_module.Ui_Form()
        self.ui.setupUi(self)
        self.hide()
        self.application.main_layout.addWidget(self)  


class LayoutWidget(QtWidgets.QWidget):
    """creates and adds widget from ui form to custom layout. Widget is hidden by default"""
    def __init__(self, ui_module:types.ModuleType, layout:QtWidgets.QLayout, *args, **kwargs)->None:
        super(LayoutWidget, self).__init__(*args, **kwargs)
        self.ui = ui_module.Ui_Form()
        self.ui.setupUi(self)
        self.hide()
        layout.addWidget(self)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)


class UIWidget(QtWidgets.QWidget):
    """creates and adds widget from ui form to custom layout. Widget is hidden by default"""
    def __init__(self, ui_module:types.ModuleType, *args, **kwargs)->None:
        super(UIWidget, self).__init__(*args, **kwargs)
        self.ui = ui_module.Ui_Form()
        self.ui.setupUi(self)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)


class DatasetButton(QtWidgets.QPushButton):
    control_click_event_signal = QtCore.pyqtSignal(object)
    control_added = False
    def __init__(self, *args, **kwargs)->None:
        QtWidgets.QPushButton.__init__(self, *args, **kwargs)
        self.setMinimumSize(QtCore.QSize(0, 32))
        self.setMaximumSize(QtCore.QSize(16777215, 32))
        self.setStyleSheet(
            "QPushButton\n"
            "{\n"
            "    border:1px solid rgb(182, 182, 182);\n"
            "    color: rgb(0, 0, 0);\n"
            "}\n"
            "QPushButton:hover\n"
            "{\n"
            "    border:1px solid rgb(150, 150, 150);\n"
            "    color: rgb(0, 0, 0);\n"
            "    background-color: rgba(180,180,180,80);\n"
            "}\n"
            "QPushButton:checked\n"
            "{\n"
            "    border:2px solid rgb(255, 255, 255);\n"
            "    color: rgb(0, 0, 0);\n"
            "    background-color: rgba(180,180,180,80);\n"
            "}"
        )

        self.setCheckable(True)

    
    def mousePressEvent(self, e:QtGui.QMouseEvent) -> None:
        if e.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
            self.control_click_event_signal.emit(self)
            return
        if self.isChecked():
            if e.button() == 1:
                self.toggled.emit(True)
            return
        return super().mousePressEvent(e)


class AnalysisCanvas(FigureCanvas):
    def __init__(self, parent:QtWidgets.QWidget=None, width:int=5, height:int=4, dpi:int=100) -> None:
        # self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        # self.axes = self.fig.add_subplot(2,2)
        self.fig, self.axes = plt.subplots(2,2, figsize=(width, height), dpi=dpi, tight_layout=True) 
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        FigureCanvas.updateGeometry(self)
        (self.rate_axis, self.enthalpy_axis), (self.entropy_axis, self.heat_capacity_axis) = self.axes

    def show_plot_labels(self, rate_axis:str)->None:
        for axl in self.axes:
            for ax in axl:
                ax.set_xlabel('Temperature (K)')
        self.rate_axis.set_xlabel(self.get_rate_xlabel(rate_axis))

        self.rate_axis.set_ylabel(self.get_rate_ylabel(rate_axis))
        self.enthalpy_axis.set_ylabel(r'$\Delta$H$^{\ddag}$ (kJ mol$^{-1}$ K$^{-1}$)')
        self.entropy_axis.set_ylabel(r'T$\Delta$S$^{\ddag}$ (kJ mol$^{-1}$ K$^{-1}$)')
        self.heat_capacity_axis.set_ylabel(r'$\Delta$C$_p^{\ddag}$ (kJ mol$^{-1}$ K$^{-1}$)')        

    def add_plot(self, plot_function:Callable, axes:str)->None:
        ax = self.match_axes(axes)
        plot_function(ax)
        self.fig.canvas.draw()

    def clear_plot(self)->None:
        for axl in self.axes:
            for ax in axl:
                ax.clear()

    @staticmethod
    def get_rate_ylabel(rate_axis:str)->str:
        match rate_axis:
            case 'deltag': return r'$\Delta$G$^{\ddag}$ (kJ mol$^{-1}$)'
            case 'lnrate': return r'ln(rate) [$\it{K}$$_{cat}$(sec$^{-1}$)]'
            case 'inverse': return r'ln(rate) [$\it{K}$$_{cat}$(sec$^{-1}$)]'
            case 'rate': return r'Rate [$\it{K}$$_{cat}$(sec$^{-1}$)]'

    @staticmethod
    def get_rate_xlabel(rate_axis:str)->str:
        match rate_axis:
            case 'inverse': return r'1/T (K$^{-1}$)'
            case _ : return 'Temperature (K)'

    def match_axes(self, axes_selection):
        match axes_selection.lower():
            case 'rate': return self.rate_axis
            case 'enthalpy': return self.enthalpy_axis
            case 'entropy': return self.entropy_axis
            case 'heat capacity': return self.heat_capacity_axis