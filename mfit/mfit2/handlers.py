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

from PyQt6 import QtWidgets, QtGui, QtCore

from mfit.mfit2.data_tools import ParamVariableType
from mfit.mfit2.file_tools import ExcelInput, CSVInput, TabDelimitedInput, MMRTFileInput, SpaceDelimitedInput
from mfit.mfit2.custom_widgets import DatasetButton, UIWidget
from mfit.mfit2.ui import dataset_change_widget_ui
import mfit.mfit2.Analysis as Analysis
import mfit.mfit2.dialogs as dialogs

from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
import matplotlib.backend_bases
from datetime import datetime
import numpy as np
import webbrowser
import pathlib
import json
import _csv
import csv
import sys
import os
import re

from typing import TYPE_CHECKING, Union, Callable
if TYPE_CHECKING:
    from main import ApplicationWindow
    from data_tools import Dataset, Parameter, FitMMRT2


class WidgetDisplayHandler:
    def __init__(self, application:'ApplicationWindow')->None:
        self.application = application

        self.get_ui_elements()
        self.link_slots()

    def get_ui_elements(self)->None:
        self.start_button = self.application.startup_widget.ui.pushButton
        self.datasets_added_signal = self.application.signals.datasets_added_signal

    def link_slots(self)->None:
        self.start_button.clicked.connect(self.show_file_input_screen)
        self.datasets_added_signal.connect(self.show_dataset_display_screen)

    def show_file_input_screen(self)->None:
        self.application.change_main_widget(self.application.file_input_widget)

    def show_dataset_display_screen(self)->None:
        self.application.change_main_widget(self.application.main_widget)
        self.application.showMaximized()


class FileInputHandler:
    def __init__(self, application:'ApplicationWindow')->None:
        self.application = application
        
        self.get_ui_elements()
        self.link_slots()

    def get_ui_elements(self)->None:
        self.frame = self.application.file_input_widget.ui.frame
        self.file_drop_event = self.frame.file_drop_event
        self.file_click_event = self.frame.file_click_event
        self.files_added_signal = self.application.signals.files_added_signal
    
    def link_slots(self)->None:
        self.file_click_event.connect(self.recieve_file_click)
        self.file_drop_event.connect(self.recieve_file_drop)

    def recieve_file_click(self)->None:
        file_list, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self.application, 
            'Import File', 
            QtCore.QDir.homePath(), 
            "DataFile (*.xlsx *.csv *.txt *.xls *.xlsm *.tsv *.tab, *.mmrt)"
            )
        if file_list:
            self.files_added_signal.emit(file_list)

    def recieve_file_drop(self, file_list:list[str])->None:
        self.files_added_signal.emit(file_list)


class FileImportHandler:
    def __init__(self, application:'ApplicationWindow')->None:
        self.application = application

        self.get_ui_elements()
        self.link_slots()

    def get_ui_elements(self)->None:
        self.files_added_signal = self.application.signals.files_added_signal
        self.files_imported_signal = self.application.signals.files_imported_signal

    def link_slots(self)->None:
        self.files_added_signal.connect(self.process_input_files)

    def process_input_files(self, file_list:list[str])->None:
        datasets = []
        failed = []
        for file_name in file_list:
            try:
                dataset = self.read_input_file_to_dataset(file_name)
                datasets.extend(dataset)
            except Exception as e:
                failed.append(file_name)
        if failed:
            self.show_error_message(failed)
        
        if datasets:
            self.files_imported_signal.emit(datasets)

    def read_input_file_to_dataset(self, file_name:str)->'Dataset':
        if file_name.endswith('csv'):
            return CSVInput(file_name)
        if any([file_name.endswith(suffix) for suffix in ['xlsx', 'xls', 'xlsm']]):
            return ExcelInput(file_name)
        if any([file_name.endswith(suffix) for suffix in ['tsv', 'tab']]):
            return TabDelimitedInput(file_name)
        if file_name.endswith('mmrt'):
            return MMRTFileInput(file_name)
        if file_name.endswith('txt'):
            try:
                return TabDelimitedInput(file_name)
            except Exception:
                return SpaceDelimitedInput(file_name)
        

    def show_error_message(self, failed_files):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        msg.setText("File Import Error")
        msg.setInformativeText("The following files could not be imported:\n\n" + '\n'.join(x for x in failed_files))
        msg.setWindowTitle("Error")
        msg.exec()   


class FileFitHandler:
    def __init__(self, application: 'ApplicationWindow') -> None:
        self.application = application

        self.get_ui_elements()
        self.link_slots()

    def get_ui_elements(self)->None:
        self.files_imported_signal = self.application.signals.files_imported_signal
        self.files_fitted_signal = self.application.signals.files_fitted_signal

    def link_slots(self)->None:
        self.files_imported_signal.connect(self.fit_imported_files)

    def fit_imported_files(self, datasets:list['Dataset'])->None:
        for dataset in datasets:
            dataset.fit_mmrt()
        self.files_fitted_signal.emit(datasets)


class DatasetAddHandler:
    def __init__(self, application: 'ApplicationWindow') -> None:
        self.application = application

        self.get_ui_elements()
        self.link_slots()

    def get_ui_elements(self)->None:
        self.database = self.application.database
        self.files_fitted_signal = self.application.signals.files_fitted_signal
        self.datasets_added_signal = self.application.signals.datasets_added_signal

    def link_slots(self)->None:
        self.files_fitted_signal.connect(self.add_datasets_to_database)

    def add_datasets_to_database(self, datasets)->None:
        self.database.add_datasets(datasets)
        self.datasets_added_signal.emit()


class ParameterSidebarHandler:
    def __init__(self, application:'ApplicationWindow')->None:
        self.application = application
        
        self.get_ui_elements()
        self.set_up_sidebar()
        self.link_slots()
        self.toggle_sidebar_widget()
        
    def get_ui_elements(self)->None:
        self.parameter_button = self.application.main_widget.ui.pushButton_4
        self.dataset_button = self.application.main_widget.ui.pushButton_5

        self.parameter_value_widget = self.application.parameter_value_widget
        self.parameter_start_widget = self.application.parameter_start_widget

        self.parameter_widget = self.application.main_widget.ui.widget_2
        self.dataset_widget = self.application.main_widget.ui.scrollArea_3

        # self.layout = self.application.main_widget.ui.verticalLayout_3

    def set_up_sidebar(self)->None:
        # self.dataset_widget.hide()

        self.parameter_value_widget.show()
        self.parameter_start_widget.show()

        # self.layout.setSpacing(6)

    def link_slots(self)->None:
        self.parameter_button.clicked.connect(self.toggle_sidebar_widget)
        self.dataset_button.clicked.connect(self.toggle_sidebar_widget)

    def toggle_sidebar_widget(self)->None:
        self.hide_widgets()
        if self.dataset_button.isChecked():
            self.dataset_widget.show()
            return
        self.parameter_widget.show()

    def hide_widgets(self)->None:
        self.parameter_widget.hide()
        self.dataset_widget.hide()


class DatasetSelectorHandler:
    def __init__(self, application:'ApplicationWindow')->None:
        self.application = application

        self.get_ui_elements()
        self.link_slots()
    
    def get_ui_elements(self)->None:
        self.scroll_area = self.application.main_widget.ui.scrollArea
        self.scroll_layout = self.application.main_widget.ui.verticalLayout_5
        self.datasets_added_signal = self.application.signals.datasets_added_signal
        self.database = self.application.database
        self.dataset_selected_signal = self.application.signals.dataset_selected_signal
        self.update_fit_success_signal = self.application.signals.update_fit_successes_signal
        self.name_changed_signal = self.application.signals.name_changed_signal
        self.mmrt1_button = self.application.main_widget.ui.pushButton_3
        self.update_dataset_display_signal = self.application.signals.update_dataset_display_signal
        self.dataset_plot_add_signal = self.application.signals.dataset_plot_add_signal
        self.reset_dataset_buttons_signal = self.application.signals.reset_dataset_buttons_signal

    def link_slots(self)->None:
        self.datasets_added_signal.connect(self.add_dataset_buttons)
        self.update_fit_success_signal.connect(self.update_fit_success)
        self.name_changed_signal.connect(self.change_name)
        self.update_dataset_display_signal.connect(self.change_name)
        self.reset_dataset_buttons_signal.connect(self.reset_buttons)

    def add_dataset_buttons(self)->None:
        self.button_group = QtWidgets.QButtonGroup(self.application)
        for idx, (key, dataset) in enumerate(self.database.get_items()):
            if hasattr(self, f'{key}_button'):
                getattr(self, f'{key}_button').setText(dataset.name)
                return
            setattr(self, f'{key}_button', DatasetButton(self.scroll_area, text=dataset.name))

            button = getattr(self, f'{key}_button')

            self.scroll_layout.insertWidget(idx, button)
            button.setObjectName(f'{key}')
            button.toggled.connect(self.dataset_button_toggled)
            button.control_click_event_signal.connect(self.control_button_clicked)
            self.button_group.addButton(button)
            button.setStatusTip('Click to select dataset. Control-click to add to plot.')
        
        self.update_fit_success()

    def control_button_clicked(self, dataset_button:DatasetButton)->None:
        if dataset_button.control_added:
            return
        if dataset_button.isChecked():
            return
        dataset_button.setStyleSheet(
            "QPushButton\n"
            "{\n"
            "    border:1px solid rgb(150, 150, 150);\n"
            "    background-color: rgba(180,180,180,80);\n"
            "}"
        )
        dataset_button.control_added = True
        self.dataset_plot_add_signal.emit(int(dataset_button.objectName()))
        
    def dataset_button_toggled(self)->None:
        key = int(self.application.sender().objectName())
        button = getattr(self, f'{key}_button')
        if button.isChecked():
            self.reset_buttons()
            self.dataset_selected_signal.emit(key)

    def update_fit_success(self)->None:
        for key, dataset in self.database.get_items():
            button = getattr(self, f'{key}_button')
            if dataset.fit.mmrt2.mmrt2_fit is None:
                self.set_stylesheet(button, 'rgb(255,0,0)')#no fit = red
                continue
            if dataset.fit.mmrt2.values['Tc'] -1 < dataset.inputs.temperature.min() or dataset.fit.mmrt2.values['Tc'] +1 > dataset.inputs.temperature.max():
                self.set_stylesheet(button, 'rgb(226,174,40)')#Tc at edge = yellow
                continue
            if dataset.fit.mmrt2.error['H'] is None or dataset.fit.mmrt2.error['H'] == 0:
                self.set_stylesheet(button, 'rgb(226,174,40)')#no error = yellow
                continue
            self.set_stylesheet(button, 'rgb(0,0,0)')#normal = black

    def set_stylesheet(self, button:QtWidgets.QPushButton, colour:str)->None:
        button.setStyleSheet(
            "QPushButton\n"
            "{\n"
            "    border:1px solid rgb(182, 182, 182);\n"
            f"    color: {colour};\n"
            "}\n"
            "QPushButton:hover\n"
            "{\n"
            "    border:1px solid rgb(150, 150, 150);\n"
            f"    color: {colour};\n"
            "    background-color: rgba(180,180,180,80);\n"
            "}\n"
            "QPushButton:checked\n"
            "{\n"
            "    border:1px solid rgb(0, 0, 0);\n"
            f"   color: {colour};\n"
            "    background-color: rgba(180,180,180,80);\n"
            "}"
        )

    def reset_buttons(self)->None:
        for key in self.database.get_keys():
            button = getattr(self, f'{key}_button')
            button.control_added = False
            button.setStyleSheet(
                "QPushButton\n"
                "{\n"
                "    border:1px solid rgb(182, 182, 182);\n"
                "}\n"
                "QPushButton:hover\n"
                "{\n"
                "    border:1px solid rgb(150, 150, 150);\n"
                "    background-color: rgba(180,180,180,80);\n"
                "}\n"
                "QPushButton:checked\n"
                "{\n"
                "    border:1px solid rgb(0, 0, 0);\n"
                "    background-color: rgba(180,180,180,80);\n"
                "}"
            ) 
            self.update_fit_success()    

    def change_name(self)->None:
        for (key, dataset) in self.database.get_items():
            getattr(self, f'{key}_button').setText(dataset.name)


class GraphHandler:
    current_selected_dataset_key:int = None
    cpl_shown = False
    def __init__(self, application:'ApplicationWindow')->None:
        self.application = application

        self.get_ui_elements()
        self.link_slots()

    def get_ui_elements(self)->None:
        self.fig = self.application.main_widget.ui.page
        self.database = self.application.database
        self.dataset_selected_signal = self.application.signals.dataset_selected_signal
        self.update_dataset_display_signal = self.application.signals.update_dataset_display_signal
        self.dataset_plot_add_signal = self.application.signals.dataset_plot_add_signal
        self.show_cpl_signal = self.application.signals.show_cpl_signal

    def link_slots(self)->None:
        self.dataset_selected_signal.connect(self.dataset_selected_changed)
        self.update_dataset_display_signal.connect(self.plot_dataset)
        self.dataset_plot_add_signal.connect(self.add_plot)
        self.show_cpl_signal.connect(self.display_cpl)

    def display_cpl(self, shown:bool)->None:
        self.cpl_shown = shown

    def dataset_selected_changed(self, key:int)->None:
        self.current_selected_dataset_key = key
        self.plot_dataset()

    def add_plot(self, key:int)->None:
        if key == self.current_selected_dataset_key:
            return
        dataset = self.database.get_dataset(key)
        fit = dataset.fit.mmrt2
        if fit.mmrt2_fit is None:
            self.fig.add_plot(dataset.plot, picker=7)
            return 
        self.fig.add_plot(fit.plot, picker=7)

    def plot_dataset(self)->None:
        self.fig.clear_plot()
        dataset = self.database.get_dataset(self.current_selected_dataset_key)
        fit = dataset.fit.mmrt2
        if fit.mmrt2_fit is None:
            self.fig.add_plot(dataset.plot)
        else:
            self.fig.add_plot(fit.plot)

        if self.cpl_shown:
            self.plot_cpl()

    def plot_cpl(self)->None:
        dataset = self.database.get_dataset(self.current_selected_dataset_key)

        if dataset.fit.mmrt2.cpl_fit is None:
            self.fig.add_plot(dataset.plot_cpl_points)
        self.fig.add_plot(dataset.fit.mmrt2.cpl_fit.plot)


class ParameterDisplayHandler:
    current_selected_dataset_key:int = None
    def __init__(self, application:'ApplicationWindow')->None:
        self.application = application

        self.get_ui_elements()
        self.link_slots()

    def get_ui_elements(self)->None:
        self.database = self.application.database
        self.dataset_selected_signal = self.application.signals.dataset_selected_signal
        self.widget = self.application.parameter_value_widget.ui
        self.update_dataset_display_signal = self.application.signals.update_dataset_display_signal

    def link_slots(self)->None:
        self.dataset_selected_signal.connect(self.dataset_selected_changed)
        self.update_dataset_display_signal.connect(self.show_dataset_parameters)

    def dataset_selected_changed(self, key:int)->None:
        self.current_selected_dataset_key = key
        self.show_dataset_parameters()

    def show_dataset_parameters(self)->None:
        dataset = self.database.get_dataset(self.current_selected_dataset_key)

        lineedit_map = {
            'H'  : [self.widget.enthalpyLineEdit , self.widget.lineEdit],
            'S'  : [self.widget.entropyLineEdit , self.widget.lineEdit_2],
            'ddH': [self.widget.enthalpyDifferenceLineEdit , self.widget.lineEdit_3],
            'Tc' : [self.widget.TcLineEdit , self.widget.lineEdit_4],
            'CpL': [self.widget.heatCapacityLowerLineEdit , self.widget.lineEdit_5],
            'CpH': [self.widget.heatCapacityUpperLineEdit , self.widget.lineEdit_6],
            'Tn' : [self.widget.tnLineEdit , self.widget.lineEdit_7]
        }

        self.write_parameters(dataset.fit.mmrt2, lineedit_map)

    def write_parameters(self, dataset_fit:'FitMMRT2', lineedit_map:dict[str, list[QtWidgets.QLineEdit]])->None:
        if dataset_fit.mmrt2_fit is None:
            for (value_lineedit, error_lineedit) in lineedit_map.values():
                value_lineedit.setText('')
                error_lineedit.setText('')
            return
        
        errors = {key:val if val is not None else 0 for (key, val) in dataset_fit.error.items()}
        for par, (value_lineedit, error_lineedit) in lineedit_map.items():
            value_lineedit.setText(str(round(dataset_fit.values[par],4)))
            error_lineedit.setText(str(round(errors[par],4)))


class ParameterEditHandler:
    current_selected_dataset_key:int = None
    def __init__(self, application:'ApplicationWindow')->None:
        self.application = application

        self.get_ui_elements()
        self.link_slots()
        self.set_validators()
        self.enable_editing(False)

    def enable_editing(self, enabled:bool):
        self.application.parameter_start_widget.setEnabled(enabled)
   
    def set_validators(self)->None: 
        lineedits = [
            self.widget.enthalpyLineEdit,
            self.widget.entropyLineEdit,
            self.widget.enthalpyDifferenceLineEdit,
            self.widget.heatCapacityUpperLineEdit,
            self.widget.tnLineEdit,
            self.widget.TcLineEdit,
            self.widget.heatCapacityLowerLineEdit
        ] 
        validator = QtGui.QDoubleValidator()

        for lineedit in lineedits:
            lineedit.setValidator(validator)
    
    def get_ui_elements(self)->None:
        self.database = self.application.database
        self.dataset_selected_signal = self.application.signals.dataset_selected_signal
        
        self.widget = self.application.parameter_start_widget.ui

        self.update_dataset_display_signal = self.application.signals.update_dataset_display_signal
        self.update_fit_success_signal = self.application.signals.update_fit_successes_signal
        self.reset_dataset_buttons_signal = self.application.signals.reset_dataset_buttons_signal

    def link_slots(self)->None:
        self.dataset_selected_signal.connect(self.dataset_selected_changed)
        self.update_dataset_display_signal.connect(self.update_cpl)
        self.enable_link_slots()

    def dataset_selected_changed(self, key:int)->None:
        self.current_selected_dataset_key = key
        self.enable_editing(True)
        self.disable_link_slots()
        self.write_starting_parameters()
        self.enable_link_slots()
    
    def get_param_objects(self, dataset:'Dataset')->list[tuple[QtWidgets.QComboBox, 'Parameter', QtWidgets.QLineEdit]]:
        mmrt2_paramater_objects = [
            (self.widget.comboBox_7 , dataset.parameters.H, self.widget.enthalpyLineEdit,),
            (self.widget.comboBox_6 , dataset.parameters.S, self.widget.entropyLineEdit),
            (self.widget.comboBox_5 , dataset.parameters.ddH, self.widget.enthalpyDifferenceLineEdit),
            (self.widget.comboBox_4 , dataset.parameters.CpH, self.widget.heatCapacityUpperLineEdit),
            (self.widget.comboBox_3 , dataset.parameters.Tn, self.widget.tnLineEdit),
            (self.widget.comboBox , dataset.parameters.Tc, self.widget.TcLineEdit),
            (self.widget.comboBox_2 , dataset.parameters.CpL, self.widget.heatCapacityLowerLineEdit)
        ]        
       
        return mmrt2_paramater_objects
       
    def write_starting_parameters(self):
        dataset = self.database.get_dataset(self.current_selected_dataset_key)
        for (combo_box, param, line_edit) in self.get_param_objects(dataset):
            combo_box.setCurrentIndex(param.variable.value)
            line_edit.setText(str(round(param.value, 4)))

    def disable_link_slots(self)->None:
        # Combobox index changed
        self.widget.comboBox_7.currentIndexChanged.disconnect()
        self.widget.comboBox_6.currentIndexChanged.disconnect()
        self.widget.comboBox_5.currentIndexChanged.disconnect()
        self.widget.comboBox_4.currentIndexChanged.disconnect()
        self.widget.comboBox_3.currentIndexChanged.disconnect()
        self.widget.comboBox.currentIndexChanged.disconnect()
        self.widget.comboBox_2.currentIndexChanged.disconnect()
        
        # Lineedit edited
        self.widget.enthalpyLineEdit.textChanged.disconnect()
        self.widget.entropyLineEdit.textChanged.disconnect()
        self.widget.enthalpyDifferenceLineEdit.textChanged.disconnect()        
        self.widget.heatCapacityUpperLineEdit.textChanged.disconnect()
        self.widget.heatCapacityLowerLineEdit.textChanged.disconnect()
        self.widget.TcLineEdit.textChanged.disconnect()
        self.widget.tnLineEdit.textChanged.disconnect()

        # Lineedit editing finished
        self.widget.enthalpyLineEdit.editingFinished.disconnect()
        self.widget.entropyLineEdit.editingFinished.disconnect()
        self.widget.enthalpyDifferenceLineEdit.editingFinished.disconnect()        
        self.widget.heatCapacityUpperLineEdit.editingFinished.disconnect()
        self.widget.heatCapacityLowerLineEdit.editingFinished.disconnect()
        self.widget.TcLineEdit.editingFinished.disconnect()
        self.widget.tnLineEdit.editingFinished.disconnect()

    def enable_link_slots(self)->None:
        # Combobox index changed
        self.widget.comboBox_7.currentIndexChanged.connect(lambda: self.combo_edited(self.widget.comboBox_7, 'H'))
        self.widget.comboBox_6.currentIndexChanged.connect(lambda: self.combo_edited(self.widget.comboBox_6, 'S'))
        self.widget.comboBox_5.currentIndexChanged.connect(lambda: self.combo_edited(self.widget.comboBox_5, 'ddH'))
        self.widget.comboBox_4.currentIndexChanged.connect(lambda: self.combo_edited(self.widget.comboBox_4, 'CpH'))
        self.widget.comboBox_3.currentIndexChanged.connect(lambda: self.combo_edited(self.widget.comboBox_3, 'Tn'))
        self.widget.comboBox.currentIndexChanged.connect(lambda: self.combo_edited(self.widget.comboBox, 'Tc'))
        self.widget.comboBox_2.currentIndexChanged.connect(lambda: self.combo_edited(self.widget.comboBox_2, 'CpL'))
        
        # Lineedit edited
        self.widget.enthalpyLineEdit.textChanged.connect(lambda: self.lineedit_edited(self.widget.enthalpyLineEdit))
        self.widget.entropyLineEdit.textChanged.connect(lambda: self.lineedit_edited(self.widget.entropyLineEdit))
        self.widget.enthalpyDifferenceLineEdit.textChanged.connect(lambda: self.lineedit_edited(self.widget.enthalpyDifferenceLineEdit))        
        self.widget.heatCapacityUpperLineEdit.textChanged.connect(lambda: self.lineedit_edited(self.widget.heatCapacityUpperLineEdit))
        self.widget.heatCapacityLowerLineEdit.textChanged.connect(lambda: self.lineedit_edited(self.widget.heatCapacityLowerLineEdit))
        self.widget.TcLineEdit.textChanged.connect(lambda: self.lineedit_edited(self.widget.TcLineEdit))
        self.widget.tnLineEdit.textChanged.connect(lambda: self.lineedit_edited(self.widget.tnLineEdit))

        # Lineedit editing finished
        self.widget.enthalpyLineEdit.editingFinished.connect(lambda: self.linedit_finished(self.widget.enthalpyLineEdit, 'H'))
        self.widget.entropyLineEdit.editingFinished.connect(lambda: self.linedit_finished(self.widget.entropyLineEdit, 'S'))
        self.widget.enthalpyDifferenceLineEdit.editingFinished.connect(lambda: self.linedit_finished(self.widget.enthalpyDifferenceLineEdit, 'ddH'))        
        self.widget.heatCapacityUpperLineEdit.editingFinished.connect(lambda: self.linedit_finished(self.widget.heatCapacityUpperLineEdit, 'CpH'))
        self.widget.heatCapacityLowerLineEdit.editingFinished.connect(lambda: self.linedit_finished(self.widget.heatCapacityLowerLineEdit, 'CpL'))
        self.widget.TcLineEdit.editingFinished.connect(lambda: self.linedit_finished(self.widget.TcLineEdit, 'Tc'))
        self.widget.tnLineEdit.editingFinished.connect(lambda: self.linedit_finished(self.widget.tnLineEdit, 'Tn'))

    def combo_edited(self, combobox:QtWidgets.QComboBox, param:str)->None:
        dataset = self.database.get_dataset(self.current_selected_dataset_key)
        parameter = getattr(dataset.parameters, param)    
        
        if combobox.currentIndex() == 0:#Free
            parameter.variable = ParamVariableType.Free
        elif combobox.currentIndex() == 1:#Fixed
            parameter.variable = ParamVariableType.Fixed
        else:
            parameter.variable = ParamVariableType.Auto

        if combobox is self.widget.comboBox_2:
            if combobox.currentIndex() == 2:
                self.widget.heatCapacityLowerLineEdit.setReadOnly(True)
            else:
                self.widget.heatCapacityLowerLineEdit.setReadOnly(False)
                self.linedit_finished(self.widget.heatCapacityLowerLineEdit, 'CpL')
        if combobox is self.widget.comboBox:
            if combobox.currentIndex() == 2:
                self.widget.TcLineEdit.setReadOnly(True)
            else:
                self.widget.TcLineEdit.setReadOnly(False)
                self.linedit_finished(self.widget.TcLineEdit, 'Tc')

        self.refit_mmrt(dataset)

    def lineedit_edited(self, lineedit:QtWidgets.QLineEdit)->None:
        lineedit.setStyleSheet("color: red;")

    def linedit_finished(self, lineedit:QtWidgets.QLineEdit, param:str)->None:
        lineedit.setStyleSheet("color: black;")        
        dataset = self.database.get_dataset(self.current_selected_dataset_key)
        parameter = getattr(dataset.parameters, param)    

        new_param = lineedit.text()
        if new_param == '':
            new_param = '0'

        parameter.value = float(new_param)

        self.refit_mmrt(dataset)

    def refit_mmrt(self, dataset:'Dataset')->None:
        dataset.fit_mmrt()
        self.update_dataset_display_signal.emit()
        self.reset_dataset_buttons_signal.emit()       
        self.update_fit_success_signal.emit()

    def update_cpl(self):
        self.dataset_selected_changed(self.current_selected_dataset_key)


class HeatCapacityLowerEditHandler:
    current_selected_dataset_key = None
    def __init__(self, application:'ApplicationWindow')->None:
        self.application = application

        self.get_ui_elements()
        self.link_slots()

        self.hide_widgets(True)
        self.low_spin.setKeyboardTracking(False)
        self.high_spin.setKeyboardTracking(False)

    def get_ui_elements(self)->None:
        self.database = self.application.database
        self.dataset_selected_signal = self.application.signals.dataset_selected_signal
        self.cpl_button = self.application.main_widget.ui.pushButton_3
        self.low_spin = self.application.main_widget.ui.spinBox
        self.high_spin = self.application.main_widget.ui.spinBox_2

        self.show_cpl_signal = self.application.signals.show_cpl_signal

        self.cpl_param_combobox = self.application.parameter_start_widget.ui.comboBox_2
        self.cpl_param_lineedit = self.application.parameter_start_widget.ui.heatCapacityLowerLineEdit
        
        self.update_dataset_display_signal = self.application.signals.update_dataset_display_signal
        self.update_fit_success_signal = self.application.signals.update_fit_successes_signal
        self.reset_dataset_buttons_signal = self.application.signals.reset_dataset_buttons_signal

    def hide_widgets(self, hidden:bool)->None:
        self.low_spin.setHidden(hidden)
        self.high_spin.setHidden(hidden)

    def link_slots(self)->None:
        self.dataset_selected_signal.connect(self.dataset_selected_changed)    
        self.cpl_button.clicked.connect(self.cpl_clicked)
        
        self.low_spin.valueChanged.connect(self.spin_changed)        
        self.high_spin.valueChanged.connect(self.spin_changed)  

        self.cpl_param_combobox.currentTextChanged.connect(self.cpl_combo_changed)

    def cpl_combo_changed(self)->None:
        current_index = self.cpl_param_combobox.currentIndex()
        if current_index == 2:#Auto
            self.cpl_button.setStyleSheet(
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
                "    border:1px solid rgb(0, 0, 0);\n"
                "    color: rgb(0, 0, 0);\n"
                "    background-color: rgba(180,180,180,80);\n"
                "}"
            )
            return    
        self.cpl_button.setStyleSheet(
            "QPushButton\n"
            "{\n"
            "    border:1px solid rgb(182, 182, 182);\n"
            "    color: rgb(0, 0, 0);\n"
            "}\n"
        )

    def dataset_selected_changed(self, key:int)->None:
        self.current_selected_dataset_key = key
        self.set_spins()

    def set_spins(self)->None:
        dataset = self.database.get_dataset(self.current_selected_dataset_key)
        self.change_spin_box(self.low_spin, dataset.parameters.CpL_Point1.value)
        self.change_spin_box(self.high_spin, dataset.parameters.CpL_Point2.value)

    def change_spin_box(self, spinbox:QtWidgets.QSpinBox, value:int)->None:
        self.disable_spins()
        spinbox.setValue(value)
        self.enable_spins()       

    def enable_spins(self)->None:
        self.low_spin.valueChanged.connect(self.spin_changed)        
        self.high_spin.valueChanged.connect(self.spin_changed) 

    def disable_spins(self)->None:     
        self.low_spin.valueChanged.disconnect(self.spin_changed)        
        self.high_spin.valueChanged.disconnect(self.spin_changed) 

    def cpl_clicked(self)->None:
        if self.current_selected_dataset_key is None or self.cpl_param_combobox.currentIndex() in [0, 1]:
            self.cpl_button.setChecked(False)
            return
        if self.cpl_button.isChecked():
            self.toggle_cpl(True)
            return
        self.toggle_cpl(False)

    def toggle_cpl(self, shown:bool):
        self.hide_widgets(not shown)
        self.show_cpl_signal.emit(shown)
        self.cpl_param_combobox.setDisabled(shown)
        self.update_dataset_display_signal.emit()
        self.reset_dataset_buttons_signal.emit()       
        self.update_fit_success_signal.emit()

    def spin_changed(self)->None:
        self.set_spinbox_ranges()
        low = self.low_spin.value()
        high = self.high_spin.value()

        dataset = self.database.get_dataset(self.current_selected_dataset_key)
        dataset.parameters.CpL_Point1.value = low
        dataset.parameters.CpL_Point2.value = high

        self.refit_mmrt(dataset)

    def set_spinbox_ranges(self)->None:
        low = self.low_spin.value()
        high = self.high_spin.value()
        length = len(self.database.get_dataset(self.current_selected_dataset_key).inputs.temperature)

        self.low_spin.setRange(1, high-2)
        self.high_spin.setRange(low+2, length)

    def refit_mmrt(self, dataset:'Dataset')->None:
        dataset.fit_mmrt()
        self.update_dataset_display_signal.emit()
        self.reset_dataset_buttons_signal.emit()       
        self.update_fit_success_signal.emit()


class DatasetChangeHandler:
    current_selected_dataset_key:int = None
    def __init__(self, application:'ApplicationWindow')->None:
        self.application = application

        self.get_ui_elements()
        self.link_slots()

    def get_ui_elements(self)->None:
        self.scroll_area = self.application.main_widget.ui.scrollArea_3
        self.scroll_layout = self.application.main_widget.ui.verticalLayout_6
        self.database = self.application.database
        self.datasets_added_signal = self.application.signals.datasets_added_signal
        self.dataset_selected_signal = self.application.signals.dataset_selected_signal
        self.update_dataset_display_signal = self.application.signals.update_dataset_display_signal
        self.update_fit_success_signal = self.application.signals.update_fit_successes_signal
        self.reset_dataset_buttons_signal = self.application.signals.reset_dataset_buttons_signal

    def link_slots(self)->None:
        self.datasets_added_signal.connect(self.add_dataset_widgets)
        self.dataset_selected_signal.connect(self.dataset_selected_changed)
    
    def dataset_selected_changed(self, key:int)->None:
        self.current_selected_dataset_key = key

    def add_dataset_widgets(self)->None:
        database_items = self.database.get_items()
        for idx, (key, dataset) in enumerate(database_items):
            setattr(self, f'{key}_dataset_widget', UIWidget(dataset_change_widget_ui, self.scroll_area))
            
            widget = getattr(self, f'{key}_dataset_widget')
            lineedit = widget.ui.lineEdit
            combo = widget.ui.comboBox_3

            self.scroll_layout.insertWidget(idx, widget)
            lineedit.setObjectName(f'{key}_lineedit')
            combo.setObjectName(f'{key}_combo')
            setattr(self, f'{key}_lineedit', lineedit)
            setattr(self, f'{key}_combo', combo)
            
            combo.setCurrentIndex(dataset.inputs.rate_type.value)
            lineedit.setText(dataset.name)

            lineedit.textChanged.connect(self.dataset_name_edited)
            lineedit.editingFinished.connect(self.dataset_name_finished)
            combo.currentIndexChanged.connect(self.rate_type_changed)
    
    def dataset_name_edited(self)->None:
        key = int(self.application.sender().objectName().split('_')[0])
        self.change_lineedit_colour(key, 'rgb(255,0,0)')

    def dataset_name_finished(self)->None:
        key = int(self.application.sender().objectName().split('_')[0])
        self.change_lineedit_colour(key, 'rgb(0,0,0)')
        dataset = self.database.get_dataset(key)
        
        new_name = getattr(self, f'{key}_lineedit').text()
        dataset.change_name(new_name)

        self.update_dataset_display_signal.emit()
        self.reset_dataset_buttons_signal.emit()

    def rate_type_changed(self)->None:
        key = int(self.application.sender().objectName().split('_')[0])
        dataset = self.database.get_dataset(key)
        combo_idx = getattr(self, f'{key}_combo').currentIndex()
        dataset.change_input_type(combo_idx)    
        self.refit_mmrt(dataset)

    def refit_mmrt(self, dataset:'Dataset')->None:
        dataset.fit_mmrt()
        self.update_dataset_display_signal.emit()
        self.reset_dataset_buttons_signal.emit()       
        self.update_fit_success_signal.emit()

    def change_lineedit_colour(self, key:int, colour:str)->None:
        widget = getattr(self, f'{key}_dataset_widget')
        widget.setStyleSheet(
            "QWidget#Form\n"
            "{\n"
            "    border:1px solid rgb(150, 150, 150);\n"
            "    background-color: rgb(255, 255, 255);\n"
            "}\n"
            "QLineEdit\n"
            "{\n"
            "    border:1px solid rgb(182, 182, 182);\n"
            "    background-color: rgb(255, 255, 255);\n"
            f"    color: {colour};\n"
            "}\n"
            "QComboBox\n"
            "{\n"
            "    background-color: rgb(255, 255, 255);\n"
            "    border:1px solid rgb(182, 182, 182);\n"
            "}\n"
        )


class RemoveDatapointHandler:
    current_selected_dataset_key:int = None
    def __init__(self, application:'ApplicationWindow')->None:
        self.application = application
        self.get_ui_elements()
        self.link_slots()

    def get_ui_elements(self)->None:
        self.database = self.application.database
        self.dataset_selected_signal = self.application.signals.dataset_selected_signal
        self.edit_action = self.application.base_ui_widget.actionEdit_Datapoints
        self.fig = self.application.main_widget.ui.page
        self.update_dataset_display_signal = self.application.signals.update_dataset_display_signal
        self.update_fit_successes_signal = self.application.signals.update_fit_successes_signal
        self.reset_dataset_buttons_signal = self.application.signals.reset_dataset_buttons_signal

    def link_slots(self)->None:
        self.dataset_selected_signal.connect(self.dataset_selection_changed)
        self.fig.fig.canvas.callbacks.connect('pick_event', self.data_picked)
        self.edit_action.toggled.connect(self.toggle_edit_mode)
    
    def dataset_selection_changed(self, key:int)->None:
        self.current_selected_dataset_key = key
        self.edit_action.setEnabled(True)

    def toggle_edit_mode(self)->None:
        if self.edit_action.isChecked():
            self.rect_selector = RectangleSelector(self.fig.axes, self.select_range, button=[1], useblit=True)
            return
        if hasattr(self, 'rect_selector'):
            del self.rect_selector
    
    def data_picked(self, event:matplotlib.backend_bases.PickEvent)->None:
        if not self.edit_action.isChecked():
            return
        pick_idx = event.ind[0]
        picker = event.artist.get_picker()

        if picker ==7:
            self.update_dataset_display_signal.emit()
            self.update_fit_successes_signal.emit()
            self.reset_dataset_buttons_signal.emit()
            return

        self.dataset_selected = self.database.get_dataset(self.current_selected_dataset_key)

        if picker == 5: #label not in ['_curve', '_o1', '_removed']:
            x_point = self.dataset_selected.inputs.temperature[self.dataset_selected.mask!=0][pick_idx]
            y_point = self.dataset_selected.inputs.deltag[self.dataset_selected.mask!=0][pick_idx]
            arr_idx = np.intersect1d(
                np.where(self.dataset_selected.inputs.temperature==x_point), 
                np.where(self.dataset_selected.inputs.deltag==y_point)
            )
            for idx in arr_idx:
                self.dataset_selected.change_mask(idx, 0)

        elif picker == 6: #label == '_removed':
            x_point = self.dataset_selected.inputs.temperature[self.dataset_selected.mask!=1][pick_idx]
            y_point = self.dataset_selected.inputs.deltag[self.dataset_selected.mask!=1][pick_idx]
            arr_idx = np.intersect1d(
                np.where(self.dataset_selected.inputs.temperature==x_point), 
                np.where(self.dataset_selected.inputs.deltag==y_point)
            )
            for idx in arr_idx:
                self.dataset_selected.change_mask(idx, 1)
        
        self.dataset_selected.fit_mmrt()
        self.update_dataset_display_signal.emit()
        self.update_fit_successes_signal.emit()
        self.reset_dataset_buttons_signal.emit()

    def select_range(self, eclick:matplotlib.backend_bases.MouseEvent, erelease:matplotlib.backend_bases.MouseEvent)->None:
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])

        self.dataset_selected = self.database.get_dataset(self.current_selected_dataset_key)

        arr_idxs = np.intersect1d(
            np.where(
                (self.dataset_selected.inputs.temperature>x0)&(self.dataset_selected.inputs.temperature<x1)
            ), 
            np.where(
                (self.dataset_selected.inputs.deltag>y0)&(self.dataset_selected.inputs.deltag<y1)
            )
        )

        filtered_mask = self.dataset_selected.mask[arr_idxs]
        avg = np.average(filtered_mask)
        
        if avg >= 0.5:
            for idx in arr_idxs:
                self.dataset_selected.change_mask(idx, 0)

        else:
            for idx in arr_idxs:
                self.dataset_selected.change_mask(idx, 1)            
        
        self.dataset_selected.fit_mmrt()
        self.update_dataset_display_signal.emit()
        self.update_fit_successes_signal.emit()
        self.reset_dataset_buttons_signal.emit()


class HelpMenuHandler:
    def __init__(self, application:'ApplicationWindow')->None:
        self.application = application

        self.get_ui_elements()
        self.link_slots()
    
    def get_ui_elements(self)->None:
        self.about = self.application.base_ui_widget.actionAbout
        self.help = self.application.base_ui_widget.actionHelp

    def link_slots(self)->None:
        self.about.triggered.connect(self.launch_about)
        self.help.triggered.connect(self.launch_help)

    def launch_about(self)->None:
        dialog = dialogs.AboutDialog(self.application)
        dialog.exec()

    def launch_help(self)->None:
        file = self.resource_path("Help.pdf")
        webbrowser.open_new(file)

    @staticmethod
    def resource_path(relative_path:Union[pathlib.Path, str])->Union[pathlib.Path, str]:
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os. path.abspath("mfit/mfit1")

        return os.path.join(base_path, relative_path)


class ExportHandler:
    def __init__(self, application:'ApplicationWindow')->None:
        self.application = application

        self.get_ui_elements()
        self.link_slots()

        self.export_data_action.setDisabled(True)
        self.export_session_action.setDisabled(True)
        self.export_curve_action.setDisabled(True)
        self.export_csv_action.setDisabled(True)
    
    def get_ui_elements(self)->None:
        self.export_session_action = self.application.base_ui_widget.actionSession
        self.database = self.application.database
        self.export_data_action = self.application.base_ui_widget.actionData
        self.datasets_added_signal = self.application.signals.datasets_added_signal
        self.export_curve_action = self.application.base_ui_widget.actionCurve
        self.export_csv_action = self.application.base_ui_widget.actionCSV

    def link_slots(self)->None:
        self.export_session_action.triggered.connect(self.export_session)
        self.export_data_action.triggered.connect(self.export_data)
        self.datasets_added_signal.connect(self.enable_export)
        self.export_curve_action.triggered.connect(self.export_curve)
        self.export_csv_action.triggered.connect(self.export_csv)

    def enable_export(self)->None:
        self.export_data_action.setEnabled(True)
        self.export_session_action.setEnabled(True)
        self.export_curve_action.setEnabled(True)
        self.export_csv_action.setEnabled(True)

    def export_csv(self)->None:
        export_file, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.application,
            'Save Curves',
            QtCore.QDir.homePath(),
            'CSV File (*.csv)'
        )

        if export_file == "":
            return

        progress_bar = self.start_progress_bar(2)
        progress_bar.show()
        progress_bar.setLabelText('Exporting CSV')

        mmrt2_params, stat_map = self.get_export_maps()

        progress_bar.setValue(1)
        
        with open(export_file, 'w', newline='') as csvfile:   
            writer = csv.writer(csvfile)
            writer.writerow(['MMRT Fits'])
            writer.writerow([''])   

            self.write_csv(writer, 'MMRT 2.0', mmrt2_params, lambda dataset: dataset.fit.mmrt2, stat_map)

        progress_bar.setValue(2)

    def export_curve(self)->None:
        export_file, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.application,
            'Save Curves',
            QtCore.QDir.homePath(),
            'CSV File (*.csv)'
        )

        if export_file == "":
            return

        progress_bar = self.start_progress_bar(2)
        progress_bar.show()
        progress_bar.setLabelText('Extracting Curves')

        names = [f for dataset in self.database.get_datasets() for f in (dataset.name, 'deltag', 'ln(rate)', 'rate', 'Enthalpy', 'Entropy', 'Heat Capacity')]
        curves = [Analysis.get_curves(dataset) for dataset in self.database.get_datasets()]
        curves_flat = [item for sublist in curves for item in sublist]
        curves_zip = zip(*curves_flat)

        progress_bar.setValue(1)
        progress_bar.setLabelText('Writing to file')

        with open(export_file, 'w', newline='') as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow(names)
            writer.writerows(curves_zip)

        progress_bar.setValue(2)

    def export_data(self)->None:
        export_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self.application, 
            "Save Directory", 
            QtCore.QDir.homePath(), 
            QtWidgets.QFileDialog.Option.ShowDirsOnly | QtWidgets.QFileDialog.Option.DontResolveSymlinks,

        )
        
        if export_dir == "":
            return

        out_path = self.make_output_directory(export_dir)
        
        database_length = self.database.get_length()
        progress_bar = self.start_progress_bar(database_length+2)
        progress_bar.show()

        self.export_graphs(out_path, progress_bar)
        progress_bar.setValue(database_length+1)
        progress_bar.setLabelText('Exporting Data Table')
        self.export_data_table(out_path)
        progress_bar.setValue(database_length+2)

    @staticmethod
    def make_output_directory(export_dir:Union[pathlib.Path, str])->Union[pathlib.Path, str]:
        now=datetime.now()
        date_str = now.strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(export_dir, f'MMRT_FittingTool_{date_str}')
        os.mkdir(out_path)
        return out_path

    def start_progress_bar(self, database_length:int)->QtWidgets.QProgressDialog:
        progress = QtWidgets.QProgressDialog("Starting Export", "", 0, database_length, self.application)
        progress.setCancelButton(None)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setWindowFlag(QtCore.Qt.WindowType.WindowCloseButtonHint, False)
        progress.setWindowFlag(QtCore.Qt.WindowType.WindowContextHelpButtonHint, False)
        progress.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, True)
        return progress

    def export_graphs(self, out_path:Union[pathlib.Path, str], progress_bar:QtWidgets.QProgressDialog)->None:
        database_length = self.database.get_length()
        
        join_fig, join_ax = plt.subplots(dpi=800, tight_layout=True, figsize=(6.4, 4.8))
        colormap = plt.get_cmap('gist_rainbow')

        join_ax.set_prop_cycle(
            color=[colormap(1.*i/database_length) for i in range(database_length)]
        )
        
        for idx, dataset in enumerate(self.database.get_datasets(), 1):
            progress_bar.setLabelText(f'Exporting {dataset.name}')
            if dataset.fit.mmrt2.mmrt2_fit is None:
                continue
            fig , ax = plt.subplots(dpi=800, tight_layout=True, figsize=(6.4, 4.8))
            
            dataset.fit.mmrt2.plot(ax)
            dataset.fit.mmrt2.plot(join_ax)

            out_fig_file = os.path.join(out_path, f'MFIT_{self.strip_name(dataset.name)}.png')
            
            count = 0
            while os.path.exists(out_fig_file):
                if count == 0:
                    out_fig_file = f'{out_fig_file}_0'
                else:
                    out_fig_file = out_fig_file[:-len(str(count))]
                    out_fig_file = f'{out_fig_file}_{count}'
                count += 1
            
            fig.savefig(out_fig_file, format='png', dpi=800, transparent=True)
            plt.close(fig)

            progress_bar.setValue(idx)
        progress_bar.setLabelText(f'Exporting Combined')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        join_fig.savefig(
            os.path.join(out_path, 'MFIT_MMRT2_All.png'),
            format='png', 
            dpi=800, 
            transparent=True,
            bbox_inches ="tight"
        )
        plt.close(join_fig)

    @staticmethod
    def strip_name(s:str)->str:
        s = re.sub(r"[^\w\s-]", '', s)
        # s = re.sub(r"\s+", '_', s)
        return s      

    @staticmethod
    def get_export_maps()->tuple[dict[str,str]]:
        mmrt2_params = {
            'H': 'Enthalpy', 
            'S': 'Entropy', 
            'ddH' : 'Enthalpy Difference',
            'CpL' : 'Heat Capacity (LowT)', 
            'CpH': 'Heat Capacity (HighT)',
            'Tc' : 'Transition Temperature',
            'Tn' : 'Reference Temperature'
        }

        stat_map = {
            'Number of Data Points': 'ndata',
            'Number of Variables': 'nvarys',
            'R-squared': 'rsq',
            'Adjusted R-squared': 'arsq',
            'AIC': 'aic',
            'AICc': 'aicc',
            'Chi-Squared': 'chi_sq',
            'Reduced Chi-Squared': 'chi_sq_reduced',
            'Bayesian Information Criterion': 'bic',
        }

        return mmrt2_params, stat_map

    def export_data_table(self, out_path:Union[pathlib.Path, str], filename=None)->None:       
        mmrt2_params, stat_map = self.get_export_maps()

        now=datetime.now()
        date_str = now.strftime('%Y%m%d_%H%M%S')
        
        with open(os.path.join(out_path, f'MFit_MMRT_{date_str}.csv'), 'w', newline='') as csvfile:  
            writer = csv.writer(csvfile)
            writer.writerow(['MMRT Fits'])
            writer.writerow([''])   

            self.write_csv(writer, 'MMRT 2.0', mmrt2_params, lambda dataset: dataset.fit.mmrt2, stat_map)

    def write_csv(self, writer:csv.writer, title:str, params:dict[str], fit_func:Callable, stat_map:dict)->None:
        writer.writerow([f'{title} Fit'])
        writer.writerow(['Datasets:'] + [dataset.name for dataset in self.database.get_datasets()])
        for param, label in params.items():
            writer.writerow([label] + [fit_func(dataset).values[param] if fit_func(dataset).mmrt2_fit is not None else 0 for dataset in self.database.get_datasets()]) 
        writer.writerow(['CpLowT range'] + [f'({fit_func(dataset).parameters.CpL_Point1.value}, {fit_func(dataset).parameters.CpL_Point2.value})' for dataset in self.database.get_datasets()])    
        writer.writerow([''])

        writer.writerow([f'{title} Error'])
        writer.writerow(['Datasets:'] + [dataset.name for dataset in self.database.get_datasets()])
        for param, label in params.items():
            writer.writerow([label] + [fit_func(dataset).error[param] if fit_func(dataset).mmrt2_fit is not None else 0 for dataset in self.database.get_datasets()])        
        writer.writerow([''])

        writer.writerow([f'{title} Fit Statistics'])
        writer.writerow(['Datasets:'] + [dataset.name for dataset in self.database.get_datasets()])
        for label, key in stat_map.items():
            writer.writerow([label] + [fit_func(dataset).fit_statistics[key] for dataset in self.database.get_datasets()])
        writer.writerow([''])

        writer.writerow([f'{title} 95% Confidence Interval'])
        writer.writerow(['Datasets:'] + [dataset.name for dataset in self.database.get_datasets()])
        for param, label in params.items():
            writer.writerow([label] + [fit_func(dataset).ci[param] if fit_func(dataset).mmrt2_fit is not None else 'None' for dataset in self.database.get_datasets()])        
        writer.writerow([''])


    def export_session(self)->None:
        file_out, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.application,
            'Save Session',
            QtCore.QDir.homePath(),
            '*.mmrt'
        )
        if file_out == '':
            return
        
        with open(file_out, 'w') as f:
            f.write(json.dumps({f'Dataset_{idx}': dataset.to_json() for (idx, dataset) in enumerate(self.database.get_datasets())}, indent=1))


class AnalysisLaunchHandler:
    currently_selected_dataset_key:int = None
    def __init__(self, application:'ApplicationWindow'):
        self.application = application
        self.get_ui_elements()
        self.link_slots()
        self.analysis.setDisabled(True)

    def get_ui_elements(self):
        self.analysis = self.application.base_ui_widget.actionAnalyse_Fit
        self.dataset_selected_signal = self.application.signals.dataset_selected_signal
        self.dg_t = self.application.base_ui_widget.actionDeltaG_vs_T
        self.ln_t = self.application.base_ui_widget.actionLn_rate_vs_T
        self.ln_in_t = self.application.base_ui_widget.actionLn_rate_vs_1_T
        self.rate_t = self.application.base_ui_widget.actionRate_vs_T

    def link_slots(self):
        self.analysis.triggered.connect(self.launch_analysis)
        self.dataset_selected_signal.connect(self.dataset_selected)
        self.dg_t.triggered.connect(self.change_rate_selection)
        self.ln_t.triggered.connect(self.change_rate_selection)
        self.ln_in_t.triggered.connect(self.change_rate_selection)
        self.rate_t.triggered.connect(self.change_rate_selection)

    def change_rate_selection(self):
        if not self.analysis.isChecked():
            return
        self.application.analysis_widget.ui.widget.clear_plot()
        self.launch_analysis()

    def dataset_selected(self, key:int):
        self.analysis.setEnabled(True)
        self.currently_selected_dataset_key = key

    def launch_analysis(self):
        if not self.analysis.isChecked():
            self.close_analysis()
            return
    
        self.application.change_main_widget(self.application.analysis_widget)

        for dataset_key in self.application.database.get_keys():
            if dataset_key == self.currently_selected_dataset_key:
                self.plot_dataset(self.application.database.get_dataset(dataset_key))
                continue
            if getattr(self.application.dataset_selector_handler, f'{dataset_key}_button').control_added:
                self.plot_dataset(self.application.database.get_dataset(dataset_key))

        self.application.analysis_widget.ui.widget.show_plot_labels(self.get_rate_selection())
        self.application.analysis_widget.ui.widget.fig.canvas.draw()

    def get_rate_selection(self):
        if self.dg_t.isChecked():
            return 'deltag'
        elif self.ln_t.isChecked():
            return 'lnrate'
        elif self.ln_in_t.isChecked():
            return 'inverse'
        elif self.rate_t.isChecked():
            return 'rate'

    def close_analysis(self):
        self.application.analysis_widget.ui.widget.clear_plot()
        self.application.change_main_widget(self.application.main_widget)

    def plot_dataset(self, dataset:'Dataset'):
        args = Analysis.get_args(dataset) 

        if args is None:
            return 

        x = np.linspace(dataset.inputs.temperature.min(), dataset.inputs.temperature.max(), 1000)
        g = Analysis.free_energy(x, args)
        h = Analysis.enthalpy(x, args)
        s = Analysis.entropy(x, args)
        c = Analysis.heat_capacity(x, args)
        lnk = Analysis.log_k(x, g)
        k = Analysis.rate(x, g)

        if self.dg_t.isChecked():
            Analysis.plot_deltag(self.application.analysis_widget.ui.widget.rate_axis, dataset, x, g)
        elif self.ln_t.isChecked():
            Analysis.plot_lnrate(self.application.analysis_widget.ui.widget.rate_axis, dataset, x, lnk)
        elif self.ln_in_t.isChecked():
            Analysis.plot_inverse(self.application.analysis_widget.ui.widget.rate_axis, dataset, x, lnk)
        elif self.rate_t.isChecked():
            Analysis.plot_rate(self.application.analysis_widget.ui.widget.rate_axis, dataset, x, k)

        Analysis.plot_enthalpy(self.application.analysis_widget.ui.widget.enthalpy_axis, dataset, x, h)
        Analysis.plot_entropy(self.application.analysis_widget.ui.widget.entropy_axis, dataset, x, s)
        Analysis.plot_heat_capacity(self.application.analysis_widget.ui.widget.heat_capacity_axis, dataset, x, c)