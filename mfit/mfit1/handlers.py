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

from mfit.mfit1.file_tools import ExcelInput, CSVInput, TabDelimitedInput, MMRTFileInput, SpaceDelimitedInput
from mfit.mfit1.custom_widgets import DatasetButton, UIWidget
from mfit.mfit1.ui import dataset_change_widget_ui
import mfit.mfit1.Analysis as Analysis
import mfit.mfit1.dialogs as dialogs

from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
import matplotlib.backend_bases
from datetime import datetime
import numpy as np
import webbrowser
import pathlib
import json
import math
import csv
import sys
import os
import re

from typing import TYPE_CHECKING, Union, Callable
if TYPE_CHECKING:
    from main import ApplicationWindow
    from data_tools import Dataset, FitMMRT1, FitMMRT15, Parameter


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
                #raise e
                import os
                print(os.getcwd())
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


class ParameterSideBarHandler:
    def __init__(self, application: 'ApplicationWindow')->None:
        self.application = application
        self.get_ui_elements()
        self.link_slots()
        self.change_parameter_mmrt_version()

    def get_ui_elements(self)->None:
        self.mmrt1_button = self.application.main_widget.ui.pushButton_3
        self.mmrt15_button = self.application.main_widget.ui.pushButton_6

        self.mmrt1_parameter_display_widget = self.application.mmrt1_parameter_display_widget
        self.mmrt15_parameter_display_widget = self.application.mmrt15_parameter_display_widget
        self.mmrt1_parameter_set_widget = self.application.mmrt1_parameter_set_widget
        self.mmrt15_parameter_set_widget = self.application.mmrt15_parameter_set_widget

        self.reset_dataset_buttons_signal = self.application.signals.reset_dataset_buttons_signal

    def link_slots(self)->None:
        self.mmrt1_button.clicked.connect(self.change_parameter_mmrt_version)
        self.mmrt15_button.clicked.connect(self.change_parameter_mmrt_version)
    
    def change_parameter_mmrt_version(self)->None:
        self.hide_parameter_widgets()
        self.reset_dataset_buttons_signal.emit()
        if self.mmrt1_button.isChecked():
            self.mmrt1_parameter_display_widget.show()
            self.mmrt1_parameter_set_widget.show()          
            return
        self.mmrt15_parameter_display_widget.show()
        self.mmrt15_parameter_set_widget.show()  

    def hide_parameter_widgets(self)->None:
        self.mmrt1_parameter_display_widget.hide()
        self.mmrt15_parameter_display_widget.hide()
        self.mmrt1_parameter_set_widget.hide()
        self.mmrt15_parameter_set_widget.hide()


class SideBarWidgetDisplayHandler:
    def __init__(self, application: 'ApplicationWindow')->None:
        self.application = application
        self.get_ui_elements()
        self.link_slots()
        self.change_display_widget()

    def get_ui_elements(self)->None:
        self.parameter_button = self.application.main_widget.ui.pushButton_4
        self.dataset_edit_button = self.application.main_widget.ui.pushButton_5
        self.parameter_widget = self.application.main_widget.ui.widget_2
        self.dataset_edit_widget = self.application.main_widget.ui.scrollArea_3

    def link_slots(self)->None:
        self.dataset_edit_button.clicked.connect(self.change_display_widget)
        self.parameter_button.clicked.connect(self.change_display_widget)

    def change_display_widget(self)->None:
        self.hide_widgets()
        if self.dataset_edit_button.isChecked():
            self.dataset_edit_widget.show()
            return
        self.parameter_widget.show()
    
    def hide_widgets(self)->None:
        self.parameter_widget.hide()
        self.dataset_edit_widget.hide()


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
            button.setStatusTip('Click to select dataset. Control-click to add to plot.')
            button.toggled.connect(self.dataset_button_toggled)
            button.control_click_event_signal.connect(self.control_button_clicked)
            self.button_group.addButton(button)
        
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
            if self.get_mmrt_fit_selected(dataset) is None:
                self.set_stylesheet(button, 'rgb(255,0,0)')#red
                continue  
            self.set_stylesheet(button, 'rgb(0,0,0)')#black

    def get_mmrt_fit_selected(self, dataset:'Dataset')->Union['FitMMRT1', 'FitMMRT15']:
        if self.mmrt1_button.isChecked():
            return dataset.fit.mmrt1
        return dataset.fit.mmrt15

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
            f"   color: {colour};\n"
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

    def change_name(self)->None:
        for (key, dataset) in self.database.get_items():
            getattr(self, f'{key}_button').setText(dataset.name)


class GraphHandler:
    current_selected_dataset_key:int = None
    def __init__(self, application:'ApplicationWindow')->None:
        self.application = application

        self.get_ui_elements()
        self.link_slots()

    def get_ui_elements(self)->None:
        self.fig = self.application.main_widget.ui.page
        self.database = self.application.database
        self.dataset_selected_signal = self.application.signals.dataset_selected_signal
        self.mmrt1_button = self.application.main_widget.ui.pushButton_3
        self.update_dataset_display_signal = self.application.signals.update_dataset_display_signal
        self.dataset_plot_add_signal = self.application.signals.dataset_plot_add_signal

    def link_slots(self)->None:
        self.dataset_selected_signal.connect(self.dataset_selected_changed)
        self.mmrt1_button.toggled.connect(self.plot_dataset)
        self.update_dataset_display_signal.connect(self.plot_dataset)
        self.dataset_plot_add_signal.connect(self.add_plot)

    def dataset_selected_changed(self, key:int)->None:
        self.current_selected_dataset_key = key
        self.plot_dataset()

    def add_plot(self, key:int)->None:
        if key == self.current_selected_dataset_key:
            return
        dataset = self.database.get_dataset(key)
        fit = self.get_mmrt_fit_selected(dataset)
        if fit is None:
            self.fig.add_plot(dataset.plot)
            return 
        self.fig.add_plot(fit.plot)

    def plot_dataset(self)->None:
        if self.current_selected_dataset_key is None:
            return
        self.fig.clear_plot()
        dataset = self.database.get_dataset(self.current_selected_dataset_key)
        fit = self.get_mmrt_fit_selected(dataset)

        if fit is None:
            self.fig.add_plot(dataset.plot)
            return 
        self.fig.add_plot(fit.plot)
    
    def get_mmrt_fit_selected(self, dataset:'Dataset')->Union['FitMMRT1', 'FitMMRT15']:
        if self.mmrt1_button.isChecked():
            return dataset.fit.mmrt1
        return dataset.fit.mmrt15


class ParameterDisplayHandler:
    current_selected_dataset_key:int = None
    def __init__(self, application:'ApplicationWindow')->None:
        self.application = application

        self.get_ui_elements()
        self.link_slots()

    def get_ui_elements(self)->None:
        self.database = self.application.database
        self.dataset_selected_signal = self.application.signals.dataset_selected_signal
        self.mmrt1_widget = self.application.mmrt1_parameter_display_widget.ui
        self.mmrt15_widget = self.application.mmrt15_parameter_display_widget.ui
        self.update_dataset_display_signal = self.application.signals.update_dataset_display_signal

    def link_slots(self)->None:
        self.dataset_selected_signal.connect(self.dataset_selected_changed)
        self.update_dataset_display_signal.connect(self.show_dataset_parameters)

    def dataset_selected_changed(self, key:int)->None:
        self.current_selected_dataset_key = key
        self.show_dataset_parameters()

    def show_dataset_parameters(self)->None:
        if self.current_selected_dataset_key is None:
            return
        
        dataset = self.database.get_dataset(self.current_selected_dataset_key)

        mmrt1_parameter_lineedit_map = {
            'H': [self.mmrt1_widget.enthalpyLineEdit , self.mmrt1_widget.lineEdit],
            'S': [self.mmrt1_widget.entropyLineEdit , self.mmrt1_widget.lineEdit_2],
            'C': [self.mmrt1_widget.heatCapacityUpperLineEdit , self.mmrt1_widget.lineEdit_6],
            'tn': [self.mmrt1_widget.tnLineEdit , self.mmrt1_widget.lineEdit_7]
        }

        mmrt15_parameter_lineedit_map = {
            'H': [self.mmrt15_widget.enthalpyLineEdit , self.mmrt15_widget.lineEdit],
            'S': [self.mmrt15_widget.entropyLineEdit , self.mmrt15_widget.lineEdit_2],
            'M': [self.mmrt15_widget.enthalpyDifferenceLineEdit , self.mmrt15_widget.lineEdit_3],
            'C': [self.mmrt15_widget.heatCapacityUpperLineEdit , self.mmrt15_widget.lineEdit_6],
            'tn': [self.mmrt15_widget.tnLineEdit , self.mmrt15_widget.lineEdit_7]
        }

        self.write_parameters(dataset.fit.mmrt1, mmrt1_parameter_lineedit_map)
        self.write_parameters(dataset.fit.mmrt15, mmrt15_parameter_lineedit_map)

    def write_parameters(self, dataset_fit:Union['FitMMRT1', 'FitMMRT15', None], lineedit_map:dict[str, list[QtWidgets.QLineEdit]])->None:
        if dataset_fit is None:
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
        self.application.mmrt1_parameter_set_widget.setEnabled(enabled)
        self.application.mmrt15_parameter_set_widget.setEnabled(enabled)
   
    def set_validators(self)->None: 
        lineedits = [
            self.mmrt1_widget.enthalpyLineEdit,
            self.mmrt1_widget.entropyLineEdit,
            self.mmrt1_widget.heatCapacityUpperLineEdit,
            self.mmrt1_widget.tnLineEdit,
            self.mmrt15_widget.enthalpyLineEdit,
            self.mmrt15_widget.entropyLineEdit,
            self.mmrt15_widget.enthalpyDifferenceLineEdit,
            self.mmrt15_widget.heatCapacityUpperLineEdit,
            self.mmrt15_widget.tnLineEdit
        ]
        validator = QtGui.QDoubleValidator()

        for lineedit in lineedits:
            lineedit.setValidator(validator)
    
    def get_ui_elements(self)->None:
        self.database = self.application.database
        self.dataset_selected_signal = self.application.signals.dataset_selected_signal
        
        self.mmrt1_widget = self.application.mmrt1_parameter_set_widget.ui
        self.mmrt15_widget = self.application.mmrt15_parameter_set_widget.ui

        self.update_dataset_display_signal = self.application.signals.update_dataset_display_signal
        self.update_fit_success_signal = self.application.signals.update_fit_successes_signal

        self.reset_dataset_buttons_signal = self.application.signals.reset_dataset_buttons_signal

    def link_slots(self)->None:
        self.dataset_selected_signal.connect(self.dataset_selected_changed)
        self.enable_link_slots()

    def dataset_selected_changed(self, key:int)->None:
        self.current_selected_dataset_key = key
        self.enable_editing(True)
        self.disable_link_slots()
        self.write_starting_parameters()
        self.enable_link_slots()

    @staticmethod
    def get_combo_index(variable:bool)->int:
        if variable:
            return 0
        return 1
    
    def get_param_objects(self, dataset:'Dataset')->list[tuple[QtWidgets.QComboBox, 'Parameter', QtWidgets.QLineEdit]]:
        mmrt1_paramater_objects = [
            (self.mmrt1_widget.comboBox_7, dataset.parameters.mmrt1.H, self.mmrt1_widget.enthalpyLineEdit,),
            (self.mmrt1_widget.comboBox_6, dataset.parameters.mmrt1.S, self.mmrt1_widget.entropyLineEdit),
            (self.mmrt1_widget.comboBox_4, dataset.parameters.mmrt1.C, self.mmrt1_widget.heatCapacityUpperLineEdit),
            (self.mmrt1_widget.comboBox_3, dataset.parameters.mmrt1.tn, self.mmrt1_widget.tnLineEdit),
        ]        
        mmrt15_paramater_objects = [
            (self.mmrt15_widget.comboBox_7, dataset.parameters.mmrt15.H, self.mmrt15_widget.enthalpyLineEdit,),
            (self.mmrt15_widget.comboBox_6, dataset.parameters.mmrt15.S, self.mmrt15_widget.entropyLineEdit),
            (self.mmrt15_widget.comboBox_5, dataset.parameters.mmrt15.C, self.mmrt15_widget.enthalpyDifferenceLineEdit),
            (self.mmrt15_widget.comboBox_4, dataset.parameters.mmrt15.M, self.mmrt15_widget.heatCapacityUpperLineEdit),
            (self.mmrt15_widget.comboBox_3, dataset.parameters.mmrt15.tn, self.mmrt15_widget.tnLineEdit),
        ]        
        return mmrt1_paramater_objects, mmrt15_paramater_objects 
       
    def write_starting_parameters(self):
        dataset = self.database.get_dataset(self.current_selected_dataset_key)
        for fit_version_parameter_objects in self.get_param_objects(dataset):
            for (combo_box, param, line_edit) in fit_version_parameter_objects:
                combo_box.setCurrentIndex(self.get_combo_index(param.variable))
                line_edit.setText(str(round(param.value, 4)))

    def disable_link_slots(self)->None:
        # Combobox index changed
        self.mmrt1_widget.comboBox_7.currentIndexChanged.disconnect()
        self.mmrt1_widget.comboBox_6.currentIndexChanged.disconnect()
        self.mmrt1_widget.comboBox_4.currentIndexChanged.disconnect()
        self.mmrt15_widget.comboBox_7.currentIndexChanged.disconnect()
        self.mmrt15_widget.comboBox_6.currentIndexChanged.disconnect()
        self.mmrt15_widget.comboBox_5.currentIndexChanged.disconnect()
        self.mmrt15_widget.comboBox_4.currentIndexChanged.disconnect()
        
        # Lineedit edited
        self.mmrt1_widget.enthalpyLineEdit.textChanged.disconnect()
        self.mmrt1_widget.entropyLineEdit.textChanged.disconnect()
        self.mmrt1_widget.heatCapacityUpperLineEdit.textChanged.disconnect()
        self.mmrt1_widget.tnLineEdit.textChanged.disconnect()
        self.mmrt15_widget.enthalpyLineEdit.textChanged.disconnect()
        self.mmrt15_widget.entropyLineEdit.textChanged.disconnect()
        self.mmrt15_widget.enthalpyDifferenceLineEdit.textChanged.disconnect()
        self.mmrt15_widget.heatCapacityUpperLineEdit.textChanged.disconnect()
        self.mmrt15_widget.tnLineEdit.textChanged.disconnect()

        # Lineedit editing finished
        self.mmrt1_widget.enthalpyLineEdit.editingFinished.disconnect()
        self.mmrt1_widget.entropyLineEdit.editingFinished.disconnect()
        self.mmrt1_widget.heatCapacityUpperLineEdit.editingFinished.disconnect()
        self.mmrt1_widget.tnLineEdit.editingFinished.disconnect()

        self.mmrt15_widget.enthalpyLineEdit.editingFinished.disconnect()
        self.mmrt15_widget.entropyLineEdit.editingFinished.disconnect()
        self.mmrt15_widget.enthalpyDifferenceLineEdit.editingFinished.disconnect()
        self.mmrt15_widget.heatCapacityUpperLineEdit.editingFinished.disconnect()
        self.mmrt15_widget.tnLineEdit.editingFinished.disconnect()

    def enable_link_slots(self)->None:
        # Combobox index changed
        self.mmrt1_widget.comboBox_7.currentIndexChanged.connect(lambda: self.combo_edited(self.mmrt1_widget.comboBox_7, 'mmrt1', 'H'))
        self.mmrt1_widget.comboBox_6.currentIndexChanged.connect(lambda: self.combo_edited(self.mmrt1_widget.comboBox_6, 'mmrt1', 'S'))
        self.mmrt1_widget.comboBox_4.currentIndexChanged.connect(lambda: self.combo_edited(self.mmrt1_widget.comboBox_4, 'mmrt1', 'C'))
        self.mmrt15_widget.comboBox_7.currentIndexChanged.connect(lambda: self.combo_edited(self.mmrt15_widget.comboBox_7, 'mmrt15', 'H'))
        self.mmrt15_widget.comboBox_6.currentIndexChanged.connect(lambda: self.combo_edited(self.mmrt15_widget.comboBox_6, 'mmrt15', 'S'))
        self.mmrt15_widget.comboBox_5.currentIndexChanged.connect(lambda: self.combo_edited(self.mmrt15_widget.comboBox_5, 'mmrt15', 'C'))
        self.mmrt15_widget.comboBox_4.currentIndexChanged.connect(lambda: self.combo_edited(self.mmrt15_widget.comboBox_4, 'mmrt15', 'M'))
        
        # Lineedit edited
        self.mmrt1_widget.enthalpyLineEdit.textChanged.connect(lambda: self.lineedit_edited(self.mmrt1_widget.enthalpyLineEdit))
        self.mmrt1_widget.entropyLineEdit.textChanged.connect(lambda: self.lineedit_edited(self.mmrt1_widget.entropyLineEdit))
        self.mmrt1_widget.heatCapacityUpperLineEdit.textChanged.connect(lambda: self.lineedit_edited(self.mmrt1_widget.heatCapacityUpperLineEdit))
        self.mmrt1_widget.tnLineEdit.textChanged.connect(lambda: self.lineedit_edited(self.mmrt1_widget.tnLineEdit))
        self.mmrt15_widget.enthalpyLineEdit.textChanged.connect(lambda: self.lineedit_edited(self.mmrt15_widget.enthalpyLineEdit))
        self.mmrt15_widget.entropyLineEdit.textChanged.connect(lambda: self.lineedit_edited(self.mmrt15_widget.entropyLineEdit))
        self.mmrt15_widget.enthalpyDifferenceLineEdit.textChanged.connect(lambda: self.lineedit_edited(self.mmrt15_widget.enthalpyDifferenceLineEdit))
        self.mmrt15_widget.heatCapacityUpperLineEdit.textChanged.connect(lambda: self.lineedit_edited(self.mmrt15_widget.heatCapacityUpperLineEdit))
        self.mmrt15_widget.tnLineEdit.textChanged.connect(lambda: self.lineedit_edited(self.mmrt15_widget.tnLineEdit))

        # Lineedit editing finished
        self.mmrt1_widget.enthalpyLineEdit.editingFinished.connect(lambda: self.linedit_finished(self.mmrt1_widget.enthalpyLineEdit, 'mmrt1', 'H'))
        self.mmrt1_widget.entropyLineEdit.editingFinished.connect(lambda: self.linedit_finished(self.mmrt1_widget.entropyLineEdit, 'mmrt1', 'S'))
        self.mmrt1_widget.heatCapacityUpperLineEdit.editingFinished.connect(lambda: self.linedit_finished(self.mmrt1_widget.heatCapacityUpperLineEdit, 'mmrt1', 'C'))
        self.mmrt1_widget.tnLineEdit.editingFinished.connect(lambda: self.linedit_finished(self.mmrt1_widget.tnLineEdit, 'mmrt1', 'tn'))

        self.mmrt15_widget.enthalpyLineEdit.editingFinished.connect(lambda: self.linedit_finished(self.mmrt15_widget.enthalpyLineEdit, 'mmrt15', 'H'))
        self.mmrt15_widget.entropyLineEdit.editingFinished.connect(lambda: self.linedit_finished(self.mmrt15_widget.entropyLineEdit, 'mmrt15', 'S'))
        self.mmrt15_widget.enthalpyDifferenceLineEdit.editingFinished.connect(lambda: self.linedit_finished(self.mmrt15_widget.enthalpyDifferenceLineEdit, 'mmrt15', 'C'))
        self.mmrt15_widget.heatCapacityUpperLineEdit.editingFinished.connect(lambda: self.linedit_finished(self.mmrt15_widget.heatCapacityUpperLineEdit, 'mmrt15', 'M'))
        self.mmrt15_widget.tnLineEdit.editingFinished.connect(lambda: self.linedit_finished(self.mmrt15_widget.tnLineEdit, 'mmrt15', 'tn'))
    
    def combo_edited(self, combobox:QtWidgets.QComboBox, fit_version:str, param:str)->None:
        dataset = self.database.get_dataset(self.current_selected_dataset_key)
        version_params = getattr(dataset.parameters, fit_version)
        parameter = getattr(version_params, param)    
        
        if combobox.currentIndex() == 0:#Free
            parameter.variable = True
        else:
            parameter.variable = False

        self.refit_mmrt(dataset)

    def lineedit_edited(self, lineedit:QtWidgets.QLineEdit)->None:
        lineedit.setStyleSheet("color: red;")

    def linedit_finished(self, lineedit:QtWidgets.QLineEdit, fit_version:str, param:str)->None:
        lineedit.setStyleSheet("color: black;")        
        dataset = self.database.get_dataset(self.current_selected_dataset_key)
        version_params = getattr(dataset.parameters, fit_version)
        parameter = getattr(version_params, param)   

        new_param = lineedit.text()
        if new_param == '':
            new_param = '0'

        parameter.value = float(new_param)

        self.refit_mmrt(dataset)

    def refit_mmrt(self, dataset:'Dataset')->None:
        dataset.fit_mmrt()
        self.update_dataset_display_signal.emit()
        self.update_fit_success_signal.emit()
        self.reset_dataset_buttons_signal.emit()


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
        self.update_fit_success_signal.emit()
        self.update_dataset_display_signal.emit()
        self.reset_dataset_buttons_signal.emit()
    
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

        self.enable_export(False)
    
    def get_ui_elements(self)->None:
        self.export_session_action = self.application.base_ui_widget.actionSession
        self.database = self.application.database
        self.export_data_action = self.application.base_ui_widget.actionData
        self.datasets_added_signal = self.application.signals.datasets_added_signal
        self.export_curve_action = self.application.base_ui_widget.actionCurve
        self.export_csv_action = self.application.base_ui_widget.actionCSV

    def link_slots(self)->None:
        self.datasets_added_signal.connect(lambda: self.enable_export(True))       
        self.export_session_action.triggered.connect(self.export_session)
        self.export_data_action.triggered.connect(self.export_data)
        self.export_curve_action.triggered.connect(self.export_curve)
        self.export_csv_action.triggered.connect(self.export_csv)
    
    def enable_export(self, enabled:bool)->None:
        self.export_data_action.setEnabled(enabled)
        self.export_session_action.setEnabled(enabled)
        self.export_curve_action.setEnabled(enabled)
        self.export_csv_action.setEnabled(enabled)

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

        mmrt1_params, mmrt15_params, stat_map = self.get_export_maps()

        progress_bar.setValue(1)

        with open(export_file, 'w', newline='') as csvfile:  
            writer = csv.writer(csvfile)
            writer.writerow(['MMRT Fits'])
            writer.writerow([''])   

            self.write_csv(writer, 'MMRT 1.0', mmrt1_params, lambda dataset: dataset.fit.mmrt1, stat_map)
            writer.writerow([''])
            self.write_csv(writer, 'MMRT 1.5', mmrt15_params, lambda dataset: dataset.fit.mmrt15, stat_map)

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
        headers = [
            'deltag mmrt 1.0', 'ln(rate) mmrt 1.0', 'rate mmrt 1.0', 
            'Enthalpy mmrt 1.0', 'Entropy mmrt 1.0', 'Heat Capacity mmrt 1.0',
            'deltag mmrt 1.5', 'ln(rate) mmrt 1.5', 'rate mmrt 1.5', 
            'Enthalpy mmrt 1.5', 'Entropy mmrt 1.5', 'Heat Capacity mmrt 1.5'
        ]
        names = [f for dataset in self.database.get_datasets() for f in (dataset.name, *headers)]
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
    
    def start_progress_bar(self, progress_length:int)->QtWidgets.QProgressDialog:
        progress = QtWidgets.QProgressDialog("Starting Export", "", 0, progress_length, self.application)
        progress.setCancelButton(None)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setWindowFlag(QtCore.Qt.WindowType.WindowCloseButtonHint, False)
        progress.setWindowFlag(QtCore.Qt.WindowType.WindowContextHelpButtonHint, False)
        progress.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, True)
        return progress

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

    def export_graphs(self, out_path:Union[pathlib.Path, str], progress_bar:QtWidgets.QProgressDialog)->None:
        database_length = self.database.get_length()
        
        mmrt1_join_fig, mmrt1_join_ax = plt.subplots(dpi=800, tight_layout=True, figsize=(6.4, 4.8))
        mmrt15_join_fig, mmrt15_join_ax = plt.subplots(dpi=800, tight_layout=True, figsize=(6.4, 4.8))
        colormap = plt.get_cmap('gist_rainbow')

        mmrt1_join_ax.set_prop_cycle(
            color=[colormap(1.*i/database_length) for i in range(database_length)]
        )
        mmrt15_join_ax.set_prop_cycle(
            color=[colormap(1.*i/database_length) for i in range(database_length)]
        )    

        for idx, dataset in enumerate(self.database.get_datasets(), 1):
            progress_bar.setLabelText(f'Exporting {dataset.name}')
            # Export MMRT 1
            if dataset.fit.mmrt1 is not None:
                fig , ax = plt.subplots(dpi=800, tight_layout=True, figsize=(6.4, 4.8))
                
                dataset.fit.mmrt1.plot(ax)
                dataset.fit.mmrt1.plot(mmrt1_join_ax)

                out_fig_file = os.path.join(out_path, f'MFIT_mmrt1_{self.strip_name(dataset.name)}')
                
                count = 0
                while os.path.exists(f'{out_fig_file}.png'):
                    if count == 0:
                        out_fig_file = f'{out_fig_file}_0'
                    else:
                        out_fig_file = out_fig_file[:-len(str(count))]
                        out_fig_file = f'{out_fig_file}_{count}'
                    count += 1
                
                fig.savefig(f'{out_fig_file}.png', format='png', dpi=800, transparent=True)
                plt.close(fig)
            # Export MMRT 1.5
            if dataset.fit.mmrt15 is not None:
                fig , ax = plt.subplots(dpi=800, tight_layout=True, figsize=(6.4, 4.8))
                
                dataset.fit.mmrt15.plot(ax)
                dataset.fit.mmrt15.plot(mmrt15_join_ax)

                out_fig_file = os.path.join(out_path, f'MFIT_mmrt15_{self.strip_name(dataset.name)}')
                
                count = 0
                while os.path.exists(f'{out_fig_file}.png'):
                    if count == 0:
                        out_fig_file = f'{out_fig_file}_0'
                    else:
                        out_fig_file = out_fig_file[:-len(str(count))]
                        out_fig_file = f'{out_fig_file}_{count}'
                    count += 1
                
                fig.savefig(f'{out_fig_file}.png', format='png', dpi=800, transparent=True)
                plt.close(fig)

            progress_bar.setValue(idx)
        progress_bar.setLabelText(f'Exporting Combined')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        mmrt1_join_fig.savefig(
            os.path.join(out_path, 'MFIT_MMRT1_All.png'),
            format='png', 
            dpi=800, 
            transparent=True,
            bbox_inches ="tight"
        )
        plt.close(mmrt1_join_fig)
        mmrt15_join_fig.savefig(
            os.path.join(out_path, 'MFIT_MMRT15_All.png'),
            format='png', 
            dpi=800, 
            transparent=True,
            bbox_inches ="tight"
        )
        plt.close(mmrt15_join_fig)

    @staticmethod
    def strip_name(s:str)->str:
        s = re.sub(r"[^\w\s-]", '', s)
        # s = re.sub(r"\s+", '_', s)
        return s  

    def get_export_maps(self)->tuple[dict[str,str]]:
        mmrt1_params = {
            'H': 'Enthalpy', 
            'S': 'Entropy', 
            'C' : 'Heat Capacity', 
            'tn' : 'Reference Temperature'
        }
        mmrt15_params = {
            'H': 'Enthalpy', 
            'S': 'Entropy', 
            'C' : 'Heat Capacity Intercept', 
            'M': 'Heat Capacity Slope',
            'tn': 'Reference Temperature'
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
            'Temperature optimum': 'topt',
            'Inflection Temperature': 'tinf'
        }   

        return (mmrt1_params, mmrt15_params, stat_map)

    def export_data_table(self, out_path:Union[pathlib.Path, str])->None:       
        mmrt1_params, mmrt15_params, stat_map = self.get_export_maps()     
        now=datetime.now()
        date_str = now.strftime('%Y%m%d_%H%M%S')
        
        with open(os.path.join(out_path, f'MFit_MMRT_{date_str}.csv'), 'w', newline='') as csvfile:  
            writer = csv.writer(csvfile)
            writer.writerow(['MMRT Fits'])
            writer.writerow([''])   

            self.write_csv(writer, 'MMRT 1.0', mmrt1_params, lambda dataset: dataset.fit.mmrt1, stat_map)
            writer.writerow([''])
            self.write_csv(writer, 'MMRT 1.5', mmrt15_params, lambda dataset: dataset.fit.mmrt15, stat_map)

    def write_csv(self, writer:csv.writer, title:str, params:dict[str,str], fit_func:Callable, stat_map:dict[str,str]):
        writer.writerow([f'{title} Fit'])
        writer.writerow(['Datasets:'] + [dataset.name for dataset in self.database.get_datasets()])
        for param, label in params.items():
            writer.writerow([label] + [fit_func(dataset).values[param] if fit_func(dataset) is not None else 0 for dataset in self.database.get_datasets()])          
        writer.writerow([''])

        writer.writerow([f'{title} Error'])
        writer.writerow(['Datasets:'] + [dataset.name for dataset in self.database.get_datasets()])
        for param, label in params.items():
            writer.writerow([label] + [fit_func(dataset).error[param] if fit_func(dataset) is not None else 0 for dataset in self.database.get_datasets()])    
        writer.writerow(['AIC'] + [fit_func(dataset).error['aic'] if fit_func(dataset) is not None else 0 for dataset in self.database.get_datasets()])      
        writer.writerow([''])

        writer.writerow([f'{title} Fit Statistics'])
        writer.writerow(['Datasets:'] + [dataset.name for dataset in self.database.get_datasets()])
        for label, key in stat_map.items():
            writer.writerow([label] + [fit_func(dataset).fit_statistics[key] for dataset in self.database.get_datasets()])
        writer.writerow([''])

        writer.writerow([f'{title} 95% Confidence Interval'])
        writer.writerow(['Datasets:'] + [dataset.name for dataset in self.database.get_datasets()])
        for param, label in params.items():
            writer.writerow([label] + [fit_func(dataset).ci[param] if fit_func(dataset) is not None else 'None' for dataset in self.database.get_datasets()])        
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
            f.write(json.dumps({f'Dataset_{idx}': dataset.to_json() for (idx, dataset) in enumerate(self.database.get_datasets())}))

        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Icon.Information)
        msg.setText("Session Exported")
        msg.setInformativeText("Session was successfully exported")
        msg.setWindowTitle("Session Exported")
        msg.exec()


class CalculatorHandler:
    current_selected_dataset_key:int = None
    def __init__(self, application:'ApplicationWindow')->None:
        self.application = application

        self.get_ui_elements()
        self.link_slots()

        self.calculate_action.setDisabled(True)
    
    def get_ui_elements(self)->None:
        self.calculate_action = self.application.base_ui_widget.actionCalculate_Topt_Tinf
        self.database = self.application.database
        self.dataset_selected_signal = self.application.signals.dataset_selected_signal
        self.datasets_added_signal = self.application.signals.datasets_added_signal
    
    def link_slots(self)->None:
        self.calculate_action.triggered.connect(self.calculate_topt_tinf)
        self.dataset_selected_signal.connect(lambda:self.calculate_action.setEnabled(True))
        self.dataset_selected_signal.connect(self.dataset_selection_changed)
        
    def dataset_selection_changed(self, key:int)->None:
        self.current_selected_dataset_key = key

    def calculate_topt_tinf(self):
        if self.current_selected_dataset_key is None:
            return
        
        
        mmrt1_topt = 0
        mmrt1_tinf = 0
        mmrt15_topt_1 = 0
        mmrt15_topt_2 = 0
        mmrt15_tinf = 0


        mmrt1 = self.database.get_dataset(self.current_selected_dataset_key).fit.mmrt1
        tn = mmrt1.mmrt1_parameters.tn.value
        h = mmrt1.values['H']
        c = mmrt1.values['C']
        r = 8.314
        if  mmrt1 is not None:
            mmrt1_topt = (c*tn - h)/(c + r)
            mmrt1_tinf = (h-c*tn)/(-c+math.sqrt(-c*r))

        mmrt15 = self.database.get_dataset(self.current_selected_dataset_key).fit.mmrt15
        tn = mmrt15.mmrt15_parameters.tn.value
        h = mmrt15.values['H']
        c = mmrt15.values['C']
        m = mmrt15.values['M']
        r=8.314
        if  mmrt15 is not None:
            mmrt15_topt_1 = -(c + r)/m + math.sqrt(c**2 + 2*c*m*tn + 2*c*r - 2*h*m + m**2*tn**2 + r**2)/m
            mmrt15_topt_2 = -(c + r)/m - math.sqrt(c**2 + 2*c*m*tn + 2*c*r - 2*h*m + m**2*tn**2 + r**2)/m
            mmrt15_tinf = (2*c*tn-2*h+m*(tn**2))/(c+r)

        dialog = dialogs.TinfDialog(self.application)
        
        dialog.ui.label.setText(f'Dataset: {self.database.get_dataset(self.current_selected_dataset_key).name}')
        dialog.ui.topt1.setText(f'{mmrt1_topt:.2f}')
        dialog.ui.tinf1.setText(f'{mmrt1_tinf:.2f}')

        dialog.ui.topt15.setText(f'{mmrt15_topt_1:.2f}/ {mmrt15_topt_2:.2f}')
        dialog.ui.tinf15.setText(f'{mmrt15_tinf:.2f}')
        dialog.exec()


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
        if self.application.main_widget.ui.pushButton_3.isChecked():
            args = Analysis.get_mmrt1_args(dataset)
        else:
            args = Analysis.get_mmrt15_args(dataset) 

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