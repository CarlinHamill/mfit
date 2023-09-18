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

import sys
import argparse
# import pyi_splash
from PyQt6 import QtWidgets, QtCore, QtGui

import mfit.mfit1.handlers as handlers
from mfit.mfit1.data_tools import Database
from mfit.mfit1.custom_widgets import PageWidget, LayoutWidget
from mfit.mfit1.ui import application_base_ui, startup_ui, file_input_ui, main_screen_widget_ui,\
    mmrt1_parameter_value_display_widget_ui, mmrt1_set_parameter_widget_ui, \
    mmrt15_parameter_value_display_widget_ui, mmrt15_set_parameter_widget_ui, analysis_ui_widget_ui


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, parser:argparse.ArgumentParser, *args, **kwargs)->None:
        super(ApplicationWindow, self).__init__(*args, **kwargs)    
        self.set_up_base_ui_elements()
        self.create_widgets()
        self.set_up_sidebar()
        self.create_signals()
        self.create_database()
        self.create_handlers()
        
        self.setWindowTitle('MMRT Fitting Tool')
        self.startup_widget.show()
        self.show()

        self.handle_input_arguments(parser)

    def set_up_base_ui_elements(self)->None:
        self.base_ui_widget = application_base_ui.Ui_MainWindow()
        self.base_ui_widget.setupUi(self)
        self.main_layout = QtWidgets.QVBoxLayout(self.base_ui_widget.centralwidget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.action_group = QtGui.QActionGroup(self)
        self.action_group.addAction(self.base_ui_widget.actionLn_rate_vs_1_T)
        self.action_group.addAction(self.base_ui_widget.actionDeltaG_vs_T)
        self.action_group.addAction(self.base_ui_widget.actionLn_rate_vs_T)
        self.action_group.addAction(self.base_ui_widget.actionRate_vs_T)

    def create_widgets(self)->None:
        self.startup_widget = PageWidget(startup_ui, self)
        self.file_input_widget = PageWidget(file_input_ui, self)
        self.main_widget = PageWidget(main_screen_widget_ui, self)
        self.analysis_widget = PageWidget(analysis_ui_widget_ui, self)

    def create_signals(self)->None:
        self.signals = ApplicationSignals()

    def create_handlers(self)->None:
        self.widget_display_handler = handlers.WidgetDisplayHandler(self)
        self.file_input_handler = handlers.FileInputHandler(self)
        self.file_import_handler = handlers.FileImportHandler(self)
        self.file_fit_handler = handlers.FileFitHandler(self)
        self.dataset_add_handler = handlers.DatasetAddHandler(self)
        self.parameter_sidebar_handler = handlers.ParameterSideBarHandler(self)
        self.sidebar_widget_handler = handlers.SideBarWidgetDisplayHandler(self)
        self.dataset_selector_handler = handlers.DatasetSelectorHandler(self)
        self.graph_handler = handlers.GraphHandler(self)
        self.parameter_display_handler = handlers.ParameterDisplayHandler(self)
        self.parameter_edit_handler = handlers.ParameterEditHandler(self)
        self.dataset_change_handler = handlers.DatasetChangeHandler(self)
        self.remove_datapoint_handler = handlers.RemoveDatapointHandler(self)
        self.help_menu_handler = handlers.HelpMenuHandler(self)
        self.export_handdler = handlers.ExportHandler(self)
        self.calculator_handler = handlers.CalculatorHandler(self)
        self.analysis_handler = handlers.AnalysisLaunchHandler(self)
        
    def get_current_widget(self)->QtWidgets.QWidget:
        return [self.main_layout.itemAt(idx).widget() for idx in range(self.main_layout.count()) if not self.main_layout.itemAt(idx).widget().isHidden()][0]

    def change_main_widget(self, new_widget:QtWidgets.QWidget)->None:
        current_widget = self.get_current_widget()
        current_widget.hide()
        new_widget.show()

    def create_database(self)->None:
        self.database = Database()

    def set_up_sidebar(self):
        parameter_value_layout = QtWidgets.QVBoxLayout(self.main_widget.ui.frame_3)
        parameter_start_layout = QtWidgets.QVBoxLayout(self.main_widget.ui.frame_4)

        parameter_value_layout.setContentsMargins(0,0,0,0)
        parameter_start_layout.setContentsMargins(0,0,0,0)

        self.mmrt1_parameter_display_widget = LayoutWidget(mmrt1_parameter_value_display_widget_ui, parameter_value_layout)
        self.mmrt15_parameter_display_widget = LayoutWidget(mmrt15_parameter_value_display_widget_ui, parameter_value_layout)
        self.mmrt1_parameter_set_widget = LayoutWidget(mmrt1_set_parameter_widget_ui,  parameter_start_layout)
        self.mmrt15_parameter_set_widget = LayoutWidget(mmrt15_set_parameter_widget_ui,  parameter_start_layout)

    def handle_input_arguments(self, parser:argparse.ArgumentParser)->None:
        args = parser.parse_args()
        if not args.input_files:
            return
        self.file_input_widget.ui.frame.file_drop_event.emit(args.input_files)


class ApplicationSignals(QtCore.QObject):
    files_added_signal = QtCore.pyqtSignal(list)
    files_imported_signal = QtCore.pyqtSignal(list)
    files_fitted_signal = QtCore.pyqtSignal(list)
    datasets_added_signal = QtCore.pyqtSignal()
    dataset_selected_signal = QtCore.pyqtSignal(object)
    update_dataset_display_signal = QtCore.pyqtSignal()
    dataset_plot_add_signal = QtCore.pyqtSignal(object)
    update_fit_successes_signal = QtCore.pyqtSignal()
    name_changed_signal = QtCore.pyqtSignal(object)
    reset_dataset_buttons_signal = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs)->None:
        QtCore.QObject.__init__(self, *args, **kwargs)


def restart_application():
    QtCore.QCoreApplication.quit()
    QtCore.QProcess.startDetached(sys.executable, sys.argv)


def main():
    # pyi_splash.update_text('UI Loaded ...')
    # pyi_splash.close()

    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', nargs='*', type=str, help='input session file')
    
    app  = QtWidgets.QApplication(sys.argv)
    window = ApplicationWindow(parser)
    window.base_ui_widget.actionRestart.triggered.connect(restart_application)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()