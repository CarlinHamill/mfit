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

from PyQt6 import QtCore
from PyQt6.QtWidgets import QDialog

from mfit.mfit1.ui import about_dialog_ui, tinf_dialog_ui
import webbrowser

class AboutDialog(QDialog):
    def __init__(self, app, *args, **kwargs):
        super().__init__(app, *args, **kwargs)
        self.ui = about_dialog_ui.Ui_Dialog()
        self.ui.setupUi(self)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setWindowTitle('About')
        self.ui.pushButton.clicked.connect(lambda:webbrowser.open('https://www.gnu.org/licenses/gpl-3.0.en.html'))

class TinfDialog(QDialog):
    def __init__(self, app, *args, **kwargs):
        super().__init__(app, *args, **kwargs)
        self.ui = tinf_dialog_ui.Ui_Dialog()
        self.ui.setupUi(self)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowContextHelpButtonHint, False)
        self.ui.pushButton.clicked.connect(lambda:self.accept())