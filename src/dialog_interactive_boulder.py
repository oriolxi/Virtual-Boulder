import os
import pickle
from PyQt6 import uic
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QDialog, QListWidgetItem, QFileDialog

import util
from boulder import renderBoulderPreview, Boulder

DEFAULT_BOULDER_NAME = "default"

class InteractiveBoulderDialog(QDialog):
    signal_start = pyqtSignal(int)
    signal_edit = pyqtSignal(int)

    def __init__(self, parent, boulder_list, holds_bboxes, ref_img):
        super().__init__(parent)
        uic.loadUi("gui_interactive_boulder.ui", self)

        self.boulders = boulder_list
        self.holds = holds_bboxes
        self.reference_img = ref_img
        for boulder in self.boulders:
            item = QListWidgetItem(boulder.getName())
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.lst_boulder_list.addItem(item)
        
        self.__setUpGui()
        self.lst_boulder_list.setCurrentRow(0)

    def updateBoulderPreview(self):
        idx = self.lst_boulder_list.currentRow()
        if (idx < 0 or idx >= len(self.boulders)): 
            self.lst_boulder_list.setCurrentRow(0)
            idx = 0
        img = renderBoulderPreview(self.boulders[idx], self.holds, util.QimageFromCVimage(self.reference_img))
        self.lbl_preview_boulder.setPixmap(img.scaled(self.lbl_preview_boulder.size().width(), self.lbl_preview_boulder.size().height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.lbl_preview_boulder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.boulders[idx].setName(self.lst_boulder_list.currentItem().text())

    def __setUpGui(self):
        self.btn_start_boulder.clicked.connect(self.__startBoulder)
        self.btn_edit_boulder.clicked.connect(self.__editBoulder)
        self.btn_load_boulder.clicked.connect(self.__loadBoulder)
        self.btn_save_boulder.clicked.connect(self.__saveBoulder)
        self.btn_new_boulder.clicked.connect(self.__newBoulder)
        self.btn_delete_boulder.clicked.connect(self.__deleteBoulder)
        self.lst_boulder_list.currentItemChanged.connect(self.updateBoulderPreview)
        #self.lst_boulder_list.currentItemChanged

    def __updateBoulderName(self):
        pass

    def __startBoulder(self):
        self.signal_start.emit(self.lst_boulder_list.currentRow())

    def __editBoulder(self):
        self.signal_edit.emit(self.lst_boulder_list.currentRow())

    def __loadBoulder(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("Boulder file (*.bldr)")
        dialog.setDirectory('./saves/boulders')
        dialog.setViewMode(QFileDialog.ViewMode.Detail)
        if dialog.exec():
            fileNames = dialog.selectedFiles()
        if os.path.isfile(fileNames[0]):
            b = Boulder()
            with open(fileNames[0], 'rb') as inp:
                b.setSteps(pickle.load(inp))
            name = os.path.basename(fileNames[0]).split(".")[0]
            b.setName(name)
            self.boulders.append(b)
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.lst_boulder_list.addItem(item)
            self.lst_boulder_list.setCurrentRow(self.lst_boulder_list.count() -1)

    def __saveBoulder(self):
        directory = QFileDialog.getExistingDirectory(self, "Open Directory","./saves/boulders")
        if directory != '':
            with open(directory + '/' + self.lst_boulder_list.currentItem().text() + '.bldr', 'wb') as output:
                pickle.dump(self.boulders[self.lst_boulder_list.currentRow()].getSteps(), output, pickle.HIGHEST_PROTOCOL)

    def __newBoulder(self):
        b = Boulder()
        self.boulders.append(b)
        item = QListWidgetItem(b.getName())
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        self.lst_boulder_list.addItem(item)
        self.lst_boulder_list.setCurrentRow(self.lst_boulder_list.count() -1)

    def __deleteBoulder(self):
        idx = self.lst_boulder_list.currentRow()
        if (self.lst_boulder_list.count() > 1):
            self.boulders.pop(idx)
            self.lst_boulder_list.takeItem(idx)
            self.lst_boulder_list.setCurrentRow(idx - 1)