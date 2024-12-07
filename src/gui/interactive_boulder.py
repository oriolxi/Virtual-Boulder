import os
import pickle
import numpy as np
from PyQt6 import uic
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QDialog, QListWidgetItem, QFileDialog

import util
from boulder import Boulder, renderBoulderPreview, mirrorBoulder

class InteractiveBoulderDialog(QDialog):
    signal_start = pyqtSignal(int, int)
    signal_edit = pyqtSignal(int)
    signal_click = pyqtSignal(np.ndarray)
    signal_close = pyqtSignal()

    def __init__(self, parent, boulder_list, holds_bboxes, ref_img):
        super().__init__(parent)
        uic.loadUi("gui/interactive_boulder.ui", self)

        self.boulders = boulder_list
        self.holds = holds_bboxes
        self.reference_img = util.QimageFromCVimage(ref_img)
        self.draw_lines = True
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
        img = renderBoulderPreview(self.boulders[idx], self.holds, self.reference_img, draw_lines=self.draw_lines)
        self.lbl_preview_boulder.setPixmap(img.scaled(self.lbl_preview_boulder.size().width(), self.lbl_preview_boulder.size().height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.lbl_preview_boulder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.boulders[idx].setName(self.lst_boulder_list.currentItem().text())

        img_black = self.reference_img.copy()
        img_black.fill(Qt.GlobalColor.transparent)
        canvas_projection = renderBoulderPreview(self.boulders[idx], self.holds, img_black, draw_lines=self.draw_lines)
        self.signal_click.emit(util.CVimageFromQimage(canvas_projection.toImage()))

        self.sbox_boulder_start.setRange(0, self.boulders[idx].getNumSteps() - 1)
        self.sbox_boulder_start.setValue(0)

    def __setUpGui(self):
        self.btn_start_boulder.clicked.connect(self.__startBoulder)
        self.btn_edit_boulder.clicked.connect(self.__editBoulder)
        self.btn_mirror_boulder.clicked.connect(self.__mirrorBoulder)
        self.btn_load_boulder.clicked.connect(self.__loadBoulder)
        self.btn_save_boulder.clicked.connect(self.__saveBoulder)
        self.btn_new_boulder.clicked.connect(self.__newBoulder)
        self.btn_delete_boulder.clicked.connect(self.__deleteBoulder)
        self.lst_boulder_list.currentItemChanged.connect(self.updateBoulderPreview)

    def __startBoulder(self):
        self.signal_start.emit(self.lst_boulder_list.currentRow(), self.sbox_boulder_start.value())

    def __editBoulder(self):
        self.signal_edit.emit(self.lst_boulder_list.currentRow())

    def __mirrorBoulder(self):
        original_boulder = self.boulders[self.lst_boulder_list.currentRow()]
        mirror_boulder = mirrorBoulder(original_boulder, self.holds, self.reference_img.size().width())
        if mirrorBoulder is None: return
        
        self.boulders.append(mirror_boulder)
        item = QListWidgetItem(mirror_boulder.getName())
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        self.lst_boulder_list.addItem(item)
        self.lst_boulder_list.setCurrentRow(self.lst_boulder_list.count() -1)

    def __loadBoulder(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("Boulder file (*.bldr)")
        dialog.setDirectory('./saves/boulders')
        dialog.setViewMode(QFileDialog.ViewMode.Detail)
        fileNames = [""]
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

    def start(self):
        self.show()
        self.updateBoulderPreview()

    def closeEvent(self, event):
        self.signal_close.emit()