#This is the version with attempts to speed up the overally process

import sys
import os

from PySide6.QtCore import QObject, QThread, Signal, Slot, QPoint
from PySide6.QtWidgets import QFileDialog, QApplication, QWidget
from OPCUI.ui_form import Ui_Widget
from OPCProcessing.processing_single_threaded import clsOPCProcessingSingleThread 
from OPCEstimateDose.EstimateDose import clsEstimatingDose

import numpy as np

from line_profiler import LineProfiler

import cProfile

class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.input_file_path = ""
        self.pixel_size_X = 0.1
        self.pixel_size_Y = 0.1
        self.iterations = 1
        self.sigma = 1.0
        self.normalising_dot_width_px = 1
        self.requiredDoseResDot = 1.0
        self.qPointAnchor = QPoint(0, 0)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)
        self.ui.pBtnSelectInputFile.clicked.connect(self.openGDSIIFile)
        self.ui.pushBtnGo.clicked.connect(self.startProcessing)
        self.ui.pushBtnEstimateDose.clicked.connect(self.startEstimatingDose)    

    def openGDSIIFile(self):
        #Record the cif file location
        tupleFName = QFileDialog.getOpenFileName(self, "Select a .gdsii file", os.getcwd(), "All Files (GDSII files (*.gds))")
        local_gdsii_file_path = tupleFName[0]
        #Report back
        self.ui.lineEditInputFileFullPath.clear()
        self.ui.lineEditInputFileFullPath.setText(local_gdsii_file_path)
        self.input_file_path = local_gdsii_file_path

    def startProcessing(self):
        #Import user input information
        self.pixel_size_X = float(self.ui.lineEditPixelSizeInX.text())
        self.pixel_size_Y = float(self.ui.lineEditPixelSizeInY.text())
        self.iterations = int(self.ui.lineEditIterations.text())
        self.sigma = float(self.ui.lineEditSigma.text())
        self.normalising_dot_width_px = int(self.ui.lineEditDotWidthPx.text())

        #Clear the status report and image views
        self.ui.textEditStatusReport.clear()
        self.ui.imageViewInputFile.clear()
        QApplication.processEvents()

        #Start processing
        #Create a OPCProcessingSingleThread object
        self.newProcessing = clsOPCProcessingSingleThread(self.input_file_path, self.pixel_size_X, self.pixel_size_Y, self.iterations, self.sigma,
        self.normalising_dot_width_px)
        #Create a QThread object
        self.thread = QThread()
        #Move the processing to the thread
        self.newProcessing.moveToThread(self.thread)
        #Connect signals and slots
        self.thread.started.connect(self.newProcessing.run)
        self.newProcessing.finished.connect(self.thread.quit)
        self.newProcessing.finished.connect(self.newProcessing.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        #Start the thread
        self.thread.start()

        #Reset the progress bar
        self.ui.progressBar.reset()

        #Clear both the input image view and the output image view
        self.newProcessing.signalClearImageViews.connect(self.qtSlot_clearImageViews)
        #Update the input image
        self.newProcessing.signalUpdateInputImage.connect(self.qtSlot_UpdateInputImage)
        #Update the current status
        self.newProcessing.signalStatusUpdate.connect(self.qtSlot_StatusReport)
        #Update the current progress
        self.newProcessing.signalUpdateProgress.connect(self.qtSlot_updateProgress)
        
        #Final resets
        self.ui.pushBtnGo.setEnabled(False)
        self.ui.pushBtnGo.setText("Under processing!")
        self.thread.finished.connect(
            lambda: self.ui.pushBtnGo.setEnabled(True)
        )
        self.thread.finished.connect(
            lambda: self.ui.pushBtnGo.setText("Start processing!")
        )

    def startEstimatingDose(self):
        self.sigma = float(self.ui.lineEditSigma.text())
        self.normalising_dot_width_px = int(self.ui.lineEditDotWidthPx.text())
        self.requiredDoseResDot = float(self.ui.lineEditDotDose.text())

        self.newEstimating = clsEstimatingDose(self.sigma, self.normalising_dot_width_px, self.requiredDoseResDot)

        #Create a QThread object
        self.thread = QThread()
        #Move the processing to the thread
        self.newEstimating.moveToThread(self.thread)
        #Connect signals and slots
        self.thread.started.connect(self.newEstimating.run)
        self.newEstimating.finished.connect(self.thread.quit)
        self.newEstimating.finished.connect(self.newEstimating.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        #Start the thread
        self.thread.start()

        self.newEstimating.signalDoseForResLines.connect(self.qtSlot_updateEstimatedDoseResLine)
        self.newEstimating.signalDoseForTypPattern.connect(self.qtSlot_updateEstimatedTypPattern)


    @Slot()
    def qtSlot_clearImageViews(self):
        self.ui.imageViewInputFile.clear()
    
    @Slot()
    def qtSlot_UpdateInputImage(self, layer_number):
        self.ui.imageViewInputFile.clear()
        self.ui.imageViewInputFile.setImage(np.flipud(np.rot90(self.newProcessing.inputImageArray)))
        self.ui.imageViewInputFile.setLevels(0, 255)
        self.ui.label_currentProcessingLayer.setText("Processing layer {}".format(layer_number))

    @Slot()
    def qtSlot_StatusReport(self, txtStatusUpdate):
        #Update the text.
        self.ui.textEditStatusReport.insertPlainText(txtStatusUpdate)
        #Anchor the vertical scroll bar to the bottom.
        vsb = self.ui.textEditStatusReport.verticalScrollBar()
        vsb.setValue(vsb.maximum())
    
    @Slot()
    def qtSlot_updateProgress(self, intCurrentProgress):
        self.ui.progressBar.setValue(intCurrentProgress)
    
    @Slot()
    def qtSlot_updateEstimatedDoseResLine(self, floatEstimatedDose):
        self.ui.label_DoseResLine.setText("Resolution lines: {} mJ/cm^2".format(floatEstimatedDose))
    
    @Slot()
    def qtSlot_updateEstimatedTypPattern(self, floatEstimatedDose):
        self.ui.label_DoseTypPattern.setText("Typical patterns: {} mJ/cm^2".format(floatEstimatedDose))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.show()

    sys.exit(app.exec())

    
