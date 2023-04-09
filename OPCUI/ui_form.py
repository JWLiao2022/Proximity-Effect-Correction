# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.3.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractScrollArea, QApplication, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QProgressBar, QPushButton,
    QSizePolicy, QTextEdit, QVBoxLayout, QWidget)

from pyqtgraph import ImageView

class Ui_Widget(object):
    def setupUi(self, Widget):
        if not Widget.objectName():
            Widget.setObjectName(u"Widget")
        Widget.resize(1080, 600)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Widget.sizePolicy().hasHeightForWidth())
        Widget.setSizePolicy(sizePolicy)
        Widget.setMinimumSize(QSize(1080, 600))
        Widget.setMaximumSize(QSize(1280, 720))
        self.pushBtnGo = QPushButton(Widget)
        self.pushBtnGo.setObjectName(u"pushBtnGo")
        self.pushBtnGo.setGeometry(QRect(240, 190, 171, 81))
        self.textEditStatusReport = QTextEdit(Widget)
        self.textEditStatusReport.setObjectName(u"textEditStatusReport")
        self.textEditStatusReport.setGeometry(QRect(10, 290, 431, 211))
        self.textEditStatusReport.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.textEditStatusReport.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContentsOnFirstShow)
        self.progressBar = QProgressBar(Widget)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setGeometry(QRect(850, 20, 211, 24))
        self.progressBar.setValue(0)
        self.imageViewInputFile = ImageView(Widget)
        self.imageViewInputFile.setObjectName(u"imageViewInputFile")
        self.imageViewInputFile.setGeometry(QRect(470, 50, 600, 450))
        self.label_8 = QLabel(Widget)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setGeometry(QRect(10, 10, 133, 23))
        self.widget = QWidget(Widget)
        self.widget.setObjectName(u"widget")
        self.widget.setGeometry(QRect(10, 30, 431, 26))
        self.horizontalLayout = QHBoxLayout(self.widget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.lineEditInputFileFullPath = QLineEdit(self.widget)
        self.lineEditInputFileFullPath.setObjectName(u"lineEditInputFileFullPath")

        self.horizontalLayout.addWidget(self.lineEditInputFileFullPath)

        self.pBtnSelectInputFile = QPushButton(self.widget)
        self.pBtnSelectInputFile.setObjectName(u"pBtnSelectInputFile")

        self.horizontalLayout.addWidget(self.pBtnSelectInputFile)

        self.widget1 = QWidget(Widget)
        self.widget1.setObjectName(u"widget1")
        self.widget1.setGeometry(QRect(10, 70, 201, 201))
        self.horizontalLayout_2 = QHBoxLayout(self.widget1)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.label = QLabel(self.widget1)
        self.label.setObjectName(u"label")

        self.verticalLayout_2.addWidget(self.label)

        self.label_2 = QLabel(self.widget1)
        self.label_2.setObjectName(u"label_2")

        self.verticalLayout_2.addWidget(self.label_2)

        self.label_3 = QLabel(self.widget1)
        self.label_3.setObjectName(u"label_3")

        self.verticalLayout_2.addWidget(self.label_3)

        self.label_4 = QLabel(self.widget1)
        self.label_4.setObjectName(u"label_4")

        self.verticalLayout_2.addWidget(self.label_4)

        self.label_5 = QLabel(self.widget1)
        self.label_5.setObjectName(u"label_5")

        self.verticalLayout_2.addWidget(self.label_5)

        self.label_6 = QLabel(self.widget1)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setWordWrap(True)

        self.verticalLayout_2.addWidget(self.label_6)


        self.horizontalLayout_2.addLayout(self.verticalLayout_2)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.lineEditPixelSizeInX = QLineEdit(self.widget1)
        self.lineEditPixelSizeInX.setObjectName(u"lineEditPixelSizeInX")

        self.verticalLayout.addWidget(self.lineEditPixelSizeInX)

        self.lineEditPixelSizeInY = QLineEdit(self.widget1)
        self.lineEditPixelSizeInY.setObjectName(u"lineEditPixelSizeInY")

        self.verticalLayout.addWidget(self.lineEditPixelSizeInY)

        self.lineEditIterations = QLineEdit(self.widget1)
        self.lineEditIterations.setObjectName(u"lineEditIterations")

        self.verticalLayout.addWidget(self.lineEditIterations)

        self.lineEditSigma = QLineEdit(self.widget1)
        self.lineEditSigma.setObjectName(u"lineEditSigma")

        self.verticalLayout.addWidget(self.lineEditSigma)

        self.lineEditDotWidthPx = QLineEdit(self.widget1)
        self.lineEditDotWidthPx.setObjectName(u"lineEditDotWidthPx")

        self.verticalLayout.addWidget(self.lineEditDotWidthPx)

        self.lineEditDotDose = QLineEdit(self.widget1)
        self.lineEditDotDose.setObjectName(u"lineEditDotDose")

        self.verticalLayout.addWidget(self.lineEditDotDose)


        self.horizontalLayout_2.addLayout(self.verticalLayout)

        self.widget2 = QWidget(Widget)
        self.widget2.setObjectName(u"widget2")
        self.widget2.setGeometry(QRect(240, 70, 172, 111))
        self.verticalLayout_4 = QVBoxLayout(self.widget2)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.pushBtnEstimateDose = QPushButton(self.widget2)
        self.pushBtnEstimateDose.setObjectName(u"pushBtnEstimateDose")

        self.verticalLayout_4.addWidget(self.pushBtnEstimateDose)

        self.groupBox = QGroupBox(self.widget2)
        self.groupBox.setObjectName(u"groupBox")
        self.widget3 = QWidget(self.groupBox)
        self.widget3.setObjectName(u"widget3")
        self.widget3.setGeometry(QRect(11, 31, 151, 38))
        self.verticalLayout_3 = QVBoxLayout(self.widget3)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.label_DoseResLine = QLabel(self.widget3)
        self.label_DoseResLine.setObjectName(u"label_DoseResLine")

        self.verticalLayout_3.addWidget(self.label_DoseResLine)

        self.label_DoseTypPattern = QLabel(self.widget3)
        self.label_DoseTypPattern.setObjectName(u"label_DoseTypPattern")

        self.verticalLayout_3.addWidget(self.label_DoseTypPattern)


        self.verticalLayout_4.addWidget(self.groupBox)

        self.widget4 = QWidget(Widget)
        self.widget4.setObjectName(u"widget4")
        self.widget4.setGeometry(QRect(470, 20, 371, 18))
        self.horizontalLayout_3 = QHBoxLayout(self.widget4)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.label_7 = QLabel(self.widget4)
        self.label_7.setObjectName(u"label_7")

        self.horizontalLayout_3.addWidget(self.label_7)

        self.label_currentProcessingLayer = QLabel(self.widget4)
        self.label_currentProcessingLayer.setObjectName(u"label_currentProcessingLayer")

        self.horizontalLayout_3.addWidget(self.label_currentProcessingLayer)


        self.retranslateUi(Widget)

        QMetaObject.connectSlotsByName(Widget)
    # setupUi

    def retranslateUi(self, Widget):
        Widget.setWindowTitle(QCoreApplication.translate("Widget", u"Optical Proximity Correction", None))
        self.pushBtnGo.setText(QCoreApplication.translate("Widget", u"Start processing!", None))
        self.label_8.setText(QCoreApplication.translate("Widget", u"Input GDS file location", None))
        self.pBtnSelectInputFile.setText(QCoreApplication.translate("Widget", u"...", None))
        self.label.setText(QCoreApplication.translate("Widget", u"Pixel size in X", None))
        self.label_2.setText(QCoreApplication.translate("Widget", u"Pixel size in Y", None))
        self.label_3.setText(QCoreApplication.translate("Widget", u"Iterations", None))
        self.label_4.setText(QCoreApplication.translate("Widget", u"Sigma", None))
        self.label_5.setText(QCoreApplication.translate("Widget", u"Resolution dot width (px)", None))
        self.label_6.setText(QCoreApplication.translate("Widget", u"Required dose for a resolution dot (mJ/cm2)", None))
        self.pushBtnEstimateDose.setText(QCoreApplication.translate("Widget", u"Estimate does (mJ/cm^2)", None))
        self.groupBox.setTitle(QCoreApplication.translate("Widget", u"Estimated dose (mJ/cm^2)", None))
        self.label_DoseResLine.setText(QCoreApplication.translate("Widget", u"Resolution lines: ", None))
        self.label_DoseTypPattern.setText(QCoreApplication.translate("Widget", u"Typical patterns:", None))
        self.label_7.setText(QCoreApplication.translate("Widget", u"Processed Input Pattern", None))
        self.label_currentProcessingLayer.setText(QCoreApplication.translate("Widget", u"Current processing layer:", None))
    # retranslateUi

