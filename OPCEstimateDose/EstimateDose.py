from PySide6.QtCore import QThread, Signal
from scipy import ndimage
import numpy as np

class clsEstimatingDose(QThread):
    signalDoseForResLines = Signal(float)
    signalDoseForTypPattern = Signal(float)

    def __init__(self, sigma, intResDotWidth, floatDoseforResDot) -> None:
        super().__init__()
        self.sigma = sigma
        self.intResDotWidthPx = intResDotWidth
        self.floatDoseforResDot = floatDoseforResDot
    
    def run(self):
        greyScaleLevelResDot = self.calculateGreyscaleLevelDot()
        greyScaleLevelResLine = self.calculateGreyscaleLevelLines()
        greyScaleLevelTypPattern = self.calculateGreyscaleLevelTypPattern()

        estDoseResLine = np.around(self.floatDoseforResDot * (greyScaleLevelResDot/greyScaleLevelResLine), 1)
        estDoseTypPattern = np.around(self.floatDoseforResDot * (greyScaleLevelResDot/greyScaleLevelTypPattern), 1)

        self.signalDoseForResLines.emit(estDoseResLine)
        self.signalDoseForTypPattern.emit(estDoseTypPattern)

        self.finished.emit()


    def calculateGreyscaleLevelDot(self):
        #Create a np 2d array with centre 2px greyscale level = 255, and the rest level = 0
        testArrayLength = 60
        inputNP2DArray = np.zeros(shape=(testArrayLength, testArrayLength))
        dotLeftPosition = int(testArrayLength/2)
        for j in range(dotLeftPosition, dotLeftPosition + self.intResDotWidthPx):
            for i in range(dotLeftPosition, dotLeftPosition + self.intResDotWidthPx):
                inputNP2DArray[j, i] = 1

        #Convolute this array with the Gaussian kernel of the user-input sigma value
        convolvedNP2DArray = ndimage.gaussian_filter(inputNP2DArray, self.sigma, order=0, output=None, mode='constant')
        #Obtain the greyscale level of the dot and return the greyscale level
        #returnedGreyscaleLevel = np.amax(convolvedNP2DArray)
        returnedGreyscaleLevel = np.mean(convolvedNP2DArray, where=inputNP2DArray!=0)

        return returnedGreyscaleLevel
    
    def calculateGreyscaleLevelLines(self):
        #Create a np 2d array with centre 2px greyscale level = 255, and the rest level = 0
        testArrayLength = 60
        inputNP2DArray = np.zeros(shape=(testArrayLength, testArrayLength))
        dotLeftPosition = int(testArrayLength/2)
        for j in range(dotLeftPosition, dotLeftPosition + self.intResDotWidthPx):
            for i in range(testArrayLength):
                inputNP2DArray[j, i] = 1

        #Convolute this array with the Gaussian kernel of the user-input sigma value
        convolvedNP2DArray = ndimage.gaussian_filter(inputNP2DArray, self.sigma, order=0, output=None, mode='constant')
        #Obtain the greyscale level of the dot and return the greyscale level
        #returnedGreyscaleLevel = np.amax(convolvedNP2DArray)
        returnedGreyscaleLevel = np.mean(convolvedNP2DArray, where=inputNP2DArray!=0)

        return returnedGreyscaleLevel
    
    def calculateGreyscaleLevelTypPattern(self):
        #Create a np 2d array with centre 2px greyscale level = 255, and the rest level = 0
        testArrayLength = 60
        inputNP2DArray = np.ones(shape=(testArrayLength, testArrayLength))

        #Convolute this array with the Gaussian kernel of the user-input sigma value
        convolvedNP2DArray = ndimage.gaussian_filter(inputNP2DArray, self.sigma, order=0, output=None, mode='constant')
        #Obtain the greyscale level of the dot and return the greyscale level
        #returnedGreyscaleLevel = np.amax(convolvedNP2DArray)
        returnedGreyscaleLevel = np.max(convolvedNP2DArray)

        return returnedGreyscaleLevel