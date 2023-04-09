import sys
import os

from PySide6.QtCore import QObject, QThread, Signal
import cv2
from scipy import ndimage
import numpy as np
from time import sleep
from random import random

import gdspy

from numba import jit, njit, vectorize

import matplotlib.pyplot as plt

from datetime import datetime

import gc

import tracemalloc

class clsOPCProcessingSingleThread(QThread):
    finished = Signal()
    signalUpdateInputImage = Signal(int)
    signalUpdateOutputImage = Signal()
    signalClearImageViews = Signal()
    signalStatusUpdate = Signal(str)
    signalUpdateProgress = Signal(int)

    def __init__(self, input_file_path, pixelSizeX, pixelSizeY, userInputIterations, userInputSigma, dotWidth, parent=None):
        super(clsOPCProcessingSingleThread, self).__init__(parent)
        #copy the user input information to local variables
        self.gdsii_file_path = input_file_path
        self.current_directory = os.path.dirname(input_file_path)
        self.pixel_size_X = pixelSizeX
        self.pixel_size_Y = pixelSizeY
        self.iterations = userInputIterations
        self.sigma = userInputSigma
        self.normalised_dot_width = dotWidth
        self.inputImageArray = np.ones((1, 1), dtype=np.uint8)
        self.outputImageArray = np.ones((1, 1), dtype=np.uint8)
        self.txtStatusUpdate = ""
        self.layers = {} # Array to hold all geometry, sorted into layers
        self.totalNumbersOfLayers = 0
        #Set threhold pixels to prevent the oupput image array occupying all the RAMs
        self.processing_image_shape_threshold = 8192
        self.currentProgress = 0
        self.progressStepSize = 100.
        #Set matplotlib objects
        self.fig = plt.figure(figsize=(1,1), frameon=False, dpi=1)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])

    def run(self):
        #Start processing
        self.producerAnalyseGDS2File()
        self.producerLayersToNPArray()
        self.resetNPArrays()
        self.finished.emit()

    #@jit(parallel=True)
    def producerAnalyseGDS2File(self):
        #Read the input file using the gdspy library
        gdsii = gdspy.GdsLibrary()
        gdsii.read_gds(self.gdsii_file_path, units='import')
        layers = {} # array to hold all geometry, sorted into layers
        cells = gdsii.top_level()
        for cell in cells:
            if cell.name == '$$$CONTEXT_INFO$$$':
                continue #skip this cell
            #combine with all referened cells (instances, SREFS, AREFS, etc.)
            cell = cell.flatten()
            #loop through paths in the cell
            for path in cell.paths:
                lnum = path.layers[0] #GDSII layer number
                #create an empty array to hold layer polygons if it does not exit yet
                layers[lnum] = [] if not lnum in layers else layers[lnum]
                #add paths (converted to polygons) of that layer
                for poly in path.get_polygons():
                    layers[lnum].append(poly)
            #loop through polygons (and boxes) in the cell:
            for polygon in cell.polygons:
                lnum = polygon.layers[0]
                layers[lnum] = [] if not lnum in layers else layers[lnum]
                for poly in polygon.polygons:
                    layers[lnum].append(poly)
            
        #Report all the polygons back
        self.layers = layers
        #Report the numbers of layers
        self.totalNumbersOfLayers = len(layers.keys())

        #Report total numbers of layers for debugging purposes
        self.signalStatusUpdate.emit("{} Total numbers of layers is {}.\n".format(self.format_time(), self.totalNumbersOfLayers))

    #@profile
    def producerLayersToNPArray(self):
        #Input all polygons in different layers
        layers = self.layers 
        totalNumbersOfLayers = self.totalNumbersOfLayers

        #Save the layer centre positions as a .txt file
        outputPositionFileName = "layerPositions.txt"
        outputPositonsFileFullPath = os.path.join(self.current_directory, outputPositionFileName)
        with open(outputPositonsFileFullPath, 'w') as f:
                f.write('Layer and its central position (mm)')
                f.write('\n')

        #Find the centre position for each layer
        #Produce the corresponding np array and output a bmp file for each layer
        #Loop through all layers
        for layer_number, polygons in layers.items():
            #Report status
            txtStatusReport = "{} Processing the layer number {}.\n".format(self.format_time(), layer_number)
            print(txtStatusReport)
            self.signalStatusUpdate.emit(txtStatusReport)

            #Clear the existing image views
            self.signalClearImageViews.emit()
            
            #Initialise a np 2D array to collect max values and min values in X and in Y
            listMaxValuesX = []
            listMinValuesX = []
            listMaxValuesY = []
            listMinValuesY = []
            #Loop through all the polygons in this layer
            for index, polygon in enumerate(polygons):
                maxValueXY = np.amax(polygon, axis=0)
                minValueXY = np.amin(polygon, axis=0)
                #Add the max X value, min X value, max Y value, min Y value to the array
                listMaxValuesX.append(maxValueXY[0])
                listMinValuesX.append(minValueXY[0])
                listMaxValuesY.append(maxValueXY[1])
                listMinValuesY.append(minValueXY[1])
            #Find the max locations and min locations of the all polygons in this layer
            maxLocationX = np.max(listMaxValuesX)
            minLocationX = np.min(listMinValuesX)
            maxLocationY = np.max(listMaxValuesY)
            minLocationY = np.min(listMinValuesY)
            #Report the centre position of patterns in this layer
            centrePositionX = np.around((maxLocationX + minLocationX)/2, 3)
            centrePositionY = np.around((maxLocationY + minLocationY)/2, 3)
            self.signalStatusUpdate.emit("{} centre Position of X = {}, center position of Y = {}.\n".format(self.format_time(), centrePositionX, centrePositionY))
            centrePositionXMM = np.around(centrePositionX/1000, 6)
            centrePositionYMM = np.around(centrePositionY/1000, 6)
            txtStatusReport = "{} Centre position of the layer number {} is at ({} mm, {} mm).\n".format(self.format_time(), layer_number, centrePositionXMM, centrePositionYMM)
            print(txtStatusReport)
            self.signalStatusUpdate.emit(txtStatusReport)
            
            #Output the center position of this layer
            with open(outputPositonsFileFullPath, 'a') as f:
                f.write('Layer {} at ( {} mm, {} mm)'.format(layer_number, centrePositionXMM, centrePositionYMM))
                f.write('\n')

            #Produce the numpy array
            imageShapeX = np.floor((maxLocationX - minLocationX)/self.pixel_size_X)
            imageShapeY = np.floor((maxLocationY - minLocationY)/self.pixel_size_Y)
            self.signalStatusUpdate.emit("{} imageShapeX = {}, imageShapeY = {}.\n".format(self.format_time(), imageShapeX, imageShapeY))

            shiftInX = np.around(minLocationX/self.pixel_size_X, 3)
            shiftInY = np.around(minLocationY/self.pixel_size_Y, 3)
            self.signalStatusUpdate.emit("{} Shift in X = {}, shift in Y = {}.\n".format(self.format_time(), shiftInX, shiftInY))

            if self.processing_image_shape_threshold > imageShapeY:
                processedImageShapeY = int(imageShapeY)
            else:
                processedImageShapeY = self.processing_image_shape_threshold

            if self.processing_image_shape_threshold > imageShapeX:
                processedImageShapeX = int(imageShapeX)
            else:
                processedImageShapeX = self.processing_image_shape_threshold

            numbersOfSubImagesX = int(np.ceil(imageShapeX/self.processing_image_shape_threshold))
            numbersOfSubImagesY = int(np.ceil(imageShapeY/self.processing_image_shape_threshold))
            totalNumbersOfSubImages = numbersOfSubImagesX * numbersOfSubImagesY
            
            #Start processing the curent image
            #Considering the threshold pixels
            #Divide total image into sub images
            #Each sub image is with the pixel processedImageShapeThreshold
            #self.outputImageArray = np.ones((int(imageShapeY), int(imageShapeX)), dtype = np.uint16)
            if (imageShapeX <= (10 * self.processing_image_shape_threshold)) and (imageShapeY <= (10 * self.processing_image_shape_threshold)):
                self.inputImageArray = np.ones((int(imageShapeY), int(imageShapeX)), dtype = np.uint16)
            else:
                self.inputImageArray = np.ones((1, 1), dtype=np.uint8)    
            
            #Set up the progress bar step size
            self.progressStepSize = int(np.floor(100/((totalNumbersOfLayers * totalNumbersOfSubImages * self.iterations))))
            self.signalStatusUpdate.emit("{} Step size of the progress bar is {}.\n".format(self.format_time(), self.progressStepSize))

            #Calculation begins
            for j in range(numbersOfSubImagesY):
                if j == 0:
                    startingPixelY = 0
                    subImageStartingPixelY = 0
                    returnStartingPixelY = 0
                else:
                    startingPixelY = (j - 1) * processedImageShapeY 
                    subImageStartingPixelY = startingPixelY + processedImageShapeY
                    returnStartingPixelY = processedImageShapeY
                if j == numbersOfSubImagesY - 1:
                    endingPixelY = int(imageShapeY)
                    subImageEndingPixelY = endingPixelY
                    returnEndingPixelY = endingPixelY
                elif (j + 2) * processedImageShapeY > int(imageShapeY):
                    endingPixelY = int(imageShapeY)
                    subImageEndingPixelY = subImageStartingPixelY + processedImageShapeY
                    returnEndingPixelY = returnStartingPixelY + processedImageShapeY
                else:
                    endingPixelY = (j + 2) * processedImageShapeY
                    subImageEndingPixelY = subImageStartingPixelY + processedImageShapeY
                    returnEndingPixelY = returnStartingPixelY + processedImageShapeY
                
                #Define the temp output image array
                outputImageArrayNumPixelsY = subImageEndingPixelY - subImageStartingPixelY
                self.outputImageArray = np.ones((outputImageArrayNumPixelsY, int(imageShapeX)), dtype=np.uint16)

                for i in range(numbersOfSubImagesX):
                    #Update the current status
                    txtStatusReport = "{} Processing images {}/{}, i = {}, j = {}.\n".format(self.format_time(), i + (j * numbersOfSubImagesX) + 1, totalNumbersOfSubImages, i, j)
                    print(txtStatusReport)
                    self.signalStatusUpdate.emit(txtStatusReport)
                    #Cut out the sub image from the full image for convolution and normalisation
                    #Find the position ((startingPixelX, startingPixelY), (endingPixelX, endingPixelY)) of the sub image in the full image
                    if i == 0:
                        startingPixelX = 0
                        subImageStartingPixelX = 0
                        returnStartingPixelX = 0
                    else:
                        startingPixelX = (i - 1) * processedImageShapeX
                        subImageStartingPixelX = startingPixelX + processedImageShapeX
                        returnStartingPixelX = processedImageShapeX
                    if i == numbersOfSubImagesX - 1:
                        endingPixelX = int(imageShapeX)
                        subImageEndingPixelX = endingPixelX
                        returnEndingPixelX = endingPixelX
                    elif (i + 2) * processedImageShapeX > int(imageShapeX):
                        endingPixelX = int(imageShapeX)
                        subImageEndingPixelX = subImageStartingPixelX + processedImageShapeX
                        returnEndingPixelX = returnStartingPixelX + processedImageShapeX
                    else:
                        endingPixelX = (i + 2) * processedImageShapeX
                        subImageEndingPixelX = subImageStartingPixelX + processedImageShapeX
                        returnEndingPixelX = returnStartingPixelX + processedImageShapeX

                    #Copy the sub image from the main image
                    processingSubImage = self.ftnFindCurrentRegionOfInterest(polygons, startingPixelX, endingPixelX, startingPixelY, endingPixelY, shiftInX, shiftInY, imageShapeX, imageShapeY, j, numbersOfSubImagesY, (subImageEndingPixelY - subImageStartingPixelY))
                    regionOfInterest = processingSubImage[returnStartingPixelY:returnEndingPixelY, returnStartingPixelX:returnEndingPixelX]
                    regionOfInterestMax = np.amax(regionOfInterest)
                    inputImageMax = np.amax(processingSubImage)
                    
                    #Updating the input image
                    #Only update when the image pixel sizes < 2 * processing_image_shape_threshold
                    #Find the memory usage
                    #tracemalloc.start()
                    if (imageShapeX <= (10 * self.processing_image_shape_threshold)) and (imageShapeY <= (10 * self.processing_image_shape_threshold)):
                        self.inputImageArray[subImageStartingPixelY:subImageEndingPixelY, subImageStartingPixelX:subImageEndingPixelX] = regionOfInterest
                        txtStatusReport = "{} Updating the input image on layer {}.\n".format(self.format_time(), layer_number)
                        self.signalStatusUpdate.emit(txtStatusReport)
                        self.signalUpdateInputImage.emit(layer_number)
                        sleep(random())
                    else:
                        txtStatusReport = "{} Input image size on layer {} is too large. The display will not be updated.\n".format(self.format_time(), layer_number)
                        self.signalStatusUpdate.emit(txtStatusReport)
                        self.inputImageArray = np.ones((1, 1), dtype=np.uint8)
                    #Clear the memory
                    del regionOfInterest
                    gc.collect()

                    #Proess this sub image
                    #Only process when the image is not empty
                    self.signalStatusUpdate.emit("{} regionOfInterestMax = {}.\n".format(self.format_time(), regionOfInterestMax))
                    if np.amax(regionOfInterestMax) > 1:
                        processedSubOutputImage = self.ftnConsumerProcessImage2(processingSubImage, inputImageMax)
                        #Return this sub image back to the output array
                        self.outputImageArray[:, subImageStartingPixelX:subImageEndingPixelX] = processedSubOutputImage[returnStartingPixelY:returnEndingPixelY, returnStartingPixelX:returnEndingPixelX]

                        #Clear the memory usage
                        del processedSubOutputImage
                        gc.collect()
                    else:
                        #Empyt image. Only update the progress bar
                        self.currentProgress = self.currentProgress + (self.progressStepSize * self.iterations)
                        self.signalUpdateProgress.emit(self.currentProgress)
                                        
                    #Clear the memory
                    del processingSubImage
                    gc.collect()
                
                #Save this part of output image
                #Update the status board
                txtStatusReport = "{} Saving part of the output image on layer {}, with j = {}/{}.\n".format(self.format_time(), layer_number, j, numbersOfSubImagesY)
                self.signalStatusUpdate.emit(txtStatusReport)

                #Save the processed output image (this part) as str
                outputImageName = "Layer " + str(layer_number) + ".str"
                outputCurrentImageFullPath = os.path.join(self.current_directory, outputImageName)
                self.ftnSaveSTRFile3(self.outputImageArray, outputCurrentImageFullPath, int(imageShapeX), int(imageShapeY), j)

                #Clear the memory
                self.outputImageArray = np.ones((1, 1), dtype=np.uint8)
                gc.collect()

            #All done.
            txtStatusReport = "{} Finish processing the image on layer {}.\n".format(self.format_time(), layer_number)
            self.signalStatusUpdate.emit(txtStatusReport)

        #Set the progress bar = 100%
        self.currentProgress = 100
        self.signalUpdateProgress.emit(self.currentProgress)
        
    def resetNPArrays(self):
        self.inputImageArray = np.empty((1, 1), dtype=np.uint8)
        self.outputImageArray = np.empty((1, 1), dtype=np.uint8)

    def format_time(self):
        now = datetime.now()
        strNow = now.strftime('%d-%m-%Y %H:%M:%S.%f')

        return strNow[:-3]
    
    def ftnFindCurrentRegionOfInterest(self, polygons, startingPixelPositionX, endingPixelPositionX, startingPixelPositionY, endingPixelPositionY, shiftInX, shiftInY, imageShapeX, imageShapeY, j, numbersOfSubImagesY, subProcessedImageShapeY):
        #Initialise the parameters
        processingImageShapeX = endingPixelPositionX - startingPixelPositionX 
        processingImageShapeY = endingPixelPositionY - startingPixelPositionY
        self.fig.set_size_inches(processingImageShapeX, processingImageShapeY)
        self.fig.set_dpi(1)
        self.fig.set_frameon(False)
        self.ax.set_xlim([-0.5, processingImageShapeX + 0.5])
        self.ax.set_ylim([-0.5, processingImageShapeY + 0.5])
        self.ax.set_axis_off()
        #Set plotting range
        #Shift the polygons positions to the centre of the current image
        #Shift the centre position of each position half pixel size to the left
        for index, polygon in enumerate(polygons):
            #Calculate starts
            if numbersOfSubImagesY > 1 and j == numbersOfSubImagesY - 1:
                N = np.array([[shiftInX + startingPixelPositionX, shiftInY]]).astype(np.float32)
            elif j == 0:
                N = np.array([[shiftInX + startingPixelPositionX, shiftInY + startingPixelPositionY + (imageShapeY - endingPixelPositionY)]]).astype(np.float32)
            else:
                N = np.array([[shiftInX + startingPixelPositionX, shiftInY + startingPixelPositionY + (imageShapeY - endingPixelPositionY) - ((j - 1) * subProcessedImageShapeY)]]).astype(np.float32)
            
            pixelSizeArray = np.array([self.pixel_size_X, self.pixel_size_Y]).astype(np.float32)
            #Normalise the polygon to the pixel size
            testPolygon2 = np.around(np.divide(polygon, pixelSizeArray), 1).astype(np.float32)
            #Shift the centre of polygon so that only region of interest is converted
            testPolygon3 = np.around(np.subtract(testPolygon2, N), 1).astype(np.float32)
            #Fill in polygon using matplotlib
            xs, ys = zip(*testPolygon3)    
            self.ax.fill(xs, ys, facecolor='black')
            self.fig.add_axes(self.ax)
        
        #Save the current image as a png file
        outputPositionFileName = "currentRegionOfInterest.png"
        outputPositonsFileFullPath = os.path.join(self.current_directory, outputPositionFileName)
        plt.savefig(outputPositonsFileFullPath)

        #Reset the figure object
        self.fig.clear()
        self.fig.set_size_inches(1, 1)
        gc.collect()

        #Read back the previously saved image using cv2 to have a np 2D array
        readImage = cv2.imread(outputPositonsFileFullPath)
        #Only read the blue channel
        slicedImage = readImage[:, :, 0]
        #Inverted the colour level
        returnedImage = np.where(slicedImage == 255, 0, 255)
        #Return the image
        return returnedImage

    #@jit(nopython=True)
    #@njit
    #@jit(parallel=True)
    def ftnConsumerProcessImage2(self, npArrayInputImage, inputImageMax):
        #Normalise the input image
        npArrayNormInputImage = np.divide(npArrayInputImage, inputImageMax)
        
        #Loop through iterations
        for i in range(self.iterations):
            #Convolute the image
            npArrayConvolutedImage = ndimage.gaussian_filter(npArrayNormInputImage, self.sigma, order=0, output=None, mode='constant')
            #Correct the image
            npArrayNormInputImage = self.ftnNormalisingImageVectorisation2(npArrayNormInputImage, npArrayConvolutedImage)
            #Update the progress bar
            self.currentProgress += self.progressStepSize
            self.signalUpdateProgress.emit(self.currentProgress)
        
        #Calculate the final greyscale level
        npArrayNormInputImage = self.ftnCalculateFinalGreyscaleLevelVectorisation(npArrayNormInputImage)
        npOutputImageArray = np.clip(npArrayNormInputImage, 0, 255).astype(np.uint16)

        #Return output image array
        return npOutputImageArray
    
    #@jit()
    def ftnNormalisingImageVectorisation2(self, originalImage, convolutedImage):
        npArrayInputImage = np.around(originalImage,1)
        arrayInvertedConvolutedImage = np.divide(1, convolutedImage, where=np.around(originalImage, 1)!=0)
        arrayCorrectFactor = np.multiply(arrayInvertedConvolutedImage, npArrayInputImage, where=np.around(originalImage, 1)!=0)
        max_correctFactorArray = np.amax(arrayCorrectFactor)
        normed_ConvolutedImage = np.around(np.divide(arrayCorrectFactor, max_correctFactorArray), 1)

        return normed_ConvolutedImage

    #@jit()
    def ftnCalculateFinalGreyscaleLevelVectorisation(self, inputImage):
        #npArrayInputImage = np.around(inputImage,1)
        npArrayInputImage = inputImage
        #Calculate the greyscale level for a give pixels dot
        normalisingGreyScaleLevel = self.calculateGreyscaleLevelDot() * 255.0
        #Convolute the inputImage
        convolvedImage = ndimage.gaussian_filter(npArrayInputImage, self.sigma, order=0, output=None, mode='constant')
        #Normalise the greyscale levels in the output image
        normalisedConvolvedImage = np.divide(normalisingGreyScaleLevel, convolvedImage, where=np.around(npArrayInputImage, 1)!=0)
        #Calculate the final greyscale levels in the output image
        npArrayOutputImage = np.multiply(normalisedConvolvedImage, npArrayInputImage, where=np.around(npArrayInputImage, 1)!=0)

        return npArrayOutputImage

    def calculateGreyscaleLevelDot(self):
        #Create a np 2d array with centre 2px greyscale level = 255, and the rest level = 0
        testArrayLength = 60
        inputNP2DArray = np.zeros(shape=(testArrayLength, testArrayLength))
        dotLeftPosition = int(testArrayLength/2)
        for j in range(dotLeftPosition, dotLeftPosition + self.normalised_dot_width):
            for i in range(dotLeftPosition, dotLeftPosition + self.normalised_dot_width):
                inputNP2DArray[j, i] = 1

        #Convolute this array with the Gaussian kernel of the user-input sigma value
        convolvedNP2DArray = ndimage.gaussian_filter(inputNP2DArray, self.sigma, order=0, output=None, mode='constant')
        #Obtain the greyscale level of the dot and return the greyscale level
        #returnedGreyscaleLevel = np.amax(convolvedNP2DArray)
        returnedGreyscaleLevel = np.mean(convolvedNP2DArray, where=inputNP2DArray!=0)

        print("{} Returned greyscale level = {}".format(self.format_time(), returnedGreyscaleLevel))
        self.signalStatusUpdate.emit("{} Returned greyscale level = {}. \n".format(self.format_time(), returnedGreyscaleLevel))

        return returnedGreyscaleLevel
    
    def ftnSaveSTRFile2(self, image, imageFileFullPath):
        #Find the image dimensions
        imageWidth = image.shape[1]
        imageHeight = image.shape[0]
        #Convert dimensions into bytes
        imageWidthByte = imageWidth.to_bytes(4, byteorder='little')
        imageHeightByte = imageHeight.to_bytes(4, byteorder='little')
        #Covert image array into a 1D array
        image1DArray = image.flatten()
        image1DArrayInt = image1DArray.astype('uint8')
        #Convert the image array into a byte array
        imageByteArray = image1DArrayInt.tobytes()
        #Save the converted image array
        f = open(imageFileFullPath, 'wb')
        f.write(imageWidthByte)
        f.write(imageHeightByte)
        f.write(imageByteArray)
        f.close()

    def ftnSaveSTRFile3(self, image, imageFileFullPath, fullImageWidth, fullImageHeight, j):
        #Find the image dimensions
        #imageWidth = image.shape[1]
        #imageHeight = image.shape[0]
        #Convert dimensions into bytes
        if j == 0:    
            imageWidthByte = fullImageWidth.to_bytes(4, byteorder='little')
            imageHeightByte = fullImageHeight.to_bytes(4, byteorder='little')
        #Covert image array into a 1D array
        image1DArray = image.flatten()
        image1DArrayInt = image1DArray.astype('uint8')
        #Convert the image array into a byte array
        imageByteArray = image1DArrayInt.tobytes()
        #Save the converted image array
        if j == 0:
            f = open(imageFileFullPath, 'wb')
            f.write(imageWidthByte)
            f.write(imageHeightByte)
            f.write(imageByteArray)
            f.close()
        else:
            f = open(imageFileFullPath, 'ab')
            f.write(imageByteArray)
            f.close()


    
        

            

    

            

