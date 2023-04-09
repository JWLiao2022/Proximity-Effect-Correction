import sys
import os

from PySide6 import QtCore
import cv2
from scipy import ndimage
import numpy as np
from multiprocessing import Process
from multiprocessing import Queue
from queue import Empty
from time import sleep
from random import random

import gdspy

import pyqtgraph as pg

import pandas as pd


class clsOPCProcessing():
    def __init__(self, input_file_path, pixelSizeX, pixelSizeY, userInputIterations, userInputSigma, dotWidth, imageShapeThreshold) -> None:
        #copy the user input information to local variables
        self.gdsii_file_path = input_file_path
        self.current_directory = os.path.dirname(input_file_path)
        self.pixel_size_X = pixelSizeX
        self.pixel_size_Y = pixelSizeY
        self.iterations = userInputIterations
        self.sigma = userInputSigma
        self.normalised_dot_width = dotWidth
        self.processing_image_shape_threshold = imageShapeThreshold
        self.inputImageArray = np.zeros((1,1))

        #dispatcher.connect(self.ftnTestReceiveMessage, signal=SIGNAL, sender=dispatcher.Any)

    def analysisGDSIIFile(self, queue):
        #Save the layer centre positions as a .txt file
        outputPositionFileName = "layerPositions.txt"
        outputPositonsFileFullPath = os.path.join(self.current_directory, outputPositionFileName)        
        with open(outputPositonsFileFullPath, 'w') as f:
                f.write('Layer and its central position (mm)')
                f.write('\n')
        
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

        #Find the centre position for each layer
        #Produce the corresponding np array and output a bmp file for each layer
        #Loop through all layers
        for layer_number, polygons in layers.items():

            #Report status
            print("Processing the layer number {}".format(layer_number))

            currentImage = np.zeros((1, 1))

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
            centrePositionX = (maxLocationX + minLocationX)/2
            centrePositionY = (maxLocationY + minLocationY)/2

            centrePositionXMM = np.around(centrePositionX/1000, 8)
            centrePositionYMM = np.around(centrePositionY/1000, 8)

            print("Centre position of the layer number {} is at ({} mm, {} mm).".format(layer_number, centrePositionXMM, centrePositionYMM))

            #Output the center position of this layer
            with open(outputPositonsFileFullPath, 'a') as f:
                f.write('Layer {} at ( {} mm, {} mm)'.format(layer_number, centrePositionXMM, centrePositionYMM))
                f.write('\n')

            #Produce the numpy array
            imageShapeX = np.floor((maxLocationX - minLocationX)/self.pixel_size_X)
            imageShapeY = np.floor((maxLocationY - minLocationY)/self.pixel_size_Y)
            image = np.ones((int(imageShapeY), int(imageShapeX)), dtype = np.uint16)

            shiftInX = minLocationX/self.pixel_size_X
            shiftInY = minLocationY/self.pixel_size_Y

            #Shift the polygons positions to the centre of the current image
            #Shift the centre position of each position half pixel size to the left
            for index, polygon in enumerate(polygons):
                N = np.float32([[shiftInX, shiftInY]])
                NRounded = np.around(N, 0)
                pixelSizeArray = np.float32([self.pixel_size_X, self.pixel_size_Y])
                testPolygon2 = np.divide(polygon, pixelSizeArray)
                testPolygon3 = np.subtract(testPolygon2, NRounded)
                testPolygon3Mean = testPolygon2.mean(axis=0)
                testPolygon3Abstract = np.subtract(testPolygon3, testPolygon3Mean)
                testPolygon3Normalised = np.divide(testPolygon3Abstract, np.absolute(testPolygon3Abstract))
                shiftPixelFactor = np.float16([0.5, 0.5])
                testPolygon3Correction = np.multiply(testPolygon3Normalised, shiftPixelFactor)
                testPolygon4 = np.subtract(testPolygon3, testPolygon3Correction)
                #testPolygon3 = np.around(testPolygon2 - np.amin(testPolygon2, axis=0),2)
                #maxTestPolygon3 = np.amax(testPolygon3, axis=0)
                #testPolygon4 = np.around(testPolygon2 + ((testPolygon3/maxTestPolygon3) * (-0.6)) - NRounded, 2)
                #intPolygon4 = np.around(testPolygon4, 0).astype('int')
                #cv2.fillPoly(image, pts = [intPolygon4], color = (255, 0, 0))
                intPolygon4 = np.around(testPolygon4, 0).astype('int')
                cv2.fillPoly(image, pts = [intPolygon4], color = (255, 0, 0))

            #Flip the image
            currentImage = cv2.flip(image, 0)

            #Save the input image
            outputImageNameBMP = "Input image" + ".bmp"
            outputCurrentImageFullPathBMP = os.path.join(self.current_directory, outputImageNameBMP)
            cv2.imwrite(outputCurrentImageFullPathBMP, currentImage)
            #Report the currentImage back
            self.inputImageArray = currentImage
            

            #Divide total image into sub images
            #Each sub image is with the pixel processedImageShapeThreshold
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

            #Copy each sub image into queue item
            print("Producer: Running", flush=True)
            queue.put('OneLayerBegin')
            queue.put(layer_number)
            queue.put(imageShapeX)
            queue.put(imageShapeY)
            #block
            sleep(random())
            for j in range(numbersOfSubImagesY):
                if j == 0:
                    startingPixelY = 0
                    subImageStartingPixelY = 0
                else:
                    startingPixelY = (j - 1) * processedImageShapeY
                    subImageStartingPixelY = startingPixelY + processedImageShapeY

                if j == numbersOfSubImagesY - 1 or j == numbersOfSubImagesY - 2:
                    endingPixelY = int(imageShapeY)
                    subImageEndingPixelY = endingPixelY
                else:
                    endingPixelY = (j + 2) * processedImageShapeY
                    subImageEndingPixelY = endingPixelY - processedImageShapeY

                for i in range(numbersOfSubImagesX):
                    print("Processing images {}/{}, i = {}, j = {}".format(i + (j * numbersOfSubImagesX) + 1, totalNumbersOfSubImages, i, j))

                    #Cut out the sub image from the full image for convolution and normalisation
                    #Find the position ((startingPixelX, startingPixelY), (endingPixelX, endingPixelY)) of the sub image in the full image
                    if i == 0:
                        startingPixelX = 0
                        subImageStartingPixelX = 0
                    else:
                        startingPixelX = (i - 1) * processedImageShapeX
                        subImageStartingPixelX = startingPixelX + processedImageShapeX

                    if i == numbersOfSubImagesX - 1 or i == numbersOfSubImagesX - 2:
                        endingPixelX = int(imageShapeX)
                        subImageEndingPixelX = endingPixelX
                    else:
                        endingPixelX = (i + 2) * processedImageShapeX
                        subImageEndingPixelX = endingPixelX - processedImageShapeX

                    #Copy the sub image from the main image
                    processingSubImage = currentImage[startingPixelY:endingPixelY, startingPixelX:endingPixelX]
                    #block
                    #sleep(random())
                    #add to the queue
                    #addd the sub image (1D array) first
                    #followed by the position
                    queue.put(processingSubImage)
                    queue.put(startingPixelX)
                    queue.put(startingPixelY)
                    queue.put(subImageStartingPixelX)
                    queue.put(subImageEndingPixelX)
                    queue.put(subImageStartingPixelY)
                    queue.put(subImageEndingPixelY)
                    #block
                    sleep(random())
            
            queue.put('OneLayerDone')
            #block
            sleep(random())
            
        #all done
        queue.put(None)
        print("Producer: Done", flush=True)

    def processingImage(self, queue):
        print("Consumer: Running", flush = True)
        #consume work
        while True:
            txtIsBegin = queue.get()
            #print(txtIsBegin)
            if txtIsBegin == 'OneLayerBegin':
                layer_number = queue.get()
                imageShapeX = queue.get()
                imageShapeY = queue.get()
                passedFinalOutputImage = np.ones((int(imageShapeY), int(imageShapeX)), dtype = np.uint16)
            
            if txtIsBegin is None:
                break

            while True:
                #get a unit of work
                inputSubImage = queue.get()
                if inputSubImage == 'OneLayerDone':
                    break
                startingPixelX = queue.get()
                startingPixelY = queue.get()
                subImageStartingPixelX = queue.get()
                subImageEndingPixelX = queue.get()
                subImageStartingPixelY = queue.get()
                subImageEndingPixelY = queue.get()
                #Print the image starting pixels positions for debugging purposes
                #print('{}, {}, {}, {}, {}, {}'.format(startingPixelX, startingPixelY, subImageEndingPixelX, subImageEndingPixelX, subImageStartingPixelY, subImageEndingPixelY))
                #Start processing the current sub image
                print("Processing one sub image on layer {}".format(layer_number), flush=True)
                #normalise the sub image first
                subimageMax = np.amax(inputSubImage)
                if subimageMax > 1:
                    normed_inputSubImage = np.divide(inputSubImage, subimageMax)
                if subimageMax == 1:
                    normed_inputSubImage = np.substrate(inputSubImage, 1)
                #Loop through the iterations
                for i in range(self.iterations - 1):
                    #convolute the sub image
                    convoluteInputSubImage = ndimage.gaussian_filter(inputSubImage, self.sigma, order=0, output=None, mode='constant')
                    #normed_inputSubImage = self.ftnNormalisingImage2(normed_inputSubImage, convoluteInputSubImage)
                    normed_inputSubImage = self.ftnNormalisingImageVectorisation(normed_inputSubImage, convoluteInputSubImage)
                #normed_inputSubImage = self.calculateFinalGreyscaleLevel(normed_inputSubImage)
                normed_inputSubImage = self.ftnCalculateFinalGreyscaleLevelVectorisation(normed_inputSubImage)
                normed_inputSubImage = normed_inputSubImage.astype('int')
                normed_inputSubImage = np.clip(normed_inputSubImage, 0, 255)
                print('Returning the processed sub image back to the final array on layer {}'.format(layer_number))
                initialPixelPositionY = subImageStartingPixelY - startingPixelY
                initialPixelPositionX = subImageStartingPixelX - startingPixelX
                endPixelPositionY = initialPixelPositionY + subImageEndingPixelY - subImageStartingPixelY
                endPixelPositionX = initialPixelPositionX + subImageEndingPixelX - subImageStartingPixelX
                passedFinalOutputImage[initialPixelPositionY:endPixelPositionY, initialPixelPositionX:endPixelPositionX] = normed_inputSubImage
                print('Successfully returned one sub image on layer {}.'.format(layer_number))
    
            #One layer done. Save the output image.
            #Save the output image as bmp
            outputImageNameBMP = "Layer " + str(layer_number) + ".bmp"
            outputCurrentImageFullPathBMP = os.path.join(self.current_directory, outputImageNameBMP)
            cv2.imwrite(outputCurrentImageFullPathBMP, passedFinalOutputImage)
            #Save the output image as STR
            outputImageName = "Layer " + str(layer_number) + ".str"
            outputCurrentImageFullPath = os.path.join(self.current_directory, outputImageName)
            self.ftnSaveSTRFile2(passedFinalOutputImage, outputCurrentImageFullPath)
                
        #all done
        print('Consumer: Done', flush=True)

    def ftnNormalisingImage2(self, originalImage, convolutedImage):
        #Change the data type of the input image array and the convoluted image array
        #npArrayInputImage = np.float32(originalImage)
        npArrayConvolutedImage = np.float32(convolutedImage)

        #print("Max intensity diffrence in the convoluted image...")
        #intensityDifference = np.amax(npArrayConvolutedImage) - np.amin(npArrayConvolutedImage)
        intensitySTD = np.std(npArrayConvolutedImage)

        #Calculate the correction factors
        #Method 1
        correctFactorArray = np.add((1 - npArrayConvolutedImage), originalImage, where=np.around(originalImage,1)!=0)
        max_correctFactorArray = np.amax(correctFactorArray)
        normed_ConvolutedImage = np.divide(correctFactorArray, max_correctFactorArray)

        npOutputArray = normed_ConvolutedImage
        return npOutputArray

    def ftnNormalisingImageVectorisation(self, originalImage, convolutedImage):
        #Change the data type of the input image array (originalIage) and the convoluted image array (convolutedImage)
        npArrayConvolutedImage = np.float32(convolutedImage)
        pdDFConvolutedImage = pd.DataFrame(npArrayConvolutedImage)
        #npArrayOriginalImage = np.float32(originalImage)
        pdDFOriginalImage = pd.DataFrame(originalImage)

        npCorrectFactorArray = np.add((1 - pdDFConvolutedImage).to_numpy(), pdDFOriginalImage.to_numpy(), where=np.around(originalImage,1)!=0)
        pdDFCorrectFactorArray = pd.DataFrame(npCorrectFactorArray)
        max_correctFactorArray = np.amax(npCorrectFactorArray)
        pdDFMax_correctorFactor = pd.DataFrame(max_correctFactorArray)
        npNormed_ConvolutedImage = np.divide(pdDFCorrectFactorArray.to_numpy(), pdDFMax_correctorFactor.to_numpy())

        return npNormed_ConvolutedImage

    def calculateFinalGreyscaleLevel(self, inputImage):
        npArrayInputImage = np.float32(inputImage)
        #Calculate the greyscale level for a give pixels dot
        normalisingGreyScaleLevel = self.calculateGreyscaleLevelDot()
        #Convolute the inputImage
        convolvedImage = ndimage.gaussian_filter(npArrayInputImage, self.sigma, order=0, output=None, mode='constant')
        #Normalise the greyscale levels in the output image
        normalisedConvolvedImage = np.divide(normalisingGreyScaleLevel, convolvedImage, where=np.around(npArrayInputImage,1)!=0)
        npArrayOutputImage = np.multiply(normalisedConvolvedImage, npArrayInputImage, where=np.around(npArrayInputImage,1)!=0) * 255.0

        return npArrayOutputImage

    def ftnCalculateFinalGreyscaleLevelVectorisation(self, inputImage):
        npArrayInputImage = np.float32(inputImage)
        #Vectorisation
        pdDFnpArrayInputImage = pd.DataFrame(npArrayInputImage)
        #Calculate the greyscale level for a give pixels dot
        normalisingGreyScaleLevel = self.calculateGreyscaleLevelDot() * 255.0
        #Convolute the inputImage
        convolvedImage = ndimage.gaussian_filter(npArrayInputImage, self.sigma, order=0, output=None, mode='constant')
        #Vectorisation
        pdDFConvolvedImage = pd.DataFrame(convolvedImage)
        #Normalise the greyscale levels in the output image
        normalisedConvolvedImage = np.divide(normalisingGreyScaleLevel, pdDFConvolvedImage.to_numpy(), where=np.around(npArrayInputImage,1)!=0)
        #Vectorisation
        pdDFNormalisedConvolvedImage = pd.DataFrame(normalisedConvolvedImage)
        npArrayOutputImage = np.multiply(pdDFNormalisedConvolvedImage.to_numpy(), pdDFnpArrayInputImage.to_numpy(), where=np.around(npArrayInputImage,1)!=0)

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
        returnedGreyscaleLevel = 0.0
        returnedGreyscaleLevel = np.amax(convolvedNP2DArray)

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


