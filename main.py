import numpy as np
import cv2
import PIL.Image                                                                # Avoid namespace issues
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math as m
import copy
from normxcorr2 import normxcorr2

xMax = 1280
yMax = 720

cropI = [0, 0, 1280, 720]

def convertImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(img)
def pushCompute(I1Compute, I2Compute, wSize, sSize):
        
    numRows_I = I1Compute.size[1]                                           # Number of rows of the image
    numCols_I = I1Compute.size[0]                                           # Number of columns of the image
    
    # Window and search half-sizes
    wSize2 = m.floor(wSize/2)                                               # Compute half-window size
    sSize2 = m.floor(sSize/2)                                               # Compute half-search size
    
    # Window centers
    wIndR = np.arange(wSize,(numRows_I-wSize2),wSize)                       # Find the centers of the row windows
    wIndC = np.arange(wSize,(numCols_I-wSize2),wSize)                       # Find the centers of the column windows
    
    # Meshgrid
    RR, CC = np.meshgrid(wIndR,wIndC)                                       # Create the row/column meshgrid
    RR     = np.transpose(RR)                                               # Transpose the row meshgrid
    CC     = np.transpose(CC)                                               # Transpose the column meshgrid
    
    # Initialize variables for the loops
    colPeak   = np.zeros((len(wIndR),len(wIndC)))                           # Column-shift peak
    rowPeak   = np.zeros((len(wIndR),len(wIndC)))                           # Row-shift peak
    colOffset = np.zeros((len(wIndR),len(wIndC)))                           # Actual column pixel shift
    rowOffset = np.zeros((len(wIndR),len(wIndC)))                           # Actual row pixel shift
    
    # Loop
    for i in range(len(wIndR)):                                             # Loop over all row window centers
        print("Iteration: ",i,"/",len(wIndR)-1)                             # Print current iteration to console
        for j in range(len(wIndC)):                                         # Loop over all column window centers
            
            # Get window centers
            rowCenter = wIndR[i]                                            # Current iteration row window center
            colCenter = wIndC[j]                                            # Current iteration column window center
            
            # Crop the image to WINDOW size
            # - im1 = im.crop((left, top, right, bottom))
            cropI1    = (colCenter-wSize2, rowCenter-wSize2,                # Define the crop region using the window center and the window size
                            colCenter+wSize2, rowCenter+wSize2)
            I1_Sub    = np.array(I1Compute.crop(cropI1))                    # Crop the image (Image 1) to its window size
                    
            # Crop the template to SEARCH size
            # - im1 = im.crop((left, top, right, bottom))
            cropI2    = (colCenter-sSize2, rowCenter-sSize2,                # Define the crop region using the window center and the search size
                            colCenter+sSize2, rowCenter+sSize2)
            I2_Sub    = np.array(I2Compute.crop(cropI2))                    # Crop the image (Image 2) to its search size
                    
            # Check whether window or search arrays have a constant value in entire array
            I1_Sub_SameVals = np.array_equal(I1_Sub,np.full(np.shape(I1_Sub),I1_Sub[0]))
            I2_Sub_SameVals = np.array_equal(I2_Sub,np.full(np.shape(I2_Sub),I2_Sub[0]))
            
            if (I1_Sub_SameVals or I2_Sub_SameVals):                        # If values in either matrix are all the same
                # print('I1_Sub/I2_Sub has same values in entire matrix!')    # Print notice (not an error)
                print(">")
                rPeak = 0                                                   # Set row peak to zero
                cPeak = 0                                                   # Set column peak to zero
                dR    = 0                                                   # Set sub-pixel row delta to zero
                dC    = 0                                                   # Set sub-pixel column delta to zero
            else:
                # Compute normalized cross-correlation
                c         = normxcorr2(I1_Sub,I2_Sub)                       # Compute the normalized cross-correlation between the two cropped images
                maxCInd   = np.unravel_index(c.argmax(), c.shape)           # Find (X,Y) tuple of max value of cross-correlation matrix
                rPeak     = maxCInd[1]                                      # Peak row-value from normalized cross-correlation
                cPeak     = maxCInd[0]                                      # Peak column-value from normalized cross-correlation
                c         = abs(c)                                          # Make sure none of the matrix values are negative (logs don't like negatives)
                c[c == 0] = 0.0001                                          # Make sure there are no zeros in the c matrix (logs don't like zeros)
                cC, rC    = np.shape(c)                                     # Get size of cross-correlation matrix (CC-matrix)
                
                # To avoid errors with subpixel peak point calculations
                if (rPeak == 0):                                            # If row peak is zero
                    rPeak = rPeak + 1                                       # Add one to row peak
                if (rPeak == rC-1):                                         # If row peak is one less than size of CC-matrix rows
                    rPeak = rC - 2                                          # Subtract one from row peak
                if (cPeak == 0):                                            # If column peak is zero
                    cPeak = cPeak + 1                                       # Add one to column peak
                if (cPeak == cC-1):                                         # If column peak is one less than size of CC-matrix columns
                    cPeak = cC - 2                                          # Subtract one from column peak
                            
                # Sub-pixel peak point (3-point Gaussian)
                numR = m.log(c[cPeak][rPeak-1]) - m.log(c[cPeak][rPeak+1])
                denR = 2*m.log(c[cPeak][rPeak-1]) - 4*m.log(c[cPeak][rPeak]) + 2*m.log(c[cPeak][rPeak+1])
                try:
                    dR   = numR/denR
                except ZeroDivisionError:
                    denR = 0.00000000001
                    dR   = numR/denR
                numC = m.log(c[cPeak-1][rPeak]) - m.log(c[cPeak+1][rPeak])
                denC = 2*m.log(c[cPeak-1][rPeak]) - 4*m.log(c[cPeak][rPeak]) + 2*m.log(c[cPeak+1][rPeak])
                dC   = numC/denC
            
                # Find the peak indices of the cross-correlation map
                colPeak[i][j] = cPeak + dC
                rowPeak[i][j] = rPeak + dR
                
                # Find the pixel offsets for X and Y directions
                colOffset[i][j] = colPeak[i][j] - wSize2 - sSize2 + 1           # Correct the column pixel shift for window and search sizes
                rowOffset[i][j] = rowPeak[i][j] - wSize2 - sSize2 + 1           # Correct the row pixel shift for window and search sizes
        
        # Move the contour plot over the image to correct location
        XX     = CC - np.min(CC)                                                # Shift X-values to origin of image
        YY     = RR - np.min(RR)                                                # Shift Y-values to origin of image
        scaleX = (cropI[2] - cropI[0])/np.max(XX)                     # Scale X-value to get sub-region correct size
        scaleY = (cropI[3] - cropI[1])/np.max(YY)                     # Scale Y-value to get sub-region correct size
        XX     = XX*scaleX + cropI[0]                                      # Shift the X-values to the sub-region
        YY     = YY*scaleY + cropI[1]                                      # Shift the Y-values to the sub-region

    return colOffset, rowOffset, CC, RR, XX, YY

index = 0
# while 1:
#     capture = cv2.VideoCapture(index)
#     capture.set(cv2.CAP_PROP_FRAME_WIDTH, xMax)
#     capture.set(cv2.CAP_PROP_FRAME_HEIGHT, yMax)
#     while cv2.waitKey(33) < 0:
#         ret, frame = capture.read()
#         cv2.imshow("VideoFrame", frame)
#     print("Index :", index)
#     if input('> ') == 1:
#         break
#     print("")
#     capture.release()
#     cv2.destroyAllWindows()
#     index += 1

capture = cv2.VideoCapture(1)
                        
capture.set(cv2.CAP_PROP_FRAME_WIDTH, xMax)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, yMax)

while cv2.waitKey(33) < 0:
    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)

ret, FirstFrame = capture.read()
cv2.imshow("FirstFrame", FirstFrame)

while cv2.waitKey(33) < 0:
    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)

ret, ReadFrame = capture.read()

capture.release()
cv2.destroyAllWindows()

FirstFrame = convertImage(FirstFrame)
ReadFrame = convertImage(ReadFrame)

I2Orig = ReadFrame

FirstFrame = FirstFrame.convert("L")
ReadFrame = ReadFrame.convert("L")

colOffset, rowOffset, CC, RR, XX, YY = pushCompute(FirstFrame, ReadFrame, 4, 8)


# Set Plot Option
contourType = 'Tot'
colorMap = 'jet'
alphaVal = 0.6
plotShow = [0, False, True, False, False, True]
def SetPlotOpt():
    global contourType, colorMap, alphaVal, plotShow
    print(f'\nSet colormap type.\nInput 0 for plasma, 1 for jet, 2 for bone, 3 for viridis:')
    k = -1
    while (not (0 <= k <= 3)):
        try:
            k = int(input('> '))
        except:
            continue 
    colorMap = ["plasma", "jet", "bone", "viridis"][k]

    print('\n')

    plotType = [0, 'Velocity', 'Contour', 'X Displacement', 'Y Displacement', 'Total Displacement']

    for i in range(1, 6):
        print(f'n. {i}\nShow [Plot:{plotType[i]}]?\nInput Y/N:')
        k = ''
        while (['Y', 'N', 'y', 'n'].count(k) == 0):
            k = input('> ')
        if (k == 'Y') or (k == 'y'):
            plotShow[i] = True
        else:
            plotShow[i] = False

        if i == 2 and plotShow[i] == True:
            print(f'Set alpha for contour.\nInput float 0 ~ 1:')
            alphaVal = float(input('> '))

            k = -1
            while (not (0 <= k <= 2)):
                print(f'Set displacement for contour.\nInput 0 for X, 1 for Y, 2 for Total:')
                k = int(input('> '))
            contourType = ["X", "Y", "Tot"][k]
            
        print("\n")

# Show Plot
quivX = None
quivY = None
quivU = None
quivV = None
quivVel = None

def PlotData(thresh):
    global quivX, quivY, quivU, quivV, quivVel

    # Get relevant instance variables
    # thresh   = 5                          
    quivX    = copy.deepcopy(CC)                                       # Set X-values from column meshgrid
    quivY    = copy.deepcopy(RR)                                       # Set Y-values from row meshgrid
    quivU    = copy.deepcopy(rowOffset)                                  # Set U-displacement from row offset
    quivV    = copy.deepcopy(colOffset)                                  # Set V-displacement from column offset

    quivU[abs(quivU) > thresh] = np.nan                                     # Apply min/max value threshold to U-displacement
    quivV[abs(quivV) > thresh] = np.nan                                     # Apply min/max value threshold to V-displacement
    quivVel = np.sqrt(np.square(quivU) + np.square(quivV))                  # Compute total displacement from thresholded values

    testVal = np.sqrt(thresh**2 + thresh**2)                                # Compute total displacement threshold
    quivU[quivVel == testVal]   = np.nan                                    # Apply threshold to X displacement
    quivV[quivVel == testVal]   = np.nan                                    # Apply threshold to Y displacement
    quivVel[quivVel == testVal] = np.nan                                    # Apply threshold to total displacement

# 1: velocity
def Plot1():

    plt.figure(1)                                                       # Select figure 4
    plt.close(1)                                                        # Close the figure
    plt.figure(1)                                                       # Select appropriate figure
    plt.cla()                                                           # Clear the axes
    plt.plot(quivX,quivY,'k.')                                          # Plot the window centers with black dots
    plt.quiver(quivX,quivY,quivU,-quivV,scale=None,color='r')           # Plot the velocity vectors (flip Y-displacement)
    plt.xlabel('X-Axis')                                                # Set X-label
    plt.ylabel('Y-Axis')                                                # Set Y-label
    plt.gca().invert_yaxis()                                            # Invert the Y-axis (to compare to MATLAB)
    plt.gca().set_aspect('equal')                                       # Set the axes to equal size
    plt.title("Displacement Vectors")                                   # Set plot title

# 2: contour
def Plot2():

    plt.figure(2)                                                       # Select figure 5
    plt.close(2)                                                        # Close figure 5
    fig2 = plt.figure(2)                                           # Select appropriate figure
    ax2  = fig2.add_subplot(111, aspect='equal')
    plt.cla()                                                           # Clear the axes
    plt.imshow(I2Orig, cmap = "gray")                              # Plot the original Image 2
    
    # Create a Rectangle patch
    rect = patches.Rectangle((cropI[0], cropI[1]),             # Create rectangle to show sub-region on image
                                cropI[2] - cropI[0],
                                cropI[3] - cropI[1],
                                edgecolor = 'k', facecolor = 'none',
                                linewidth = 1, linestyle = '--')
    ax2.add_patch(rect)                                            # Add the rectangle to the axes
    
    if (contourType == "X"): 
        target = quivU
    elif (contourType == "Y"):           
        target = quivV                                  
    elif (contourType == "Tot"):         
        target = quivVel
    
    plt.contourf(XX,YY,target,100,                      
                cmap = colorMap,
                extend = 'both',
                alpha = alphaVal,
                antialiased = True)
    plt.colorbar()                                                      # Display the colorbar

# 3: XDis
def Plot3():

    plt.figure(3)                                                       # Select figure 6
    plt.close(3)                                                        # Close the figure
    plt.figure(3)                                                       # Select appropriate figure
    plt.cla()                                                           # Clear the axes
    plt.contourf(quivX,quivY,quivU,100,                                 # Plot the X-displacement contour
                    cmap = colorMap, extend = 'both')
    plt.gca().invert_yaxis()                                            # Invert the Y-axis
    plt.gca().set_aspect('equal')                                       # Set the axes equal
    plt.colorbar()                                                      # Show the colorbar
    plt.title('X Displacement')                                         # Set the plot title

# 4: YDis
def Plot4():

    plt.figure(4)                                                       # Select figure 7
    plt.close(4)                                                        # Close the figure
    plt.figure(4)                                                       # Select appropriate figure
    plt.cla()                                                           # Clear the axes
    plt.contourf(quivX,quivY,quivV,100,                                 # Plot the Y-displacement contour
                    cmap = colorMap, extend = 'both')
    plt.gca().invert_yaxis()                                            # Invert the Y-axis
    plt.gca().set_aspect('equal')                                       # Set the axes equal
    plt.colorbar()                                                      # Show the colorbar
    plt.title('Y Displacement')                                         # Set the plot title

# 5: TotDis
def Plot5():

    plt.figure(5)                                                       # Select figure 8
    plt.close(5)                                                        # Close figure 8
    plt.figure(5)                                                       # Select appropriate figure
    plt.cla()                                                           # Clear the axes
    plt.contourf(quivX,quivY,quivVel,100,                               # Plot the total displacement contour
                    cmap = colorMap, extend = 'both')
    plt.gca().invert_yaxis()                                            # Invert the Y-axis
    plt.gca().set_aspect('equal')                                       # Set the axes equal
    plt.colorbar()                                                      # Show the colorbar
    plt.title('Total Displacement')                                     # Set the plot title

def showPlot():
    if(plotShow[1]):
        Plot1()                                                   
        plt.figure(1)
        plt.show()
        # print("Press ENTER to see next plot.")
        # input('> ')
    if(plotShow[2]):
        Plot2()
        plt.figure(2)
        plt.show()
        # print("Press ENTER to see next plot.")
        # input('> ')
    if(plotShow[3]):
        Plot3()
        plt.figure(3)
        plt.show()
        # print("Press ENTER to see next plot.", end='')
        # input('> ')
    if(plotShow[4]):
        Plot4()        
        plt.figure(4)
        plt.show()
        # print("Press ENTER to see next plot.", end='')
        # input('> ')
    if(plotShow[5]):
        Plot5()
        plt.figure(5)
        plt.show()
        # print("Press ENTER to see next plot.", end='')
        # input('> ')

# SetPlotOpt()
PlotData(5)
showPlot()
