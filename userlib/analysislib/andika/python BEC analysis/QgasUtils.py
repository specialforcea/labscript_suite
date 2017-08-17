"""
Created on Mon Sep  9 15:51:35 2013

QgasUtils: Basic Quantum Gas Utilities functions
@author: ispielman

Modified on Wed Dec 10 11:26: 2014
@author: aputra
"""

import numpy
import scipy.ndimage


def ImageSlice(xVals, yVals, Image, r0, Width, Scaled = False):
    """
    Produces a pair of slices from image of a band with 'Width' centered at r0 = [x y]
    
    Scaled : 'False' use pixels directly, and 'True' compute scaling from (xvals and yvals) assuming
        they are linearly spaced
    
    Currently Width and x,y are in scaled units, not pixel units.
    the return will be ((xvals xslice) (yvals yslice)), where each entry is a numpy array.
    these are copies, not views.
    """
    if (Scaled):
        (xMin, yMin) = numpy.floor(GetPixelCoordsFromImage(r0, - Width/2, xVals, yVals));
        (xMax, yMax) = numpy.ceil(GetPixelCoordsFromImage(r0,  Width/2, xVals, yVals));
    else:
        (xMin, yMin) = r0 - numpy.round(Width/2);
        (xMax, yMax) = r0 + numpy.round(Width/2);
 
    # Extract bands of desired width
    # These are slices, so views of the initial data

    if xMin<0: xMin =0
    if yMin<0: yMin =0
    if xMax>xVals.shape[1]: xMax = xVals.shape[1]
    if yMax>yVals.shape[0]: yMax = yVals.shape[0]

    # Compute averages
    ySlice = Image[:,xMin:xMax].mean(1); # along y, so use x center
    xSlice = Image[yMin:yMax,:].mean(0); # along x, so use y center
    
    yValsSlice = yVals[:,0].copy();
    xValsSlice = xVals[0,:].copy();

    return ((xValsSlice, xSlice), (yValsSlice, ySlice));


def ImageCrop(xVals, yVals, Image, r0, Width, Scaled = False, Center = True):
    """
    crops an image along with the associated matrix of x and y 
    to a specified area and returns the cropped image
    this will be a copy not a view
    
    Image, xVals, yVals : (2D image, xvals, yvals) 
    r0                  : center of ROI in physical units (two element list or array)
    Width               : length of box-sides in physical units (two element list or array)

    Scaled  : If true, will attempt to use the x and y waves, to generate pixel values
    Center  : Recenter on cropped region
    """   
    error = False;    
    Cropped_Image={'OptDepth':0,'xVals':0,'yVals':0,'Error':error}
    if (Scaled):
        if(ScaleTest(xVals, yVals)):
            rMinPixel = numpy.floor(GetPixelCoordsFromImage(r0, -Width/2, xVals, yVals));
            rMaxPixel = numpy.ceil(GetPixelCoordsFromImage(r0, Width/2, xVals, yVals));
        else:
            rMinPixel = numpy.floor(r0)-numpy.floor(Width/2);
            rMaxPixel = numpy.ceil(r0)+numpy.ceil(Width/2);
            error = True;
    else:
        rMinPixel = numpy.floor(r0)-numpy.floor(Width/2);
        rMaxPixel = numpy.ceil(r0)+numpy.ceil(Width/2);

    if rMinPixel[0]<0: rMinPixel[0]=0
    if rMinPixel[1]<0: rMinPixel[1]=0
    if rMaxPixel[0]>xVals.shape[1]: rMaxPixel[0] = xVals.shape[1]
    if rMaxPixel[1]>yVals.shape[0]: rMaxPixel[1] = yVals.shape[0]

    Cropped_Image['OptDepth'] = Image[rMinPixel[1]:rMaxPixel[1],rMinPixel[0]:rMaxPixel[0]].copy();
    Cropped_Image['xVals'] = xVals[rMinPixel[1]:rMaxPixel[1],rMinPixel[0]:rMaxPixel[0]].copy();
    Cropped_Image['yVals'] = yVals[rMinPixel[1]:rMaxPixel[1],rMinPixel[0]:rMaxPixel[0]].copy();
    
    if (Center):
        Cropped_Image['xVals'] -= r0[0];
        Cropped_Image['yVals'] -= r0[1];
    
    return Cropped_Image;


def ImageSliceFromMax(Image, width, pScale = True):
    """
    Produces a pair of slices from image of a band with 'Width' centered at the maximum val of Image

    Scaled : 'False' use pixels directly, and 'True' compute scaling from (xvals and yvals) assuming
        they are linearly spaced

    Currently Width and x,y are in scaled units, not pixel units.
    the return will be ((xvals xslice) (yvals yslice)), where each entry is a numpy array.
    these are copies, not views.
    """

    Z = scipy.ndimage.gaussian_filter(Image['OptDepth'], sigma=3);
    id = Z.argmax()
    r0max = (numpy.ravel(Image['xVals'])[id], numpy.ravel(Image['yVals'])[id])
    imgSlice = ImageSlice(Image['xVals'], Image['yVals'], Image['OptDepth'], r0max, width, Scaled = pScale)
    imgSlicefromMax={'xVals':0,'yVals':0,'xSlice':0, 'ySlice':0, 'xMax':r0max[0], 'yMax':r0max[1]}
    imgSlicefromMax['yVals'] = imgSlice[1][0]
    imgSlicefromMax['xVals'] = imgSlice[0][0]
    imgSlicefromMax['ySlice'] = imgSlice[1][1]
    imgSlicefromMax['xSlice'] = imgSlice[0][1]
    return imgSlicefromMax


def GetPixelCoordsFromImage(r0, Offset, xVals, yVals):
    """
    Returns the pixel coordinates associated with the scaled values in the 2D arrays xVals and yVals
    remember in r0 the ordering is r0 = (x0, y0)
    """ 
    # Assume that the correct arrays were passed
    dy = yVals[1][0] - yVals[0][0];
    dx = xVals[0][1] - xVals[0][0];
    y0 = yVals[0][0];
    x0 = xVals[0][0];
    #want offset to be an integer number of pixels
    Offset = numpy.round(Offset/numpy.array([dx,dy]));
  
    return (r0 - numpy.array([x0, y0])) /numpy.array([dx, dy])+Offset;
    
    
def ScaleTest(xVals, yVals):
    """
    Returns the pixel coordinates associated with the scaled values in the 2D arrays xVals and yVals
    remember in r0 the ordering is r0 = (x0, y0)
    """     
    # Assume that the correct arrays were passed
    dy = yVals[1][0] - yVals[0][0];
    dx = xVals[0][1] - xVals[0][0];
    
    if ((dx == 0) or (dy == 0)):
        print("ImageSlice: generating scaled axes failed");
        print(dx,dy,xVals[0][1],xVals[0][0],yVals[1][0],yVals[0][0],xVals,yVals)
        return False;
    else:
        return True;