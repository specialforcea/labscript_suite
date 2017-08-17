# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 16:21:43 2013

A batch processor designed to study the intensity dependance of absorption images

@author: ispielma
"""

import numpy
import qgasfileio
import qgasutils
import qgasfunctions
import scipy

# Some physical constants (these) will be local to this module
hbar = 1.05457e-34;
m40K = 6.63618e-26;
m87Rb = 1.44316e-25;
lambda40K = 767e-9;

#
# First write some basic functions that solve the three state problem
#



#
# Batch loading of data files
# 

def BatchProcess(FileTuple, PulseTime):
    
    points = len(FileTuple);
    index = 0;
        
    # Define arrays to store variables xxxx in the Image["ExpInf"]["xxxx"]
    # these will appear as arrays in the batch processed data
    # remember following comma for single element tuples.
    VarsToIndex = ("Scan",);#was KprobeInt
    Results = {name: numpy.array([]) for name in VarsToIndex}

    # Add dictionary items to be filled by the code below
    Results["OD"] = numpy.array([]);
    Results["OD_FIT"] = numpy.array([]);
    Results["OD_BG"] = numpy.array([]);
    Results["Counts"] = numpy.array([]);
    Results["CountsVar"] = numpy.array([]);
    
    # Define location of boxes
    r0 = numpy.array([0,0]); # Note that this is ordered so [0] is x and [1] is y
    WidthFit = numpy.array([350,350]);
    WidthPeak = numpy.array([20,20]);

    # Prepare a graphical view to hold any status updates
#    matplotlib.rcParams['toolbar'] = 'None';
#    fig, ax = matplotlib.pyplot.subplots(1,3, figsize=(8,4))
#    ax[0].set_xlabel('x position');
#    ax[0].set_ylabel('y position');
#    ax[1].set_xlabel('x position');
#    ax[1].set_ylabel('y position');


    for FileName in FileTuple:
        # Load the image and construct desired OD by hand    
        Image = qgasfileio.LoadImg(FileName);
        
        
        #
        # Do fancy-pants analysis
        #

        RawSmooth = scipy.ndimage.gaussian_filter(Image["Raw"][1], sigma=4);
        RawNoise = Image["Raw"][1] - RawSmooth;

    
        (xValsFit, yValsFit, OptDepthFit) = qgasutils.ImageCrop(Image["xVals"], Image["yVals"], Image["OptDepth"], r0, WidthFit, Scaled = True);
        xyValsFit = (xValsFit, yValsFit);

        (xValsPeak, yValsPeak, OptDepthPeak) = qgasutils.ImageCrop(Image["xVals"], Image["yVals"], Image["OptDepth"], r0, WidthPeak, Scaled = True);
        (xValsPeak, yValsPeak, RawPeak) = qgasutils.ImageCrop(Image["xVals"], Image["yVals"], Image["Raw"][1], r0, WidthPeak, Scaled = True);
        (xValsPeak, yValsPeak, RawPeakNoise) = qgasutils.ImageCrop(Image["xVals"], Image["yVals"], RawNoise, r0, WidthPeak, Scaled = True);
    
        p_guess = (+0.05,0.6,2,46,2,46);
        res = qgasfunctions.curve_fit_qgas(qgasfunctions.gaussian2D, p_guess, OptDepthFit, xyValsFit, full_output=1);

        #
        # Update variabes
        #
    
        # test for accept or reject conditions
        if (RawPeak.max() < 4095 ):
            # update the variable defined arrays
            for name in VarsToIndex:
                Results[name].resize(index+1);
                Results[name][index] = Image["ExpInf"][name];


            # Fill in arrays from analysis
            Results["OD"].resize(index+1);
            Results["OD"][index] = OptDepthPeak.mean();
        
            Results["OD_FIT"].resize(index+1);
            Results["OD_FIT"][index] = res[0][1];
            
            Results["OD_BG"].resize(index+1);
            Results["OD_BG"][index] = res[0][0];
        
            Results["Counts"].resize(index+1);
            Results["Counts"][index] =  RawPeak.mean();

            Results["CountsVar"].resize(index+1);
            Results["CountsVar"][index] =  RawPeakNoise.var();

            index += 1;

#            ax[0].imshow(Image["Raw"][1]);
#            ax[1].imshow(RawSmooth);
#            ax[2].imshow(RawPeakNoise);
#            fig.canvas.draw();
#
#            return None;

        # Now update graphical view
#        ax[0].clear();
#        ax[0].imshow(Image["OptDepth"]);
#        ax[0].axis('tight');
#
#        ax[1].clear();
#        ax[1].imshow(Image["Raw"][1]);
#        ax[1].axis('tight');
#        
#        fig.canvas.draw();
        
        
    #
    # Print and report the slope of the variance data
    #
        
    p_guess = (0,1);
    res = qgasfunctions.curve_fit_qgas(qgasfunctions.line, p_guess, Results["CountsVar"], (Results["Counts"],), full_output=1);
    print "Residual Offset variance (read noise) = ", res[0][0], "; slope =", res[0][1]
    
    #
    # Now try to reprocess this with different guesses for ISatCounts
    #
    
    Results["OD"] -= Results["OD_BG"].mean();  # now remove background
    Results["OD_FIT"] -= Results["OD_BG"].mean();  # now remove background    
    
    ISatCounts = numpy.linspace(1,5000,num=128)
    Sdev = numpy.zeros_like(ISatCounts);
    
    index = 0;

    tau = 1/(hbar*(2*numpy.pi/lambda40K)**2 /(2*m40K));
    for ISat in ISatCounts:
        ODCorrect = qgasfunctions.CorrectOD(Results["OD"], Results["Counts"], PulseTime, ISat, tau);

        p_guess = (0,1);
        res = qgasfunctions.curve_fit_qgas(qgasfunctions.line, p_guess, ODCorrect, (Results["Counts"],), full_output=1);

        Sdev[index]=res[0][1]/res[0][0];

        index += 1;
    
    Results["ISatCounts"] = ISatCounts;
    Results["Sdev"] = Sdev;
#    matplotlib.pyplot.close(fig)
#    return (Results, fig);     
    return Results;     

    