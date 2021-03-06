"""
Created on Thu Sep  5 12:45:39 2013
@author: ispielman

Modified on Wed Dec 10 10:57 2014
@author: aputra

QgasFileIO: Quantum Gas File IO procedures
"""

import IgorLoad as igor
import numpy as np


#==============================================================================
# IAN'S DESCRIPTION:
#
# The basic method of file loading will be to:
#     (1) Load a file into a new dictionary.
#     (2) Register the basic variables
#         * Fill basic variables (including all provided in the string variables)
#     (3) decide on the basic processing module
#         * register the variables to be generated by the processing
#         * perform processing and fill
# Thus each loaded-image instance will contain only variables
# and those will have been generated by that file
# 
# Unlike the igor code, this implies that each loaded-file will only
# contain the required variables and there will never be extra's.
# 
# At the higher level, a sequence will then be a dictionary of images
# from which we can generate arrays of variables.
#
#==============================================================================


def LoadImg(FileName):
    """LoadImg: This loads an IGOR binary file saved by labview.
    Loads an image from igor and extracts a bunch of interesting
    information (inf) from the image header"""

    IBWData = igor.LoadIBW(FileName);

    # I am going to store the experimental information in a dictionary
    Image = {"Note": IBWData["Note"]};

    # Add error checking here

    # Internal properties associated with igor waves
    ExpInf = {"FileName": FileName};

    # Begin acquitting experimental information from note
   
    # I should consider passing some sort of default wave above.
    ExpInf["Camera"] = igor.StringByKey('', 'Camera_ID', Image["Note"],'=','\r\n');
    ExpInf["Experiment"] = igor.StringByKey('', 'Experiment', Image["Note"],'=','\r\n');
    ExpInf["DataType"] = igor.StringByKey('', 'DataType', Image["Note"],'=','\r\n');
    ExpInf["ImageType"] = igor.StringByKey('', 'ImageType', Image["Note"],'=','\r\n');
    ExpInf["magnification"] = igor.NumberByKey(1, 'Magnification', Image["Note"],'=','\r\n');
    ExpInf["ImageDirection"] = igor.StringByKey('', 'ImageDirection', Image["Note"],'=','\r\n');
    
    # Indexed waves and variables contain all the information passed by the sequencer
    ExpInf["IndexedWaves"] = igor.StringByKey('', 'IndexedWaves', Image["Note"],'=','\r\n');
    ExpInf["IndexedValues"] = igor.StringByKey('', 'IndexedValues', Image["Note"],'=','\r\n');
    ExpInf["IndexedWaves"] = ExpInf["IndexedWaves"].strip(";\n\r");     #cleaning up extra characters
    ExpInf["IndexedValues"] = ExpInf["IndexedValues"].strip(";\n\r");   #cleaning up extra characters
    
    # Igor runs a function called SetCamDir here, but it seemed to just
    # turn "ImageDirection" into a string.
   
    if (ExpInf["Experiment"] == 'Rubidium_II'):
        ExpInf["Ibias"] = igor.NumberByKey(0,'I_Bias', Image["Note"],'=','\r\n');
        ExpInf["expand_time"] = igor.NumberByKey(1,'t_Expand', Image["Note"],'=','\r\n');
        ExpInf["detuning"] = igor.NumberByKey(0,'Detuning',Image["Note"],'=','\r\n');
    elif (ExpInf["Experiment"] == 'Rubidium_I'):
        ExpInf["Irace"] = igor.NumberByKey(0,'I_Race', Image["Note"],'=','\r\n');
        ExpInf["Ipinch"] = igor.NumberByKey(0,'I_Pinch', Image["Note"],'=','\r\n');
        ExpInf["trapmin0"] = igor.NumberByKey(0,'Trap_Min', Image["Note"],'=','\r\n');
        ExpInf["Ibias"] = igor.NumberByKey(0,'I_Bias', Image["Note"],'=','\r\n');
        ExpInf["expand_time"] = igor.NumberByKey(1,'t_Expand', Image["Note"],'=','\r\n');
        ExpInf["detuning"] = igor.NumberByKey(0,'Detuning', Image["Note"],'=','\r\n');
    elif (ExpInf["Experiment"] == 'Rubidium_Chip'):
        ExpInf["expand_time"] = igor.NumberByKey(1,'t_Expand', Image["Note"],'=','\r\n');
        ExpInf["detuning"] = igor.NumberByKey(0,'Detuning', Image["Note"],'=','\r\n');
    else:
        ExpInf["expand_time"] = igor.NumberByKey(1,'t_Expand', Image["Note"],'=','\r\n');
        ExpInf["detuning"] = igor.NumberByKey(0,'Detuning', Image["Note"],'=','\r\n');

    # There is a bunch of stuff in the Igor code here dealing with indexes

    # Now that the data is out of the file, we can begin to work with it.
    # First break the image into its composite pieces

    if (ExpInf["ImageType"] == "Raw"): # In this case the loaded file contains three images which needed 
                                       # to be split apart and processed.
        # First split into two/three/four images and save as a list of raw data

        ImageDimensions = IBWData["Data"].shape;
        Raw = []; 
        if (ExpInf["Camera"] == "PIXIS"):
            xsize = ImageDimensions[1]/4;
            Raw.append(IBWData["Data"][:,0:xsize-1].astype(float));
            Raw.append(IBWData["Data"][:,xsize:2*xsize-1].astype(float));
            Raw.append(IBWData["Data"][:,2*xsize:3*xsize-1].astype(float));
            Raw.append(IBWData["Data"][:,3*xsize:4*xsize-1].astype(float));
            
        elif (ExpInf["Camera"] == "PIXISSlowReadOut"):
            # In this case the loaded file contains 1 frame with 2 images
            xsize = ImageDimensions[1]/2;
            Raw.append(IBWData["Data"][:,0:xsize-1].astype(float));
            Raw.append(IBWData["Data"][:,xsize:2*xsize-1].astype(float));
            
        else:
            xsize = ImageDimensions[1]/3;
            Raw.append(IBWData["Data"][:,0:xsize-1].astype(float));
            Raw.append(IBWData["Data"][:,xsize:2*xsize-1].astype(float));
            Raw.append(IBWData["Data"][:,2*xsize:3*xsize-1].astype(float));
            

        # Now store the base images, to be considered immutable
        Image["Raw"] = Raw;

        # Create the correct data
        if (ExpInf["DataType"] == "Absorption" or ExpInf["DataType"] == "Absorbsion"):
            # Account for historical spelling error.
            # Conditional is used to add one to pixels with zero counts in the denominator
            Image["OptDepth"] = -np.log(((Image["Raw"][0]<1)+Image["Raw"][0])/((Image["Raw"][1] < 1) + Image["Raw"][1]));
        elif (ExpInf["DataType"] == "Fluorescence"):
            Image["OptDepth"] = Image["Raw"][0] - Image["Raw"][1];
        elif (ExpInf["DataType"] == "PhaseContrast"):
            Image["OptDepth"] = Image["Raw"][0]/Image["Raw"][1];
        elif (ExpInf["DataType"] == "RawImage"):
            Image["OptDepth"] = Image["Raw"][0].copy();
        else:
            # Conditional is used to add one to pixels with zero counts in the denominator
            Image["OptDepth"] = Image["Raw"][0]/((Image["Raw"][1] < 1) * np.abs(Image["Raw"][1] + 1) + Image["Raw"][1]);

    else:	# Data is pre-processed (difference is for fluorescence, ratio is for absorption)
        Image["OptDepth"] = IBWData["Data"].astype(float);

    Image["ExpInf"] = ExpInf;

    # Update based on properties of the camera
    # This should be updated AFTER the image is broken into Raw1 to Raw4
    Update_Magnification(Image); 

    # Extract any optional variables associated with this wave
    ExpandIndexedString(Image);

    # Generate arrays of the x and y values
    yarray = np.linspace(Image["ExpInf"]["x0"][0],Image["ExpInf"]["x1"][0],Image["OptDepth"].shape[0]);
    xarray = np.linspace(Image["ExpInf"]["x0"][1],Image["ExpInf"]["x1"][1],Image["OptDepth"].shape[1]);
    (xVals,yVals) = np.meshgrid(xarray,yarray);
    Image["xVals"] = xVals;
    Image["yVals"] = yVals;

    return Image;


def LoadSC(FileName):
    """LoadSC: This loads an IGOR binary file saved by LabView.
    Loads LabView Scope data from igor and extracts a bunch of interesting
    information (inf) from the data header"""

    IBWData = igor.LoadIBW(FileName);

    # I am going to store the experimental information in a dictionary
    SCdata = {"Note": IBWData["Note"], "Data": IBWData["Data"]};

    # ====================================
    # From lab book on 2014_11_18
    # TDS2014_GetTraces_DT should be working fine now. No glitches anymore but the data header is different.
    #
    # The first 6 numbers is what you need to rescale the data using
    #
    # Xreal = XZERO + XINCR*(data - PT_OFF)
    # Yreal = YZERO + YMULT*(data - YOFF)
    #
    # and they appear in the exact order as above. Next you get #4 which is useless and then number of points.
    # ====================================

    SCtime = np.zeros((4,2500))
    SCdiode = np.zeros((4,2500))

    for kk in range(4):
        fac = ''.join(chr(i) for i in SCdata["Data"][kk][:36])
        a1 = np.fromstring(fac, dtype=float, sep=";")
        SCtime[kk] = a1[0] + a1[1]*(np.linspace(1,2500,2500) - a1[2])
        SCdiode[kk] = a1[3] + a1[4]*(SCdata["Data"][kk][-2501:-1] - a1[5])

    return SCtime, SCdiode


def LoadAI(FileName):
    """LoadSC: This loads an IGOR binary file saved by LabView.
    Loads LabView Scope data from igor and extracts a bunch of interesting
    information (inf) from the data header"""

    IBWData = igor.LoadIBW(FileName);

    # I am going to store the experimental information in a dictionary
    AIdata = {"Note": IBWData["Note"], "Data": IBWData["Data"]};
    return AIdata

# TODO: Consider editing FileNameTuple function. Or use "import os" for file IO related stuff
def FileNameTuple(FileHeaderTuple, InitialIndexTuple, FinalIndexTuple, Extension = ""):
    """
    FileNameTuple : this function returns a tuple of files that should be evaluated
        Taken as parameters are tuple of file name headers,
        starting indices and ending indices
    """
    
    # Verify that all tuples have non-zero length
    if (len(FileHeaderTuple) == 0 or len(InitialIndexTuple) == 0 or len(FinalIndexTuple) == 0):
        print("FileNameTuple: passed initial object of zero length");
        return ();
    
    # Verify that both index tuples have the same length
    if (len(InitialIndexTuple) != len(FinalIndexTuple)):
        print("FileNameTuple: index tuples have different length");
        return ();
    
    # Make the NameLen tuple the same length as the others by padding with its last element
    NameLen = len(FileHeaderTuple);
    IndexLen =  len(InitialIndexTuple);
    
    if (NameLen < IndexLen):
        FileHeaderTupleLong = FileHeaderTuple + (FileHeaderTuple[-1],)*(IndexLen-NameLen);
    else:
        FileHeaderTupleLong = FileHeaderTuple;

    # Now build the tuple of file names
    FileList = [];
    for FileNameHeader, InitialIndex, FinalIndex in zip(FileHeaderTupleLong, InitialIndexTuple, FinalIndexTuple):
        Initial = int(InitialIndex);
        Final = int(FinalIndex);

        if (Final < Initial):
            print ("FileNameTuple: Initial index must be smaller than Final index");
            return ();
        
        # Since we want the last element, add one
        FileList += [FileNameHeader +  "{:0>4d}".format(i) + Extension for i in range(Initial, Final+1)];

    return tuple(FileList);


#========================================================================================
# 
# The below functions are helper functions intended for use only within QgasFileIO module
#
#========================================================================================

def Update_Magnification(Image):
# Uses the camera settings (pixel size) and to rescale image magnification. Units in microns

    if (Image["ExpInf"]["Camera"] == 'PixelFly'): # Pixelfly Camera
        delta_X = 6.45 / Image["ExpInf"]["magnification"];
        delta_Y = 6.45 / Image["ExpInf"]["magnification"];
    elif (Image["ExpInf"]["Camera"] == 'PI'):  # Old PI Camera
        delta_X = 15 / Image["ExpInf"]["magnification"];
        delta_Y = 15 / Image["ExpInf"]["magnification"];
    elif (Image["ExpInf"]["Camera"] == 'LG3'):  # LG3 Frame grabber
        delta_X = 10 / Image["ExpInf"]["magnification"];
        delta_Y = 10 / Image["ExpInf"]["magnification"];
    elif (Image["ExpInf"]["Camera"] == 'THOR'):  # Thorlabs CCD Cameras
        delta_X = 7.4 / Image["ExpInf"]["magnification"];
        delta_Y = 7.4 / Image["ExpInf"]["magnification"];
    elif (Image["ExpInf"]["Camera"] == 'Flea3'):  # PointGrey Flea3
        delta_X = 5.6 / Image["ExpInf"]["magnification"];
        delta_Y = 5.6 / Image["ExpInf"]["magnification"];
    elif (Image["ExpInf"]["Camera"] == 'PIXIS'):  # PI Pixis camera
        delta_X = 13 / Image["ExpInf"]["magnification"];
        delta_Y = 13 / Image["ExpInf"]["magnification"];
    elif (Image["ExpInf"]["Camera"] == 'PIXISSlowReadOut'):  # PI Pixis camera, slow readout
        delta_X = 13 / Image["ExpInf"]["magnification"];
        delta_Y = 13 / Image["ExpInf"]["magnification"];
    else :           
        delta_X = 1 / Image["ExpInf"]["magnification"];
        delta_Y = 1 / Image["ExpInf"]["magnification"];
    
    Image["ExpInf"]["dx"] = np.array([delta_X, delta_Y]);
    Image["ExpInf"]["x0"] = -np.array(Image["OptDepth"].shape) * Image["ExpInf"]["dx"] / 2;
    Image["ExpInf"]["x1"] = +np.array(Image["OptDepth"].shape) * Image["ExpInf"]["dx"] / 2;
 
    
def ExpandIndexedString(Image):
    """ExpandIndexedString Creates new variables in Image["ExpInf"] for
    IndexedWaves - IndexedValues pairs """
    
    # Indexed waves and variables contain all the information passed by the sequencer
    WaveNameList = Image["ExpInf"]["IndexedWaves"].split(";");
    WaveValuesList = Image["ExpInf"]["IndexedValues"].split(";");
    
    # Strip empty strings from both lists
    WaveNameListStrip = [x for x in WaveNameList if x != ""];
    WaveValuesListStrip = [float(x) for x in WaveValuesList if x != ""];
    
    # Verify that both lists have the same length
    if (len(WaveNameListStrip) == len(WaveValuesListStrip)):
        NewDict = {k : v for k,v in zip(WaveNameListStrip, WaveValuesListStrip)}
    else:
        NewDict = {};
    
    Image["ExpInf"].update(NewDict);