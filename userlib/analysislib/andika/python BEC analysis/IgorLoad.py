"""
Created on Wed Sep 4 08:07:01 2013
@author: ispielman

Modified on Wed Dec 10 10:47 2014
@author: aputra

This module contains functions to load igor binary files
into numerical python arrays

IgorLoad v1.0: initial version, seems to work, but limited data types and does not pull all scaling information
"""

import struct   #Interpret strings as packed binary data
import numpy
import re       #provides regular expression patterns.


def NumberByKey(Default, Key, TestString, keySepStr, listSepStr):
    "NumberByKey Extracts a number from a string by identifying a key"
    "Default        is the number to return upon failure"
    "Key            is the keyword to search for"
    "TestString     is the string to search"
    "keySepStr      is a token separating the Key from the number"
    "listSepStr     identifies when the number ends"
    
    RegExp = Key + r"\s*" + keySepStr + "\s*(?P<ExtractStr>[^" + listSepStr + r"]+)";
    matchObj = re.search(RegExp,TestString,re.I)
    if matchObj:
        MatchString = matchObj.group("ExtractStr");
        Num = float(MatchString);
    else:
        Num = Default;

    return Num;


def StringByKey(Default, Key, TestString, keySepStr, listSepStr):
    "StringByKey Extracts a string from a string by identifying a key"
    "Default        is the string to return upon failure"
    "Key            is the keyword to search for"
    "TestString     is the string to search"
    "keySepStr      is a token separating the Key from the number"
    "listSepStr     identifies when the number ends"
    
    RegExp = Key + r"\s*" + keySepStr + "\s*(?P<ExtractStr>[^" + listSepStr + r"]+)";
    matchObj = re.search(RegExp,TestString,re.I)
    if matchObj:
        MatchString = matchObj.group("ExtractStr");
    else:
        MatchString = Default;

    return MatchString;


def LoadIBW(filename):
    "Loads an igor binary file keeping the data and the note"
    with open(filename, "rb") as f:
        
        # Endianness
        # endian is zero big-endian byte ordering, otherwise little-endian byte ordering
        try:
            Endian = ord(f.read(1));    #converting a single character to its ASCII
        except:
            return "LoadIBW Error: File ", filename, ", endian byte not readable";
        
        # In the struct.unpack()'s that follow this will define the endian order
        EndianChar = '>' if (Endian==0) else '<'
        
        # Version
        # now reset file pointer (the endian information is partially encoded in the two-byte version)
        try:
            f.seek(0);
            VersionTuple = struct.unpack(EndianChar+"h",f.read(2));
            Version = VersionTuple[0];
        except:
            return "LoadIBW Error: File ", filename, ", version code not readable";
      
        # Full Header
        # reset the pointer (the version will be included in the header code)
        try:
            f.seek(0);
            if Version == 1:
                BinHeaderString = f.read(8);
                WaveHeaderString = f.read(110);
                BinHeaderTuple = struct.unpack(EndianChar+"2xl2x",BinHeaderString);
                WaveHeaderTuple = struct.unpack(EndianChar+"h40xl64x",WaveHeaderString);
                NoteSize=0;
                FormulaSize=0;
                Type = WaveHeaderTuple[0];
                Size = WaveHeaderTuple[1];
                Dimensions = Size,0,0,0;
            elif Version == 2:
                BinHeaderString = f.read(16);
                WaveHeaderString = f.read(110);
                BinHeaderTuple = struct.unpack(EndianChar+"2xll6x",BinHeaderString);
                WaveHeaderTuple = struct.unpack(EndianChar+"h40xl64x",WaveHeaderString);
                NoteSize=BinHeaderTuple[1];
                FormulaSize = 0;
                Type = WaveHeaderTuple[0];
                Size = WaveHeaderTuple[1];
                Dimensions = Size,0,0,0;
            elif Version == 3:
                BinHeaderString = f.read(20);
                WaveHeaderString = f.read(110);
                BinHeaderTuple = struct.unpack(EndianChar+"2xlll",BinHeaderString);
                WaveHeaderTuple = struct.unpack(EndianChar+"h40xl64x",WaveHeaderString);
                NoteSize=BinHeaderTuple[1];
                FormulaSize=BinHeaderTuple[2];
                Type = WaveHeaderTuple[0];
                Size = WaveHeaderTuple[1];
                Dimensions = Size,0,0,0;
            elif Version == 5:
                BinHeaderString = f.read(64);
                WaveHeaderString = f.read(320);
                BinHeaderTuple = struct.unpack(EndianChar+"4x4l44x",BinHeaderString);
                WaveHeaderTuple = struct.unpack(EndianChar+"12xlh50x4l236x",WaveHeaderString);
                NoteSize=BinHeaderTuple[2];
                FormulaSize=BinHeaderTuple[1];
                Type = WaveHeaderTuple[1];
                Size = WaveHeaderTuple[0];
                Dimensions = WaveHeaderTuple[2:6];
            else:
                print ("LoadIBW Error: File ", filename, " has an invalid version code ", Version);
                return -1;
        except:
            print ("LoadIBW Error: File ", filename, ", Bin or Wave header invalid");
            return -1;

        # Load Data
        # reset the pointer (the version will be included in the header code)
        # Complex not supported yet
        if (Type == 0x02):
            TypeString = EndianChar + "f";
        elif (Type == 0x04):
            TypeString = EndianChar + "d";
        elif (Type == 0x08): # Signed Char
            TypeString = EndianChar + "b";
        elif (Type == 0x48): # Unsigned Char
            TypeString = EndianChar + "B";
        elif (Type == 0x10): # Signed Int
            TypeString = EndianChar + "h";
        elif (Type == 0x50): # Unsigned Int
            TypeString = EndianChar + "H";
        elif (Type == 0x20): # Signed Long
            TypeString = EndianChar + "l";
        elif (Type == 0x60): # Unsigned Long
            TypeString = EndianChar + "L";
        else:
            print ("LoadIBW Error: unsupported data type", Type);
            return -1;
           
        dt = numpy.dtype(TypeString);
        try:
            Data = numpy.fromfile(f,dtype=dt,count=Size)
        except:
            print ("LoadIBW Error: File ", filename, ", data block read failed");
            return -1;
        Dims = tuple(y for y in reversed(Dimensions) if y);
        DataOut = numpy.reshape(Data, Dims);
        
        # Read the wave note if possible
        try:
            if Version == 1:
                NoteString = "";
            elif Version == 2:
                f.read(16); # padding
            elif Version == 3:
                f.read(16); # padding
            elif Version == 5:
                f.read(FormulaSize); # skip formula if present
            else:
                print ("LoadIBW Error: File ", filename, " has an invalid version code ", Version);
                return -1;
            if (NoteSize > 0): NoteString = f.read(NoteSize);
            else: NoteString = "";
        except:
            print ("LoadIBW Error: File ", filename, ", Note read failed");
            return -1;
    return {"Note": NoteString.decode('utf8'), "Data": DataOut};