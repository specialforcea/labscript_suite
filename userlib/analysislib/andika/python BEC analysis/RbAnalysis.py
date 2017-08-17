"""
Project   : python BEC analysis
Filename  : RbAnalysis

Created on  : 2014 Dec 14 00:52
Author      : aputra

A GUI interface for quantum gas analysis using Tkinter.
"""
# TODO: Error checking for out of bound values or commands is not yet included
# TODO: Make a list of procedures so it can be called through selections, i.e. Load Images + Three Slices + TF fit + etc


import Tkinter as tk
import ttk
import tkFileDialog
import QgasFileIO as IO
import QgasUtils as Util
import lmfit
import QgasFunctionsLmfit as Qfunc
import PhysConstants as phys
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from pylab import *
import numpy as np

import matplotlib
# matplotlib.use('TkAgg')

import scipy.ndimage
import os       # for directory / filename analysis
import h5py     # for hd5 data processing or?
from skimage.segmentation import join_segmentations

class Application(tk.Frame):

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.grid()
        self.parent.title("Rb Analysis")

        # Initialize values for GUI
        # TODO: initialize values for the GUI and include load or save procedure
        self.dataFolder = "/Users/andika28putra/Dropbox/December/17"
        # "/Users/andika28putra/Dropbox/December/17/AI_17Dec2014_2000.ibw"
        self.day = "17"
        self.month = "December"
        self.year = "2014"
        self.camera = "Flea3"
        self.idStart = 2200
        self.idEnd = 2230
        self.filename = self.dataFolder + "/" + self.camera + "_" + self.day + self.month[:3] + \
                       self.year + "_" + str("%.4d" % self.idStart) + ".ibw"
        self.ODimage = None
        # self.ODimage = IO.LoadImg(self.filename)

        # Create all the widgets
        self.createMenus()
        self.createEntries()
        self.createButtons()
        # self.createRadiobuttons()
        self.createComboboxes()
        self.createCanvases()


    def createComboboxes(self):
        labelYear = tk.LabelFrame(self, text = 'Year')
        years = ('2015', '2014', '2013', '2012')
        self.cbYear = ttk.Combobox(labelYear, values=years)
        self.cbYear.set(self.year)
        self.cbYear.grid()
        self.cbYear.bind("<<ComboboxSelected>>", self.yearSel)
        labelYear.grid(row=2, column=0)

        labelMonth = tk.LabelFrame(self, text = 'Month')
        months = ('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
                  'October', 'November', 'December')
        self.cbMonth = ttk.Combobox(labelMonth, values=months)
        self.cbMonth.set(self.month)
        self.cbMonth.grid()
        self.cbMonth.bind("<<ComboboxSelected>>", self.monthSel)
        labelMonth.grid(row=2, column=1)

        labelDay = tk.LabelFrame(self, text = 'Day')
        days = ["%.2d" % i for i in range(1,32)]  # somehow range(1,32) gives number from 1 to 31
        self.cbDay = ttk.Combobox(labelDay, values=days)
        self.cbDay.set(self.day)
        self.cbDay.grid()
        self.cbDay.bind("<<ComboboxSelected>>", self.daySel)
        labelDay.grid(row=2, column=2)

    def yearSel(self,event):
        self.year = self.cbYear.get()
        print(self.year)

    def monthSel(self,event):
        self.month = self.cbMonth.get()
        print(self.month)

    def daySel(self,event):
        self.day = self.cbDay.get()
        print(self.day)



    def createMenus(self):
        menubar = tk.Menu(self.parent)
        self.parent.config(menu=menubar)

        fileMenu = tk.Menu(menubar)
        menubar.add_cascade(label="File", menu=fileMenu)
        fileMenu.add_command(label="Set Folder...", command=self.askdirectory)
        fileMenu.add_command(label="Test print day", command=self.printDay)
        fileMenu.add_command(label="Test file index", command=self.printIndex)
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command=self.onExit)

    def printDay(self):
        self.day = self.cbDay.get()
        print(self.day)

    def printIndex(self):
        self.idStart = int(self.fileNoStart.get())
        self.idEnd = int(self.fileNoEnd.get())

        self.dataFolder = "/Users/andika28putra/Dropbox/December/05"
        self.camera = "Flea3"
        self.day = "05"
        self.year = "2014"
        self.month = "December"
        self.filename = self.dataFolder + "/Flea3_05Dec2014_0208.ibw"
        for x in range(self.idStart,self.idEnd+1):
            files = self.dataFolder + "/" + self.camera + "_" + self.day + self.month[:3] + self.year + "_" + str("%.4d" % x)
            print files
            # print x

    def askdirectory(self):
        self.dataFolder = tkFileDialog.askdirectory()
        print self.dataFolder
        self.foldLoc.delete(0, 'end')
        self.foldLoc.insert(0,str(self.dataFolder))

        self.filename = self.dataFolder + "/" + self.camera + "_" + self.day + self.month[:3] + \
                       self.year + "_" + str("%.4d" % self.idStart) + ".ibw"
        self.ODimage = IO.LoadImg(self.filename)

    def onExit(self):
        self.quit()



    def createEntries(self):
        frameFiles = tk.LabelFrame(self, text = "Files to be analyzed")

        tk.Label(frameFiles, text="Folder Location").grid(row=0, column=0)
        self.foldLoc = tk.Entry(frameFiles, width=40)
        self.foldLoc.insert(0,self.dataFolder)
        self.foldLoc.grid(row=0, column=1, columnspan=3)

        tk.Label(frameFiles, text="File Indices").grid(row=1, column=0)
        self.fileNoStart = tk.Entry(frameFiles, width = 8)
        self.fileNoStart.insert(0,self.idStart)
        self.fileNoStart.grid(row=1, column=1)

        tk.Label(frameFiles, text=" to ").grid(row=1, column=2)
        self.fileNoEnd = tk.Entry(frameFiles, width = 8)
        self.fileNoEnd.insert(0,self.idEnd)
        self.fileNoEnd.grid(row=1, column=3)

        tk.Label(frameFiles, text="Currently running").grid(row=2, column=0)
        self.filesRunning = tk.Entry(frameFiles, width=70)
        self.filesRunning.grid(row=2, column=1, columnspan=3)

        frameFiles.grid(row=1, column=0, columnspan=5)


        frameFitInfos = tk.LabelFrame(self, text = "Fit results:")
        tk.Label(frameFitInfos, text ="TF atom no").grid(row=0, column=0)
        tk.Label(frameFitInfos, text ="TF chem pot (in Hz)").grid(row=1, column=0)
        tk.Label(frameFitInfos, text ="TF Rx insitu (in um)").grid(row=2, column=0)
        tk.Label(frameFitInfos, text ="TF Ry insitu (in um)").grid(row=3, column=0)

        self.labelAtomNo = tk.Label(frameFitInfos, text ="1000")
        self.labelChemPot = tk.Label(frameFitInfos, text ="10")
        self.labelRxInsitu = tk.Label(frameFitInfos, text ="50")
        self.labelRyInsitu = tk.Label(frameFitInfos, text ="120")

        self.labelAtomNo.grid(row=0, column=1)
        self.labelChemPot.grid(row=1, column=1)
        self.labelRxInsitu.grid(row=2, column=1)
        self.labelRyInsitu.grid(row=3, column=1)
        frameFitInfos.grid(row=3, column=0, columnspan=5)



    def createButtons(self):
        self.loadImgButton = tk.Button(self, text='Load Images', command=self.loadImg)
        self.loadImgButton.grid(row=0, column=0)

        self.imshowButton = tk.Button(self, text='Show OD', command=self.imshowOD)
        self.imshowButton.grid(row=0, column=1)

        self.TFfitButton = tk.Button(self, text='Load with TF fit', command=self.imTFfit)
        self.TFfitButton.grid(row=0, column=2)

        self.cropAnalButton = tk.Button(self, text='Cropped Analysis', command=self.cropImgAnal)
        self.cropAnalButton.grid(row=0, column=3)

    def imshowOD(self):
        fig = plt.figure(1)
        plt.show()



    def createCanvases(self):
        self.fig1 = plt.figure(1, figsize=(3.5, 4))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1,4], height_ratios=[4,1])
        ax2 = plt.subplot(gs[1])
        ax1 = plt.subplot(gs[0], sharey=ax2)
        ax4 = plt.subplot(gs[3], sharex=ax2)

        self.canvasMain = FigureCanvasTkAgg(self.fig1, self.parent)
        self.canvasMain.show()
        self.canvasMain.get_tk_widget().grid(row=0, column=10, rowspan=5, columnspan=1)

        self.fig2 = plt.figure(2, figsize=(3.5, 2))
        gs = gridspec.GridSpec(1, 3)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharey=ax1)
        ax3 = plt.subplot(gs[2], sharex=ax1)

        self.canvasCrops = FigureCanvasTkAgg(self.fig2, self.parent)
        self.canvasCrops.show()
        self.canvasCrops.get_tk_widget().grid(row=5, column=10, rowspan=5, columnspan=1)

        self.fig3 = plt.figure(3, figsize=(3.5, 6))
        gs = gridspec.GridSpec(3, 2) #, width_ratios=[1,1], height_ratios=[1,1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax3 = plt.subplot(gs[2], sharex=ax1)
        ax4 = plt.subplot(gs[3], sharex=ax1)
        ax5 = plt.subplot(gs[4], sharex=ax1)
        ax6 = plt.subplot(gs[5], sharex=ax1)

        self.canvas2Dfit = FigureCanvasTkAgg(self.fig3, self.parent)
        self.canvas2Dfit.show()
        self.canvas2Dfit.get_tk_widget().grid(row=0, column=11, rowspan=10, columnspan=1)

        self.fig4 = plt.figure(4, figsize=(7, 2.5))
        gs = gridspec.GridSpec(2, 2) #, width_ratios=[1,1], height_ratios=[1,1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax3 = plt.subplot(gs[2], sharex=ax1)
        ax4 = plt.subplot(gs[3], sharex=ax1)

        self.canvasXYplots = FigureCanvasTkAgg(self.fig4, self.parent)
        self.canvasXYplots.show()
        self.canvasXYplots.get_tk_widget().grid(row=11, column=10, rowspan=3, columnspan=2)



#  TODO: Replace loop of filenames with FileNameTuple for parallel processing? (Ian's QgasFileIO module)
#=======================================================================================================
    def loadImg(self):
        self.idStart = int(self.fileNoStart.get())
        self.idEnd = int(self.fileNoEnd.get())
        for x in range(self.idStart,self.idEnd+1):
            filename = self.dataFolder + "/" + self.camera + "_" + self.day + self.month[:3] + \
                       self.year + "_" + str("%.4d" % x) + ".ibw"

            self.filesRunning.delete(0, 'end')
            self.filesRunning.insert(0,filename)

            self.ODimage = IO.LoadImg(filename)

            self.fig1.clear()
            plt.figure(1, figsize=(5, 5))
            gs = gridspec.GridSpec(2, 2, width_ratios=[1,4], height_ratios=[4,1])

            ax2 = plt.subplot(gs[1])
            self.imshowImg(self.ODimage,self.ODimage['OptDepth'].min(),self.ODimage['OptDepth'].max(),ax2)

            self.canvasMain.draw()



    def cropImgAnal(self):
        self.idStart = int(self.fileNoStart.get())
        self.idEnd = int(self.fileNoEnd.get())
        fraction = np.array([])
        idVal = np.array([])
        for x in range(self.idStart,self.idEnd+1):
            filename = self.dataFolder + "/" + self.camera + "_" + self.day + self.month[:3] + \
                       self.year + "_" + str("%.4d" % x) + ".ibw"
            self.filesRunning.delete(0, 'end')
            self.filesRunning.insert(0,filename)
            self.ODimage = IO.LoadImg(filename)

            # self.scaleFactor = self.ODimage["ExpInf"]["dx"]
            # print phys.NumberCalcAbsOD(self.ODimage['OptDepth'],1)

            self.fig1.clear()
            plt.figure(1, figsize=(5, 5))
            gs = gridspec.GridSpec(2, 2, width_ratios=[4,1], height_ratios=[4,1])

            ax2 = plt.subplot(gs[0])
            self.imshowImg(self.ODimage,self.ODimage['OptDepth'].min(),self.ODimage['OptDepth'].max(),ax2)

            img = Util.ImageSliceFromMax(self.ODimage, 5, pScale = True)

            pguess = lmfit.Parameters()
            pguess.add('offset', value = 1)
            pguess.add('A', value = 3)
            pguess.add('x0', value = img['xMax'])
            pguess.add('R', value = 50)

            ax4 = plt.subplot(gs[2], sharex=ax2)
            ax4.plot(img['xVals'], img['xSlice'])
            fit_par = Qfunc.FitQgas(img['xVals'], img['xSlice'], Qfunc.TF_only, pguess)
            fitdata2 = Qfunc.TF_only(img['xVals'], **fit_par.values)
            ax4.plot(img['xVals'],fitdata2)

            pguess.add('x0', value = img['yMax'])
            pguess.add('xc', value = 10)

            ax1 = plt.subplot(gs[1], sharey=ax2)
            ax1.plot(img['ySlice'], img['yVals'])
            fit_par = Qfunc.FitQgas(img['yVals'], img['ySlice'], Qfunc.TF_onlyChopped, pguess)
            fitdata = Qfunc.TF_onlyChopped(img['yVals'], **fit_par.values)
            ax1.plot(fitdata, img['yVals'])

            self.canvasMain.draw()

            Z = self.ODimage['OptDepth']

#===== analysis for 3 cropped images =====
            zmin = Z.min()
            zmax = Z.max()
            self.fig2.clear()
            plt.figure(2, figsize=(5, 2.5))
            gs = gridspec.GridSpec(1, 4)

            ax1 = plt.subplot(gs[0])
            Cropped1 = Util.ImageCrop(self.ODimage['xVals'], self.ODimage['yVals'], self.ODimage['OptDepth'], [-290, -20], array([250, 300]), Scaled = True, Center = False)
            self.imshowImg(Cropped1, zmin, zmax, ax1)

            img = Util.ImageSliceFromMax(Cropped1, 2, pScale = True)

            pguess = lmfit.Parameters()
            pguess.add('offset', value = 1)
            pguess.add('A', value = 3)
            pguess.add('x0', value = img['yMax'])
            pguess.add('R', value = 50)
            pguess.add('xc', value = -3)

            ax4 = plt.subplot(gs[3], sharey=ax1)

            fit_par = Qfunc.FitQgas(img['yVals'], img['ySlice'], Qfunc.TF_onlyChopped, pguess)
            fitdata2 = Qfunc.TF_onlyChopped(img['yVals'], **fit_par.values)
            ax4.plot(fitdata2, img['yVals'])

            ax2 = plt.subplot(gs[1], sharey=ax1)
            Cropped2 = Util.ImageCrop(self.ODimage['xVals'], self.ODimage['yVals'], self.ODimage['OptDepth'], [-10, 0], array([250, 300]), Scaled = True, Center = False)
            self.imshowImg(Cropped2, zmin, zmax, ax2)

            imgSlice = Util.ImageSliceFromMax(Cropped2, 3, pScale = True)
            pguess.add('x0', value = imgSlice['yMax'])

            fit_par = Qfunc.FitQgas(imgSlice['yVals'], imgSlice['ySlice'], Qfunc.TF_onlyChopped, pguess)
            fitdata2 = Qfunc.TF_onlyChopped(imgSlice['yVals'], **fit_par.values)
            ax4.plot(fitdata2, imgSlice['yVals'])

            ax3 = plt.subplot(gs[2], sharey=ax1)
            Cropped3 = Util.ImageCrop(self.ODimage['xVals'], self.ODimage['yVals'], self.ODimage['OptDepth'], [240, 20], array([250, 300]), Scaled = True, Center = False)
            self.imshowImg(Cropped3, zmin, zmax, ax3)

            imgSlice = Util.ImageSliceFromMax(Cropped3, 3, pScale = True)
            pguess.add('x0', value = imgSlice['yMax'])

            fit_par = Qfunc.FitQgas(imgSlice['yVals'], imgSlice['ySlice'], Qfunc.TF_onlyChopped, pguess)
            fitdata2 = Qfunc.TF_onlyChopped(imgSlice['yVals'], **fit_par.values)
            ax4.plot(fitdata2, imgSlice['yVals'])


            scalefac = 1
            mOne = phys.NumberCalcAbsOD(Cropped1['OptDepth'],scalefac)
            zero = phys.NumberCalcAbsOD(Cropped2['OptDepth'],scalefac)
            pOne = phys.NumberCalcAbsOD(Cropped3['OptDepth'],scalefac)


            self.canvasCrops.draw()

            fraction = np.append(fraction,[zero/(zero+pOne+mOne)],1)
            idVal = np.append(idVal, [float(self.ODimage['ExpInf']['IndexedValues'])], 1)
            self.fig3.clear()
            plt.figure(3, figsize=(5, 2.5))
            gs = gridspec.GridSpec(3, 1)

            ax1 = plt.subplot(gs[0])
            ax1.plot(idVal,fraction,'b.')

            # Cropped4 = join_segmentations(Cropped1['OptDepth'], Cropped3['OptDepth'])
            # im = ax1.imshow(Cropped4)
            # im.set_interpolation('bilinear')

            self.canvas2Dfit.draw()

            SCname = self.dataFolder + "/SC1_" + self.day + self.month[:3] + \
                       self.year + "_" + str("%.4d" % x) + ".ibw"
            SCtime, SCdiode = IO.LoadSC(SCname)

            self.fig4.clear()
            plt.figure(4, figsize=(7, 2.5))
            gs = gridspec.GridSpec(2, 2) #, width_ratios=[1,1], height_ratios=[1,1])
            ax1 = plt.subplot(gs[0])
            ax1.plot(SCtime[0], SCdiode[0], 'r.-')

            ax2 = plt.subplot(gs[1], sharey=ax1, sharex=ax1)
            ax2.plot(SCtime[1], SCdiode[1], 'bx-')

            self.canvasXYplots.draw()
        # idsort = np.argsort(idVal)
        # print fraction[idsort]
        # print idVal[idsort]



    def imTFfit(self):
        self.idStart = int(self.fileNoStart.get())
        self.idEnd = int(self.fileNoEnd.get())
        for x in range(self.idStart,self.idEnd+1):
            filename = self.dataFolder + "/" + self.camera + "_" + self.day + self.month[:3] + \
                       self.year + "_" + str("%.4d" % x) + ".ibw"
            self.ODimage = IO.LoadImg(filename)

            self.fig1.clear()
            plt.figure(1, figsize=(5, 5))
            gs = gridspec.GridSpec(2, 2, width_ratios=[4,1], height_ratios=[4,1])

            ax2 = plt.subplot(gs[0])
            self.imshowImg(self.ODimage,self.ODimage['OptDepth'].min(),self.ODimage['OptDepth'].max(),ax2)

            imgSlice = Util.ImageSliceFromMax(self.ODimage, 5, pScale = True)

            pguess = lmfit.Parameters()
            pguess.add('offset', value = 0)
            pguess.add('A', value = 3)
            pguess.add('x0', value = imgSlice['yMax'])
            pguess.add('R', value = 50)

            ax1 = plt.subplot(gs[1], sharey=ax2)
            ax1.plot(imgSlice['ySlice'], imgSlice['yVals'])
            fit_par = Qfunc.FitQgas(imgSlice['yVals'], imgSlice['ySlice'], Qfunc.TF_only, pguess)
            fitdata = Qfunc.TF_only(imgSlice['yVals'], **fit_par.values)
            ax1.plot(fitdata, imgSlice['yVals'])

            TFRadiusShort =  fit_par.params['R'].value

            pguess.add('x0', value = imgSlice['xMax'])

            ax4 = plt.subplot(gs[2], sharex=ax2)
            ax4.plot(imgSlice['xVals'], imgSlice['xSlice'])
            fit_par = Qfunc.FitQgas(imgSlice['xVals'], imgSlice['xSlice'], Qfunc.TF_only, pguess)
            fitdata = Qfunc.TF_only(imgSlice['xVals'], **fit_par.values)
            ax4.plot(imgSlice['xVals'],fitdata)

            TFRadiusLong = fit_par.params['R'].value

            self.canvasMain.draw()

            self.filesRunning.delete(0, 'end')
            self.filesRunning.insert(0, filename)

            # ====================================
            # print self.ODimage
            TOFTime = self.ODimage['ExpInf']['expand_time']/1000.0;
            # print self.ODimage['Note']
            # print self.ODimage['ExpInf']
            omegashort = 2*np.pi*100;
            omegalong = 2*np.pi*150;
            a,b,c,d = phys.NumberCalcTOF_TFfit(TFRadiusLong,TFRadiusShort,omegashort,omegalong,TOFTime)
            self.labelAtomNo.config(text=str(round(a,0)))
            self.labelChemPot.config(text=str(round(b,2)))
            self.labelRxInsitu.config(text=str(round(c,2)))
            self.labelRyInsitu.config(text=str(round(d,2)))
            # ====================================


    def imshowImg(self,Image,zmin,zmax,axis):
        Z = np.flipud(Image['OptDepth'])
        xmin = Image['xVals'].min()
        ymin = Image['yVals'].min()
        xmax = Image['xVals'].max()
        ymax = Image['yVals'].max()
        im = axis.imshow(Z, cmap=cm.jet, vmin=zmin, vmax=zmax, extent=[xmin, xmax, ymin, ymax], aspect ='auto')
        im.set_interpolation('bilinear')



# Functions to be executed by the widgets


root = tk.Tk()
root.grid()
app = Application(root)
root.mainloop()