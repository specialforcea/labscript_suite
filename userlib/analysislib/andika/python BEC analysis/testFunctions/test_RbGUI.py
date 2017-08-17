# Project   : pycharm
# Filename  : qGasGUI
# Created on: 2014 Dec 11 02:19
# Author    : aputra

import Tkinter as tk
import tkFileDialog
import QgasUtils

class Application(tk.Frame):

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.grid()
        self.parent.title("Rb Analysis")
        self.dir_opt = options = {}
        self.createMenus()
        self.createButtons()
        self.createEntries()
        self.createCanvas()

    def createMenus(self):
        menubar = tk.Menu(self.parent)
        self.parent.config(menu=menubar)

        fileMenu = tk.Menu(menubar)
        menubar.add_cascade(label="File", menu=fileMenu)
        fileMenu.add_command(label="New", command=self.onExit)
        fileMenu.add_command(label="Set Folder...", command=self.askdirectory)
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command=self.onExit)

        editMenu = tk.Menu(menubar)
        menubar.add_cascade(label="Edit", menu=editMenu)
        editMenu.add_command(label="Copy", command=self.onExit)
        editMenu.add_command(label="Paste", command=self.onExit)


    def createButtons(self):
        self.runButton = tk.Button(self, text='Run', command=self.quit)
        self.runButton.grid(row=1, column=1, rowspan=1, columnspan=1)

        self.quitButton = tk.Button(self, text='Quit', command=self.quit)
        self.quitButton.grid(row=2, column=1, rowspan=1, columnspan=1)


    def createEntries(self):
        tk.Label(self, text="Folder Location").grid(row=3,column=1)
        self.foldLoc = tk.Entry(self)
        self.foldLoc.grid(row=3,column=2)

        tk.Label(self, text="File Index").grid(row=4,column=1)
        self.fileIndex = tk.Entry(self)
        self.fileIndex.grid(row=4,column=2)

        tk.Label(self, text="File Name:").grid(row=5,column=1)
        self.fileName = tk.Entry(self)
        self.fileName.grid(row=5,column=2)


    def createCanvas(self):

        self.qgasDisp = tk.Canvas(self, width=80, height=40)
        self.qgasDisp.grid(row=1, column=3, rowspan=5, columnspan=3)

        self.qgasDisp.create_line(0,15,80,35,fill="red")

        self.aiDisp = tk.Canvas(self, width=80, height=40)
        self.aiDisp.grid(row=6, column=3, rowspan=5, columnspan=3)

        self.aiDisp.create_line(0,15,80,35,fill="green")

        self.scopeDisp = tk.Canvas(self, width=80, height=40)
        self.scopeDisp.grid(row=6, column=6, rowspan=5, columnspan=3)

        self.scopeDisp.create_line(0,15,80,35,fill="blue")

    def askdirectory(self):
        dirFolder = tkFileDialog.askdirectory()
        self.foldLoc.insert(1,str(dirFolder))
        return tkFileDialog.askdirectory(**self.dir_opt)


    def onExit(self):
        self.quit()


root = tk.Tk()
root.grid()
app = Application(root)
root.mainloop()