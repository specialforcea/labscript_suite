# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'chipfpga.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(663, 635)
        Form.setMaximumSize(QtCore.QSize(15777191, 16777215))
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.scrollArea = QtGui.QScrollArea(Form)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName(_fromUtf8("scrollArea"))
        self.scrollAreaWidgetContents_2 = QtGui.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 643, 310))
        self.scrollAreaWidgetContents_2.setObjectName(_fromUtf8("scrollAreaWidgetContents_2"))
        self.gridLayout_6 = QtGui.QGridLayout(self.scrollAreaWidgetContents_2)
        self.gridLayout_6.setObjectName(_fromUtf8("gridLayout_6"))
        self.tab = QtGui.QTabWidget(self.scrollAreaWidgetContents_2)
        self.tab.setObjectName(_fromUtf8("tab"))
        self.load_tab = QtGui.QWidget()
        self.load_tab.setObjectName(_fromUtf8("load_tab"))
        self.gridLayout_3 = QtGui.QGridLayout(self.load_tab)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.load_table = QtGui.QTableWidget(self.load_tab)
        self.load_table.setObjectName(_fromUtf8("load_table"))
        self.load_table.setColumnCount(0)
        self.load_table.setRowCount(0)
        self.gridLayout_3.addWidget(self.load_table, 0, 0, 1, 1)
        self.tab.addTab(self.load_tab, _fromUtf8(""))
        self.read_table_tab = QtGui.QWidget()
        self.read_table_tab.setObjectName(_fromUtf8("read_table_tab"))
        self.gridLayout_2 = QtGui.QGridLayout(self.read_table_tab)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.read_table = QtGui.QTableWidget(self.read_table_tab)
        self.read_table.setObjectName(_fromUtf8("read_table"))
        self.read_table.setColumnCount(0)
        self.read_table.setRowCount(0)
        self.gridLayout_2.addWidget(self.read_table, 0, 0, 1, 1)
        self.tab.addTab(self.read_table_tab, _fromUtf8(""))
        self.read_graph_tab = QtGui.QWidget()
        self.read_graph_tab.setObjectName(_fromUtf8("read_graph_tab"))
        self.verticalLayout = QtGui.QVBoxLayout(self.read_graph_tab)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.read_graph = QtGui.QGraphicsView(self.read_graph_tab)
        self.read_graph.setObjectName(_fromUtf8("read_graph"))
        self.verticalLayout.addWidget(self.read_graph)
        self.tab.addTab(self.read_graph_tab, _fromUtf8(""))
        self.gridLayout_6.addWidget(self.tab, 0, 0, 1, 1)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents_2)
        self.gridLayout.addWidget(self.scrollArea, 0, 1, 1, 1)
        self.scrollArea_2 = QtGui.QScrollArea(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea_2.sizePolicy().hasHeightForWidth())
        self.scrollArea_2.setSizePolicy(sizePolicy)
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollArea_2.setObjectName(_fromUtf8("scrollArea_2"))
        self.scrollAreaWidgetContents = QtGui.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 643, 297))
        self.scrollAreaWidgetContents.setObjectName(_fromUtf8("scrollAreaWidgetContents"))
        self.gridLayout_7 = QtGui.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_7.setObjectName(_fromUtf8("gridLayout_7"))
        self.file_dir_Edit = QtGui.QLineEdit(self.scrollAreaWidgetContents)
        self.file_dir_Edit.setObjectName(_fromUtf8("file_dir_Edit"))
        self.gridLayout_7.addWidget(self.file_dir_Edit, 1, 1, 1, 1)
        self.read_status_edit = QtGui.QLineEdit(self.scrollAreaWidgetContents)
        self.read_status_edit.setObjectName(_fromUtf8("read_status_edit"))
        self.gridLayout_7.addWidget(self.read_status_edit, 3, 1, 1, 1)
        self.byte_to_read_edit = QtGui.QLineEdit(self.scrollAreaWidgetContents)
        self.byte_to_read_edit.setObjectName(_fromUtf8("byte_to_read_edit"))
        self.gridLayout_7.addWidget(self.byte_to_read_edit, 5, 1, 1, 1)
        self.Load_Button = QtGui.QPushButton(self.scrollAreaWidgetContents)
        self.Load_Button.setObjectName(_fromUtf8("Load_Button"))
        self.gridLayout_7.addWidget(self.Load_Button, 1, 0, 1, 1)
        self.Write_Button = QtGui.QPushButton(self.scrollAreaWidgetContents)
        self.Write_Button.setObjectName(_fromUtf8("Write_Button"))
        self.gridLayout_7.addWidget(self.Write_Button, 3, 0, 1, 1)
        self.write_status_edit = QtGui.QLineEdit(self.scrollAreaWidgetContents)
        self.write_status_edit.setObjectName(_fromUtf8("write_status_edit"))
        self.gridLayout_7.addWidget(self.write_status_edit, 5, 2, 1, 1)
        self.Read_button = QtGui.QPushButton(self.scrollAreaWidgetContents)
        self.Read_button.setObjectName(_fromUtf8("Read_button"))
        self.gridLayout_7.addWidget(self.Read_button, 5, 0, 1, 1)
        self.correct_button = QtGui.QPushButton(self.scrollAreaWidgetContents)
        self.correct_button.setObjectName(_fromUtf8("correct_button"))
        self.gridLayout_7.addWidget(self.correct_button, 7, 0, 1, 1)
        self.correct_byte_edit = QtGui.QLineEdit(self.scrollAreaWidgetContents)
        self.correct_byte_edit.setObjectName(_fromUtf8("correct_byte_edit"))
        self.gridLayout_7.addWidget(self.correct_byte_edit, 7, 1, 1, 1)
        self.write_status = QtGui.QLabel(self.scrollAreaWidgetContents)
        self.write_status.setObjectName(_fromUtf8("write_status"))
        self.gridLayout_7.addWidget(self.write_status, 2, 1, 1, 1, QtCore.Qt.AlignBottom)
        self.read_status = QtGui.QLabel(self.scrollAreaWidgetContents)
        self.read_status.setObjectName(_fromUtf8("read_status"))
        self.gridLayout_7.addWidget(self.read_status, 4, 2, 1, 1, QtCore.Qt.AlignBottom)
        self.table_dir = QtGui.QLabel(self.scrollAreaWidgetContents)
        self.table_dir.setMaximumSize(QtCore.QSize(308, 48))
        self.table_dir.setObjectName(_fromUtf8("table_dir"))
        self.gridLayout_7.addWidget(self.table_dir, 0, 1, 1, 1, QtCore.Qt.AlignBottom)
        self.correct_value_edit = QtGui.QLineEdit(self.scrollAreaWidgetContents)
        self.correct_value_edit.setObjectName(_fromUtf8("correct_value_edit"))
        self.gridLayout_7.addWidget(self.correct_value_edit, 7, 2, 1, 1)
        self.bytes_to_read = QtGui.QLabel(self.scrollAreaWidgetContents)
        self.bytes_to_read.setObjectName(_fromUtf8("bytes_to_read"))
        self.gridLayout_7.addWidget(self.bytes_to_read, 4, 1, 1, 1, QtCore.Qt.AlignBottom)
        self.correct_byte = QtGui.QLabel(self.scrollAreaWidgetContents)
        self.correct_byte.setObjectName(_fromUtf8("correct_byte"))
        self.gridLayout_7.addWidget(self.correct_byte, 6, 1, 1, 1, QtCore.Qt.AlignBottom)
        self.corret_value = QtGui.QLabel(self.scrollAreaWidgetContents)
        self.corret_value.setObjectName(_fromUtf8("corret_value"))
        self.gridLayout_7.addWidget(self.corret_value, 6, 2, 1, 1, QtCore.Qt.AlignBottom)
        self.Load_Button.raise_()
        self.Load_Button.raise_()
        self.Write_Button.raise_()
        self.Read_button.raise_()
        self.byte_to_read_edit.raise_()
        self.read_status_edit.raise_()
        self.file_dir_Edit.raise_()
        self.correct_button.raise_()
        self.correct_byte_edit.raise_()
        self.correct_value_edit.raise_()
        self.write_status_edit.raise_()
        self.table_dir.raise_()
        self.write_status.raise_()
        self.read_status.raise_()
        self.bytes_to_read.raise_()
        self.correct_byte.raise_()
        self.corret_value.raise_()
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout.addWidget(self.scrollArea_2, 1, 1, 1, 1)

        self.retranslateUi(Form)
        self.tab.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.tab.setTabText(self.tab.indexOf(self.load_tab), _translate("Form", "Loaded table", None))
        self.tab.setTabText(self.tab.indexOf(self.read_table_tab), _translate("Form", "Read table", None))
        self.tab.setTabText(self.tab.indexOf(self.read_graph_tab), _translate("Form", "Read graph", None))
        self.Load_Button.setText(_translate("Form", "Load", None))
        self.Write_Button.setText(_translate("Form", "Write", None))
        self.Read_button.setText(_translate("Form", "Read", None))
        self.correct_button.setText(_translate("Form", "Correct", None))
        self.write_status.setText(_translate("Form", "Write status", None))
        self.read_status.setText(_translate("Form", "Read status", None))
        self.table_dir.setText(_translate("Form", "Table file directory", None))
        self.bytes_to_read.setText(_translate("Form", "bytes to read", None))
        self.correct_byte.setText(_translate("Form", "correct byte", None))
        self.corret_value.setText(_translate("Form", "correct_value", None))

