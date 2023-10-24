# -*- coding: utf-8 -*-
"""
        Universidade Federal de Minas Gerais - UFMG
            Instituto de Ciências Exatas - ICEx
          Departamento de Ciência da Computação - DCC
        Laboratório de Bioinformática e Sistemas - LBS

@Author: Frederico Chaves Carvalho

@Advisor: Raquel Cardoso de Melo-Minardi

@Contributors: Leonardo Henrique Franca de Lima
               Eduardo Azevedo Correa
               Ana Paula de Abreu
"""
# Import the necessary modules and dependecies
import sys
import os
import pandas as pd
import numpy as np
import pymol
import joblib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as stats
import scipy
#import seaborn as sns
from matplotlib import colors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from matplotlib.backends.qt_compat import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

# Read the config file and select the appropriate window to start
#Ui_MainWindow, QtBaseClass = uic.loadUiType("ui/Dialog.ui")

# Main window class
#class Interface(QtWidgets.QMainWindow, Ui_MainWindow):
class Interface(QtWidgets.QMainWindow):
    def __init__(self):
        super(Interface, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('Main.ui', self) # Load the .ui file
        self.show() # Show the GUI
        
        # Actions
        self.actionReset.triggered.connect(self.resetAll)
        
        # Connect buttons to appropriate functions
        # 1: MGLTools tab
        self.mglLigPathButton.clicked.connect(self.seleLigPathMGL)
        self.mglRecPathButton.clicked.connect(self.seleRecPathMGL)
        self.pytonshPathButton.clicked.connect(self.selePythonshPathMGL)
        self.prepareLigButton.clicked.connect(self.prepLigMGL)
        self.prepareRecButton.clicked.connect(self.prepRecMGL)
        self.pdbqtDestButton.clicked.connect(self.pdbqtDestinationMGL)
        self.convertButton.clicked.connect(self.convertMGL)
        
        # 2: Docking - VINA tab
        self.vinaLigPathButton.clicked.connect(self.seleLigPathVINA)
        self.vinaRecPathButton.clicked.connect(self.seleRecPathVINA)
        self.vinaConfigFileButton.clicked.connect(self.loadConfigVINA)
        self.vinaResultsPathButton.clicked.connect(self.resultsDestVINA)
        self.vinaLogsPathButton.clicked.connect(self.logsDestVINA)
        self.vinaDockButton.clicked.connect(self.dockVINA)
        self.vinaSaveFileButton.clicked.connect(self.csvNameVINA)
        self.vinaSaveButton.clicked.connect(self.saveLogsVINA)
        
        # 3: Rescoring tab
        self.rescoringVinaLogsButton.clicked.connect(self.seleVinaLogsRESCORING)
        self.rescoringReceptorButton.clicked.connect(self.seleReceptorFolderRESCORING)
        self.rescoringOutputButton.clicked.connect(self.seleOutputFolderRESCORING)
        self.convexFileButton.clicked.connect(self.seleConvexFileRESCORING)
        self.convexSummaryButton.clicked.connect(self.csvNameCONVEX)
        self.startConvex.clicked.connect(self.rescoreCONVEX)
        self.sminaFileButton.clicked.connect(self.seleSminaFileRESCORING)
        self.sminaSummaryButton.clicked.connect(self.csvNameSMINA)
        self.startSmina.clicked.connect(self.rescoreSMINA)
        
        # 4: Cluster analysis tab
        self.addSummaryButton.clicked.connect(self.addFileCLUSTER)
        self.makeScoresFileButton.clicked.connect(self.makeFileCLUSTER)
        self.loadScoresFileButton.clicked.connect(self.loadFileCLUSTER)
        #self.viewScoreFileButton.clicked.connect(self.cluster)
        self.makeClustersButton.clicked.connect(self.makeCLUSTER)
        self.showCurrentButton.clicked.connect(self.showCLUSTER)
        self.showDistributionsButton.clicked.connect(self.showDistCLUSTER)
        self.loadClusterButton.clicked.connect(self.loadCLUSTER)
        self.updateClusterButton.clicked.connect(self.updateCLUSTER)
        self.clusterColorButton.clicked.connect(self.recolorCLUSTER)
        self.reclusterButton.clicked.connect(self.reCLUSTER)
        # radial buttons
        self.clusterAllRadio.toggled.connect(self.choseTypeCLUSTER)
        self.clusterControlRadio.toggled.connect(self.choseTypeCLUSTER)
        self.cmapRadio.toggled.connect(self.choseColorCLUSTER)
        self.colorsRadio.toggled.connect(self.choseColorCLUSTER)
        self.viewMatrixradio.toggled.connect(self.showMatrixCLUSTER)
        self.view3Dradio.toggled.connect(self.showCLUSTER)
        self.colorByCluster.toggled.connect(self.changeTypeCLUSTER)
        # check box
        self.centersCheck.stateChanged.connect(self.showCentersCLUSTER)
        # combo boxes
        #self.xAxisBox.currentTextChanged.connect(self.updateCLUSTER)
        #self.yAxisBox.currentTextChanged.connect(self.updateCLUSTER)
        #self.zAxisBox.currentTextChanged.connect(self.updateCLUSTER)
        self.clusterNumberBox.currentTextChanged.connect(self.highlightCLUSTER)
        
        self.summaries = {}
        self.errorDict = {0: 'The ligands folder is not defined!',
                          1: 'The receptors folder is not defined!',
                          2: "MGLTools' pythonsh file is not defined!",
                          3: "MGLTools' prepare_ligand4.py file is not defined!",
                          4: "MGLTools' prepare_receptor4.py file is not defined!",
                          5: 'No folder was defined to store the PDBQT files!',
                          6: 'The config file is not defined!',
                          7: 'No folder was defined to store results!',
                          8: 'No folder was defined to store the logs!',
                          9: 'No results prefix defined!',
                          10:'No logs prefix defined!',
                          11:'No name is defined for the summary file!',
                          12:'Vina logs folder not defined!',
                          13:'CONVEX-PL file not defined!',
                          14:'smina.static file not defined!',
                          15: 'No rescoring output folder is defined!'}
        self.clusterDF = pd.DataFrame()
        self.kmeans = None
        self.axes = None
        self.clusterColors = {}
        self.rgba = []
        self.rgba_t = []
        self.alphas = []
        self.alphas_t = []
        #self.test()
        self.compoundTable.setColumnWidth(0, 150)
        self.compoundTable.setColumnWidth(1, 40)
        self.compoundTable.setColumnWidth(2, 75)
        self.loadedFile = False
        
        
    # Test function
    def test(self):
        print(self.clusterAllRadio.isChecked())
    
    # General purpose functions
    
    def resetAll(self):
        # Clear all the fields in the MGL tools tab
        self.mglLigPath.clear()
        self.mglRecPath.clear()
        self.prepareLigPath.clear()
        self.prepareRecPath.clear()
        self.pythonshPath.clear()
        self.pdbqtDest.clear()
        self.mglRecFolderName.setText('Receptors/')
        self.mglLigFolderName.setText('Ligands/')
        self.convertProgress.setValue(0)
        
        # Clear all the fields in the Docking-Vina tab
        self.vinaLigPath.clear()
        self.vinaRecPath.clear()
        self.vinaConfigFile.clear()
        self.vinaResultsPath.clear()
        self.vinaLogsPath.clear()
        self.vinaResultsPrefix.clear()
        self.vinaLogsPrefix.clear()
        self.vinaSaveName.clear()
        self.vinaN.clear()
        self.vinaN.addItem('1')
        self.vinaN.setCurrentIndex(0)
        self.vinaDockProgress.setValue(0)
        
        # Clear all the fields in the Rescoring tab
        self.rescoringVinaLogs.clear()
        self.rescoringReceptor.clear()
        self.rescoringOutput.clear()
        self.convexFile.clear()
        self.convexSummary.clear()
        self.convexN.clear()
        self.convexN.addItem('1')
        self.convexProgress.setValue(0)
        self.sminaFile.clear()
        self.sminaSummary.clear()
        self.sminaFunction.setCurrentIndex(0)
        self.sminaN.clear()
        self.sminaN.addItem('1')
        self.sminaProgress.setValue(0)
        
        # Cluster Analysis Tab
        # Clear all the fields in the Data analysis subtab
        self.summaryList.clear()
        try:
            self.pcaFrame.layout().takeAt(0).widget().deleteLater() 
        except:
            pass
        self.posContList.clear()
        self.negContList.clear()
        self.clusterN.setValue(1)
        self.functionList.clear()
        self.makeScoresFileButton.setEnabled(False)
        
        # Clear all the fields in the Cluster view subtab
        self.showCurrentButton.setEnabled(False)
        self.updateClusterButton.setEnabled(False)
        self.showDistributionsButton.setEnabled(False)
        self.seleClusterViewGroup.setEnabled(False)
        self.xAxisBox.clear()
        self.yAxisBox.clear()
        self.zAxisBox.clear()
        self.xAxisBox.setEnabled(False)
        self.yAxisBox.setEnabled(False)
        self.zAxisBox.setEnabled(False)
        self.clusterNumberBox.clear()
        self.clusterNumberBox.addItem('All')
        self.clusterNumberBox.setEnabled(False)
        self.colorByGroup.setEnabled(False)
        self.colorGroup.setEnabled(False)
        self.compoundTable.clear()
        self.alphaSpinBox.setValue(1.0)
        self.clusterColorButton.setStyleSheet("background-color: #bebebe")
        try:
            self.clusterViewFrame.layout().takeAt(0).widget().deleteLater()
        except:
            pass
        
        # Clear all the fields in the Recluster subtab
        self.selectClusters.clear()
        self.reclusterFunctionList.clear()
        self.newClusterN.setValue(1)
        self.reclusterButton.setEnabled(False)
        
        # Reinitialize global variables
        self.summaries = {}
        self.clusterDF = pd.DataFrame()
        self.kmeans = None
        self.axes = None
        self.clusterColors = {}
        self.rgba = []
        self.alphas = []
        self.compoundTable.setColumnWidth(0, 150)
        self.compoundTable.setColumnWidth(1, 40)
        self.compoundTable.setColumnWidth(2, 75)
        self.compoundTable.setRowCount(0) 
        self.loadedFile = False
        
    def checkPath(self, pathstr):
        if pathstr[-1] != '/':
            pathstr = pathstr + '/'
        return pathstr
    
    def checkConversion(self, source):
        missing = []
        answer = False
        if source == 'MGL':
            if self.mglLigPath.text().replace(' ','') == '':
                missing.append(0)
                answer = True
            if self.mglRecPath.text().replace(' ','') == '':
                missing.append(1)
                answer = True
            if self.pythonshPath.text().replace(' ','') == '':
                missing.append(2)
                answer = True
            if self.prepareLigPath.text().replace(' ','') == '':
                missing.append(3)
                answer = True
            if self.prepareRecPath.text().replace(' ','') == '':
                missing.append(4)
                answer = True
            if self.pdbqtDest.text().replace(' ','') == '':
                missing.append(5)
                answer = True
                
        elif source == 'VINA':
            if self.vinaLigPath.text().replace(' ','') == '':
                missing.append(0)
                answer = True
            if self.vinaRecPath.text().replace(' ','') == '':
                missing.append(1)
                answer = True
            if self.vinaConfigFile.text().replace(' ','') == '':
                missing.append(6)
                answer = True
            if self.vinaResultsPath.text().replace(' ','') == '':
                missing.append(7)
                answer = True
            if self.vinaLogsPath.text().replace(' ','') == '':
                missing.append(8)
                answer = True
            if self.vinaResultsPrefix.text().replace(' ','') == '':
                missing.append(9)
                answer = True
            if self.vinaLogsPrefix.text().replace(' ','') == '':
                missing.append(10)
                answer = True
        elif source == 'VINASAVE':  
            if self.vinaSaveName.text().replace(' ','') == '':
                missing.append(11)
                answer = True
            if self.vinaConfigFile.text().replace(' ','') == '':
                missing.append(6)
            if self.vinaLogsPath.text().replace(' ','') == '':
                missing.append(8)
                answer = True
        elif source == 'CONVEX':  
            if self.rescoringVinaLogs.text().replace(' ','') == '':
                missing.append(12)
                answer = True
            if self.rescoringReceptor.text().replace(' ','') == '':
                missing.append(1)
                answer = True
            if self.rescoringOutput.text().replace(' ','') == '':
                missing.append(15)
                answer = True
            if self.convexFile.text().replace(' ','') == '':
                missing.append(13)
                answer = True
            if self.convexSummary.text().replace(' ','') == '':
                missing.append(11)
                answer = True
        elif source == 'SMINA':  
            if self.rescoringVinaLogs.text().replace(' ','') == '':
                missing.append(12)
                answer = True
            if self.rescoringReceptor.text().replace(' ','') == '':
                missing.append(1)
                answer = True
            if self.rescoringOutput.text().replace(' ','') == '':
                missing.append(15)
                answer = True
            if self.sminaFile.text().replace(' ','') == '':
                missing.append(14)
                answer = True
            if self.sminaSummary.text().replace(' ','') == '':
                missing.append(11)
                answer = True
                
        return (answer, missing)
        
    def updateBar(self, bar):
        pass
    
    def getDir(self,prompt,target, diagType):
        if diagType == 'd':
            chosen_dir = str(QtWidgets.QFileDialog.getExistingDirectory(self, prompt))
            chosen_dir += '/'
            target.setText(chosen_dir)
        elif diagType == 'f':
            chosen_file = str(QtWidgets.QFileDialog.getOpenFileName(self, prompt)[0])
            target.setText(chosen_file)
        else:
            chosen_file = str(QtWidgets.QFileDialog.getSaveFileName(self, prompt)[0])
            if chosen_file[-4:] == '.csv':
                target.setText(chosen_file)
            else:
                target.setText(chosen_file + '.csv')
            
    def readFile(self, filename):
        for line in open(filename):
            yield line  
    
    # MGL Tools tab functions
    def seleLigPathMGL(self):
        self.getDir('Select ligands folder', self.mglLigPath, 'd')
    
    def seleRecPathMGL(self):
        self.getDir('Select receptors folder', self.mglRecPath, 'd')
    
    def selePythonshPathMGL(self):
        self.getDir("Select MGLTools' pythonsh", self.pythonshPath, 'f')
    
    def prepLigMGL(self):
        self.getDir("Select MGLTools' prepare_ligand4.py", self.prepareLigPath, 'f')
    
    def prepRecMGL(self):
        self.getDir("Select MGLTools' prepare_receptor4.py", self.prepareRecPath, 'f')
    
    def pdbqtDestinationMGL(self):
        self.getDir('Select the folder to store PDBQT files', self.pdbqtDest,'d')

    def convertMGL(self):
        preCheck = self.checkConversion('MGL')
        if preCheck[0] == True:
            message = 'The following errors were found:\n\n'
            for m in preCheck[1]:
                message = message + '\t•' + self.errorDict[m] + '\n'
            errorWidget = QtWidgets.QErrorMessage(self)
            errorWidget.setFixedSize(500,200)
            errorWidget.showMessage(message)
        else:
            lig_dir = self.mglLigPath.text()
            rec_dir = self.mglRecPath.text()
            pythonsh_dir = self.pythonshPath.text()
            preparelig_dir = self.prepareLigPath.text()
            preparerec_dir = self.prepareRecPath.text()
            pdbqt_dir = self.pdbqtDest.text()
            pdbqtL_dir = self.checkPath(pdbqt_dir + self.mglLigFolderName.text())
            pdbqtR_dir = self.checkPath(pdbqt_dir + self.mglRecFolderName.text())
                 
            # Step 1 - Create folders if they don't exist
            if not os.path.isdir(pdbqt_dir):
                os.mkdir(pdbqt_dir)
            if not os.path.isdir(pdbqtR_dir):
                os.mkdir(pdbqtR_dir)    
            if not os.path.isdir(pdbqtL_dir):
                os.mkdir(pdbqtL_dir)   
            
            # Conversion of input files to pdbqt format
            total = len(os.listdir(rec_dir)) + len(os.listdir(lig_dir))
            i = 0
            self.convertProgress.setValue(int(i/total * 100))
            for r in os.listdir(rec_dir):
                rec_name = rec_dir + r
                rec_pdbqt = pdbqtR_dir + r[:-4] + '.pdbqt'
                os.system(pythonsh_dir + ' ' + preparerec_dir + " -r " + rec_name + " -o " + rec_pdbqt)
                i += 1
                self.convertProgress.setValue(int(i/total * 100))
            for l in os.listdir(lig_dir):
                i += 1
                lig_name = lig_dir + l
                lig_pdbqt = pdbqtL_dir + l[:-5] + '.pdbqt'
                os.system(pythonsh_dir + ' ' + preparelig_dir + ' -l '+ lig_name + ' -o ' + lig_pdbqt)
                self.convertProgress.setValue(int(i/total * 100))
            
    # Docking - VINA tab functions
    def seleLigPathVINA(self):
        self.getDir('Select ligands folder', self.vinaLigPath, 'd')
    
    def seleRecPathVINA(self):
        self.getDir('Select receptors folder', self.vinaRecPath, 'd')
        
    def loadConfigVINA(self):
        self.getDir('Select the config file', self.vinaConfigFile, 'f')
        self.vinaN.clear()
        if self.vinaConfigFile.text().replace(' ','') != '':
            for line in self.readFile(self.vinaConfigFile.text()):
                if line.split(' ')[0] == 'num_modes':
                    N = int(line.split(' ')[2])
                    for i in range(N):
                        self.vinaN.addItem(str(i+1))
                    self.vinaN.setCurrentIndex(0)
                    break
    
    def resultsDestVINA(self):
        self.getDir('Select the folder to store docking results', self.vinaResultsPath, 'd')
    
    def logsDestVINA(self):
        self.getDir('Select the folder to store docking logs', self.vinaLogsPath, 'd')
    
    def csvNameVINA(self):
        self.getDir('Save log file', self.vinaSaveName, 's')
    
    def dockVINA(self):
        preCheck = self.checkConversion('VINA')
        if preCheck[0] == True:
            message = 'The following errors were found:\n\n'
            for m in preCheck[1]:
                message = message + '\t•' + self.errorDict[m] + '\n'
            errorWidget = QtWidgets.QErrorMessage(self)
            errorWidget.setFixedSize(500,200)
            errorWidget.showMessage(message)
        else:
            lig_dir = self.vinaLigPath.text()
            rec_dir = self.vinaRecPath.text()
            config_file = self.vinaConfigFile.text()
            results_dir = self.vinaResultsPath.text()
            logs_dir = self.vinaLogsPath.text()
            resultsprefix = self.vinaResultsPrefix.text()
            if resultsprefix[-1] != '_':
                resultsprefix += '_'
            logsprefix = self.vinaLogsPrefix.text()
            if logsprefix[-1] != '_':
                logsprefix += '_'
            
            if not os.path.isdir(results_dir):
                os.mkdir(results_dir)
            if not os.path.isdir(logs_dir):
                os.mkdir(logs_dir)
            
            i = 0
            total = len(os.listdir(rec_dir)) * len(os.listdir(lig_dir))
            self.vinaDockProgress.setValue(int(i/total * 100))
            for r in os.listdir(rec_dir):
                rec_pdbqt = rec_dir + r
                for l in os.listdir(lig_dir):
                    lig_pdbqt = lig_dir + l
                    command = ''.join(['vina --receptor ', rec_pdbqt, ' --ligand ', lig_pdbqt, ' --out ', results_dir, resultsprefix, l[:-6], "_", r[:-6], '.pdbqt --log ', logs_dir, logsprefix,l[:-6], "_", r[:-6], '.log --config ', config_file])
                    os.system(command) 
                    i += 1
                    self.vinaDockProgress.setValue(int(i/total * 100))
    
    def saveLogsVINA(self):
        preCheck = self.checkConversion('VINASAVE')
        if preCheck[0] == True:
            message = 'The following errors were found:\n\n'
            for m in preCheck[1]:
                message = message + '\t•' + self.errorDict[m] + '\n'
            errorWidget = QtWidgets.QErrorMessage(self)
            errorWidget.setFixedSize(500,200)
            errorWidget.showMessage(message)
        else:
            logs_dir = self.vinaLogsPath.text()
            if '.csv' in self.vinaSaveName.text():
                csvName = self.vinaSaveName.text()
            else:
                csvName = self.vinaSaveName.text() + '.csv'
            data = {"Ligand":[], "Receptor":[]}
            N = int(self.vinaN.currentText())
            for i in range(N):
                data['Affinity' + str(i+1)] = []
    
            for filename in os.listdir(logs_dir):
                full_filename = logs_dir + filename
                try:
                    ligname = filename.split('_')[1]
                    recname = filename.split('_')[2][:-4]
                except:
                    print(filename)
                    print('\n\n\n\n')
                data["Ligand"].append(ligname)
                data["Receptor"].append(recname)
                j = 0
                for line in self.readFile(full_filename):
                    if len(line)> 3:
                        l = [a for a in line.split(' ') if len(a) > 0]
                        if l[0].isdigit():
                            digit = int(l[0])
                            if digit <= N:
                                j = 1
                                aff = 'Affinity' + str(digit)
                                data[aff].append(float(l[1]))
                if j == 0:
                    print(filename)
    
            df_vina = pd.DataFrame(data)
            df_vina = df_vina.set_index("Ligand", drop=True)
            df_vina.sort_values(['Ligand', 'Receptor'],inplace=True)
            df_vina[df_vina.columns[1:]] = df_vina[df_vina.columns[1:]].apply(pd.to_numeric)
            df_vina.to_csv(csvName)
        
    # Rescoring tab functions (CONVEX and SMINA)
    def seleVinaLogsRESCORING(self):
        self.getDir('Select the Vina docking results folder', self.rescoringVinaLogs, 'd')
    
    def seleReceptorFolderRESCORING(self):
        self.getDir('Select the folder containing receptors PDB files', self.rescoringReceptor, 'd')
        
    def seleOutputFolderRESCORING(self):
        self.getDir('Select the folder to store rescoring results', self.rescoringOutput, 'd')
        
    def seleConvexFileRESCORING(self):
        self.getDir('Select the folder to store docking results', self.convexFile, 'f')
    
    def csvNameCONVEX(self):
        self.getDir('Save log file', self.convexSummary, 's')
    
    def rescoreCONVEX(self):
        preCheck = self.checkConversion('CONVEX')
        if preCheck[0] == True:
            message = 'The following errors were found:\n\n'
            for m in preCheck[1]:
                message = message + '\t•' + self.errorDict[m] + '\n'
            errorWidget = QtWidgets.QErrorMessage(self)
            errorWidget.setFixedSize(500,200)
            errorWidget.showMessage(message)
        else:
            receptor_dir = self.rescoringReceptor.text()
            vinaResults_dir = self.rescoringVinaLogs.text()
            rescoring_dir = self.rescoringOutput.text()
            convexPL = self.convexFile.text()
            rescoring_pdbdir = rescoring_dir +'PDB/'
            convexlogs_dir = rescoring_dir + 'ConvexLogs/' #This is the outputs folder
            vina_results = [o for o in os.listdir(vinaResults_dir) if o.lower() != 'logs'] #Make it better
            summary_fileName = self.convexSummary.text() #rescoring_dir + 'Sumario_CONVEX.csv'
            
            if not os.path.isdir(rescoring_dir):
                os.mkdir(rescoring_dir)
            if not os.path.isdir(rescoring_pdbdir):
                os.mkdir(rescoring_pdbdir)
            if not os.path.isdir(convexlogs_dir):
                os.mkdir(convexlogs_dir)
            
            for v in vina_results:
                pymol.cmd.load(vinaResults_dir + v)
                pymol.cmd.save(rescoring_pdbdir + v[:-6] + ".pdb", state = 0)
                pymol.cmd.reinitialize()
            
            summary_file = open(summary_fileName, "w")
            summary_file.write("Ligand,Receptor")
            N = int(self.convexN.currentText())
            for j in range(N):
                summary_file.write(',Affinity' + str(j+1))
            summary_file.write("\n")
            i = 0
            total = 2 * len(vina_results)
            for v in vina_results:
                ligandName = v.split('_')[1]
                receptorName = v.split('_')[2].split('.')[0]
                summary_file.write(ligandName + ',' + receptorName)
                outputName =  ''.join([convexlogs_dir,ligandName, '_', receptorName, '.txt'])
                command = ''.join([convexPL, ' --receptor ',receptor_dir, receptorName, '.pdb --ligand ', rescoring_pdbdir + v[:-6] + '.pdb >', outputName])
                os.system(command)
                i += 1
                self.convexProgress.setValue(int(i/total * 100))
                lst = []
                for line in self.readFile(outputName):
                    l = line.split()
                    if l[0]=="model":
                        lst.append(float(l[-1]))
                        #summary_file.write("," + l[-1])
                lst = sorted(lst)
                lst.reverse()
                for k in range(N):
                    summary_file.write("," + str(lst[k]))
                summary_file.write("\n")
                i += 1
                self.convexProgress.setValue(int(i/total * 100))
            summary_file.close()
            
            # Fix the csv file 
            df_convex = pd.read_csv(summary_fileName)
            df_convex = df_convex.set_index("Ligand", drop=True)
            df_convex.sort_values(['Ligand', 'Receptor'],inplace=True)
            df_convex[df_convex.columns[1:]] = df_convex[df_convex.columns[1:]].apply(pd.to_numeric)
            df_convex.to_csv(summary_fileName)
    
    def seleSminaFileRESCORING(self):
        self.getDir('Select the smina.static file', self.sminaFile, 'f')
    
    def csvNameSMINA(self):
        self.getDir('Save log file', self.sminaSummary, 's')
        
    def rescoreSMINA(self):
        preCheck = self.checkConversion('SMINA')
        if preCheck[0] == True:
            message = 'The following errors were found:\n\n'
            for m in preCheck[1]:
                message = message + '\t•' + self.errorDict[m] + '\n'
            errorWidget = QtWidgets.QErrorMessage(self)
            errorWidget.setFixedSize(500,200)
            errorWidget.showMessage(message)
        else:
            receptor_dir = self.rescoringReceptor.text()
            vinaResults_dir = self.rescoringVinaLogs.text()
            rescoring_dir = self.rescoringOutput.text()
            rescoring_sdfdir = rescoring_dir + 'SDF/'
            sminaST = self.sminaFile.text()
            scFunction = self.sminaFunction.currentText().lower()
            sminaScoring_dir = rescoring_dir + scFunction + 'Logs/'
            vina_results = [o for o in os.listdir(vinaResults_dir) if o.lower() != 'logs']
            summary_fileName = self.sminaSummary.text()            
            
            if not os.path.isdir(rescoring_sdfdir):
                os.mkdir(rescoring_sdfdir)
            if not os.path.isdir(sminaScoring_dir):
                os.mkdir(sminaScoring_dir)
                
            N = int(self.sminaN.currentText())
            dic = {'Ligand':[], 'Receptor':[]}
            columns = ['Ligand', 'Receptor']
            for j in range(N):
                aff = 'Affinity' + str(j+1)
                dic[aff] = []
                columns.append(aff)
            summary = pd.DataFrame(dic)
            i = 0
            total = 2 * len(vina_results)
            for v in vina_results:
                ligandName = v.split('_')[1]
                receptorName = v.split('_')[2].split('.')[0]
                source = vinaResults_dir + v
                ligand_sdf = rescoring_sdfdir + ligandName + '_' + receptorName + '.sdf'
                dest = sminaScoring_dir + ligandName + '_' + receptorName + '.sdf'
                pymol.cmd.load(source)
                pymol.cmd.save(ligand_sdf, state = 0, format = 'sdf')
                pymol.cmd.reinitialize()
                affinities = [ligandName, receptorName]
                #command = ''.join(['./smina.static -r ', receptor_dir, receptorName, '.pdb -l ', ligand_sdf, ' -o ', dest, ' --seed 0 --score_only --scoring ad4_scoring'])
                command = ''.join([sminaST, ' -r ', receptor_dir, receptorName, '.pdb -l ', ligand_sdf, ' -o ', dest, ' --seed 0 --score_only --scoring ', scFunction])
                os.system(command)
                p = False
                i +=1 
                self.sminaProgress.setValue(int(i/total * 100))
                for line in self.readFile(dest):
                    if '> <minimizedAffinity>' in line:
                        p = True
                    elif p == True:
                        affinities.append(float(line))
                        p = False
                    else:
                        p = False
                affinities[2:] = sorted(affinities[2:])
                newLine = pd.DataFrame(affinities[:N+2],index = columns).transpose()
                summary = summary.append(newLine)
                i +=1 
                self.sminaProgress.setValue(int(i/total * 100))
                
            summary.sort_values(['Ligand', 'Receptor'],inplace=True)
            summary.set_index(['Ligand'], inplace=True)
            summary[summary.columns[1:]] = summary[summary.columns[1:]].apply(pd.to_numeric)
            summary.to_csv(summary_fileName)
    
    # Cluster Analysis tab functions
    
    def addFileCLUSTER(self):
        prompt = 'Select a docking or rescoring summary file'
        chosen_file = str(QtWidgets.QFileDialog.getOpenFileName(self, prompt, filter='*.csv')[0])
        name = chosen_file.split('/')[-1][:-4]
        self.summaryList.addItem(name)
        self.summaries[name] = chosen_file
        if self.summaryList.count() > 1 and not self.makeScoresFileButton.isEnabled():
            self.makeScoresFileButton.setEnabled(True)
    
    def loadFileCLUSTER(self):
        prompt = 'Select a joined summary file'
        chosen_file = str(QtWidgets.QFileDialog.getOpenFileName(self, prompt, filter='*.csv')[0])
        self.clusterDF = pd.read_csv(chosen_file, index_col=0)
        if 'Receptor' in list(self.clusterDF.columns):
            print('Error! Invalid file selected')
            return
        else:
            self.loadedFile = True
            self.summaryList.clear()
            self.makeScoresFileButton.setEnabled(False)
            l = list()
            for c in self.clusterDF.columns:
                if c not in ['Ligand','Receptor','Cluster','Type']:
                    self.summaryList.addItem(c)
                    l.append(c)
            self.summaries = self.clusterDF[l].to_dict()
            self.posContList.clear()
            self.negContList.clear()
            self.functionList.clear()
            self.posContList.addItems(pd.unique(list(self.clusterDF.index)))
            self.negContList.addItems(pd.unique(list(self.clusterDF.index)))
            self.functionList.addItems(l)
            pca = PCA()
            pca = pca.fit(self.clusterDF[list(self.summaries.keys())])
            fig = Figure()
            pcaCanvas = FigureCanvas(fig)
            pcaViewLayout = QtWidgets.QHBoxLayout()
            pcaViewLayout.addWidget(pcaCanvas)
            self.pcaFrame.setLayout(pcaViewLayout)
            axes = fig.add_subplot()
            axes.plot(100-pca.explained_variance_)
            axes.grid()
    
        
    def makeFileCLUSTER(self):
        Scores = {}
        saveDialog = QtWidgets.QFileDialog(self, "Save score file", filter = '*.csv')
        if saveDialog.exec():
            scores_file = saveDialog.selectedFiles()[0]
            if scores_file[-4:] != '.csv':
                scores_file = scores_file + '.csv'
            if self.summaryList.count == 0:
                print('Error: You need to add at least one summary file')
            else:
               if not self.loadedFile:
                   for key in self.summaries.keys():
                           file = self.summaries[key]
                           df = pd.read_csv(file,index_col=0)
                           try:
                               df.sort_values(['Ligand', 'Receptor'],inplace=True)
                               r = 1
                           except:
                               df.sort_values(['Ligand'],inplace=True)
                               r = 0
                           n = len(df.columns) - r
                           df_names = []
                           df_scores = []
                           for i in range(len(df)):
                               for j in range(n):
                                   df_names.append(df.index[i])
                               df_scores.extend(list(df.iloc[i][r:]))
                           #I need to ensure that the names match
                           Scores['Ligand'] = df_names
                           Scores[key] = df_scores   
                   self.posContList.clear()
                   self.negContList.clear()
                   self.functionList.clear()
                   self.posContList.addItems(pd.unique(df_names))
                   self.negContList.addItems(pd.unique(df_names))
                   self.functionList.addItems(pd.unique(list(self.summaries.keys())))
                   print(Scores)
                   Scores_df = pd.DataFrame(Scores)
                   Scores_df = Scores_df.set_index('Ligand')
               else:
                   Scores_df = self.clusterDF.copy()
               Scores_df.to_csv(scores_file)    
               pca = PCA()
               pca = pca.fit(Scores_df[list(self.summaries.keys())])
               fig = Figure()
               pcaCanvas = FigureCanvas(fig)
               pcaViewLayout = QtWidgets.QHBoxLayout()
               pcaViewLayout.addWidget(pcaCanvas)
               self.pcaFrame.setLayout(pcaViewLayout)
               axes = fig.add_subplot()
               axes.plot(100-pca.explained_variance_)
               axes.grid()
        else:
            pass
           
    
    def choseColorCLUSTER(self):
        if self.cmapRadio.isChecked():
           self.cmapBox.setEnabled(True)
           self.clusterColorButton.setEnabled(False)
           self.alphaSpinBox.setEnabled(False)
        else:
           self.cmapBox.setEnabled(False)
           self.clusterColorButton.setEnabled(True)
           self.alphaSpinBox.setEnabled(True)
        
    def choseTypeCLUSTER(self):
        # There is really no need for anything here
        pass
    
    def cluster(self):
        saveDialog = QtWidgets.QFileDialog(self, "Save score file")
        if saveDialog.exec():
            fileName = saveDialog.selectedFiles()[0]
            print(fileName)
        else:
            print('not ok')
            pass
        
    def makeCLUSTER(self):
        fileDialog = QtWidgets.QFileDialog(self, "Open scores file", filter = '*.csv')
        if fileDialog.exec():
            scores_file = fileDialog.selectedFiles()[0]
            if len(self.summaries.keys()) > 0:
                Scores = pd.read_csv(scores_file, index_col = 0)
                n = self.clusterN.value()
                km = KMeans(n_clusters = n)
                positives = [item.text() for item in self.posContList.selectedItems()]
                negatives = [item.text() for item in self.negContList.selectedItems()]
                functions = [item.text() for item in self.functionList.selectedItems()]
                if self.clusterAllRadio.isChecked():
                    #cScores = km.fit_predict(Scores[list(self.summaries.keys())])
                    cScores = km.fit_predict(Scores[functions])
                    Scores['Cluster'] = cScores
                    Scores['Type'] = [None]*len(Scores)
                    Scores.loc[Scores.index.isin(positives), 'Type'] = '+'
                    Scores.loc[Scores.index.isin(negatives), 'Type'] = '-'
                    Scores.loc[(~Scores.index.isin(positives) & ~Scores.index.isin(negatives)), 'Type'] = 'o'
                    self.clusterDF = Scores
                    total_clusters = n
                else:
                    control_df = Scores.loc[(Scores.index.isin(positives) | Scores.index.isin(negatives))]
                    oil_df = Scores.loc[(~Scores.index.isin(positives) & ~Scores.index.isin(negatives))]
                    #control_scores = km.fit_predict(control_df[list(self.summaries.keys())])
                    control_scores = km.fit_predict(control_df[functions])
                    #oil_scores = km.predict(oil_df[list(self.summaries.keys())])
                    if len(oil_df) != 0:
                        oil_scores = km.predict(oil_df[functions])
                    #for i in range(len(oil_scores)):
                    #    oil_scores[i] += n
                    control_df['Cluster'] = control_scores
                    oil_df['Cluster'] = oil_scores
                    oil_df['Cluster'] += n
                    fScores = pd.concat([control_df, oil_df])
                    fScores['Type'] = [None]*len(fScores)
                    fScores.loc[fScores.index.isin(positives), 'Type'] = '+'
                    fScores.loc[fScores.index.isin(negatives), 'Type'] = '-'
                    fScores.loc[(~fScores.index.isin(positives) & ~fScores.index.isin(negatives)), 'Type'] = 'o'
                    self.clusterDF = fScores
                    total_clusters = 2*n
                self.showCurrentButton.setEnabled(True)
                self.showDistributionsButton.setEnabled(True)
                self.updateClusterButton.setEnabled(True)
                self.seleClusterViewGroup.setEnabled(True)
                self.reclusterButton.setEnabled(True)
                self.xAxisBox.clear()
                self.xAxisBox.addItems(list(self.summaries.keys()))
                self.xAxisBox.setCurrentIndex(0)
                self.xAxisBox.setEnabled(True)
                self.yAxisBox.clear()
                self.yAxisBox.addItems(list(self.summaries.keys()))
                self.yAxisBox.setCurrentIndex(0)
                self.yAxisBox.setEnabled(True)
                self.zAxisBox.clear()
                self.zAxisBox.addItems(list(self.summaries.keys()))
                self.zAxisBox.setCurrentIndex(0)
                self.zAxisBox.setEnabled(True)
                self.clusterNumberBox.clear()
                self.selectClusters.clear()
                self.clusterNumberBox.addItem('All')
                self.clusterNumberBox.setEnabled(True)
                self.colorByGroup.setEnabled(True)
                self.colorGroup.setEnabled(True)
                self.clusterColorButton.setStyleSheet("background-color: #bebebe")
                self.reclusterFunctionList.clear()
                self.reclusterFunctionList.addItems(list(self.summaries.keys()))
                for i in range(total_clusters):
                    self.clusterNumberBox.addItem(str(i))
                    self.selectClusters.addItem(str(i))
                self.compoundTable.setRowCount(0) 
                self.compoundTable.setRowCount(len(pd.unique(self.clusterDF.index)))
                i = 0
                for compound in pd.unique(self.clusterDF.index):
                    col1 = QtWidgets.QTableWidgetItem(compound)
                    col2 = QtWidgets.QTableWidgetItem(str(self.clusterDF.loc[self.clusterDF.index == compound , 'Type'][0]))
                    col3 = QtWidgets.QTableWidgetItem('100')
                    self.compoundTable.setItem(i,0, col1)
                    self.compoundTable.setItem(i,1, col2)
                    self.compoundTable.setItem(i,2, col3)
                    i += 1
                self.kmeans = km
                joblib.dump(km, scores_file[:-4] + '.kmeans')
                self.clusterDF.to_csv(scores_file)
      
    def loadCLUSTER(self):
        fileDialog = QtWidgets.QFileDialog(self, "Open scores file", filter = '*.csv')
        if fileDialog.exec():
            scores_file = fileDialog.selectedFiles()[0]
            kmeans_file = scores_file[:-4] + '.kmeans'
            if scores_file[-4:] == '.csv' and os.path.isfile(kmeans_file):
                self.clusterDF = pd.read_csv(scores_file,index_col=0)
                self.summaries = {}
                self.kmeans = joblib.load(kmeans_file)
                for c in pd.unique(self.clusterDF.columns):
                    if c in ['Cluster', 'Type']:
                        pass
                    else:
                        self.summaries[c] = []
                self.clusterNumberBox.setEnabled(False)
                self.showCurrentButton.setEnabled(True)
                self.showDistributionsButton.setEnabled(True)
                self.updateClusterButton.setEnabled(True)
                self.seleClusterViewGroup.setEnabled(True)
                self.xAxisBox.clear()
                self.xAxisBox.addItems(list(self.summaries.keys()))
                self.xAxisBox.setCurrentIndex(0)
                self.yAxisBox.clear()
                self.yAxisBox.addItems(list(self.summaries.keys()))
                self.yAxisBox.setCurrentIndex(1)
                self.zAxisBox.clear()
                self.zAxisBox.addItems(list(self.summaries.keys()))
                self.zAxisBox.setCurrentIndex(2)
                self.xAxisBox.setEnabled(True)
                self.yAxisBox.setEnabled(True)
                self.zAxisBox.setEnabled(True)
                self.colorByGroup.setEnabled(True)
                self.colorGroup.setEnabled(True)
                self.reclusterButton.setEnabled(True)
                total_clusters = len(pd.unique(self.clusterDF.Cluster))
                self.clusterNumberBox.clear()
                self.selectClusters.clear()
                self.clusterNumberBox.addItem('All')
                self.clusterNumberBox.setEnabled(True)
                self.clusterColorButton.setStyleSheet("background-color: #bebebe")
                self.reclusterFunctionList.clear()
                self.reclusterFunctionList.addItems(list(self.summaries.keys()))
                for i in range(total_clusters):
                    self.clusterNumberBox.addItem(str(i))
                    self.selectClusters.addItem(str(i))
                self.compoundTable.setRowCount(0)
                self.compoundTable.setRowCount(len(pd.unique(self.clusterDF.index)))
                i = 0
                for compound in pd.unique(self.clusterDF.index):
                    col1 = QtWidgets.QTableWidgetItem(compound)
                    col2 = QtWidgets.QTableWidgetItem(str(self.clusterDF.loc[self.clusterDF.index == compound , 'Type'][0]))
                    col3 = QtWidgets.QTableWidgetItem('100')
                    self.compoundTable.setItem(i,0, col1)
                    self.compoundTable.setItem(i,1, col2)
                    self.compoundTable.setItem(i,2, col3)
                    i += 1
                fig = Figure()
                static_canvas = FigureCanvas(fig)
                try:
                    self.clusterViewFrame.layout().takeAt(0).widget().deleteLater()
                except:
                    self.clusterViewFrame.layout().takeAt(0)
                self.clusterViewFrame.layout().addWidget(static_canvas)
                axes = fig.add_subplot(111, projection='3d')
                #axes.scatter(self.clusterDF[self.xAxisBox.currentText()], self.clusterDF[self.yAxisBox.currentText()], self.clusterDF[self.zAxisBox.currentText()],c=self.clusterDF['Cluster'], cmap = self.cmapBox.currentText())
                axes.set_xlabel(self.xAxisBox.currentText())
                axes.set_ylabel(self.yAxisBox.currentText())
                axes.set_zlabel(self.zAxisBox.currentText())
                self.axes = axes
                clt = sorted(pd.unique(self.clusterDF.Cluster))
                if self.cmapRadio.isChecked() or len(self.rgba) == 0:
                    cmap = cm.get_cmap(self.cmapBox.currentText())
                    self.alphas = [1.0]*len(clt)
                    self.rgba = cmap(np.linspace(0,1,len(clt)))
                for y, c, a in zip(clt, self.rgba, self.alphas):
                    part = self.clusterDF.loc[self.clusterDF.Cluster == y]
                    axes.scatter(part[self.xAxisBox.currentText()], part[self.yAxisBox.currentText()], part[self.zAxisBox.currentText()], color=c, alpha = a)
                if self.centersCheck.isChecked():
                    x = self.xAxisBox.currentIndex()
                    y = self.yAxisBox.currentIndex()
                    z = self.zAxisBox.currentIndex()
                    centroids = self.kmeans.cluster_centers_
                    axes.scatter(centroids[:,x], centroids[:,y], centroids[:,z], c = 'k', marker = 'D', s = 10)
            else:
                print("Kmeans file doesn't exist")
                
    def showCLUSTER(self):
        fig = Figure()
        static_canvas = FigureCanvas(fig)
        try:
            self.clusterViewFrame.layout().takeAt(0).widget().deleteLater()
        except:
            self.clusterViewFrame.layout().takeAt(0)
        self.clusterViewFrame.layout().addWidget(static_canvas)
        axes = fig.add_subplot(111, projection='3d')
        #axes.scatter(self.clusterDF[self.xAxisBox.currentText()], self.clusterDF[self.yAxisBox.currentText()], self.clusterDF[self.zAxisBox.currentText()],c=self.clusterDF['Cluster'], cmap = self.cmapBox.currentText())
        axes.set_xlabel(self.xAxisBox.currentText())
        axes.set_ylabel(self.yAxisBox.currentText())
        axes.set_zlabel(self.zAxisBox.currentText())
        self.axes = axes
        if self.colorByCluster.isChecked():
            clt = sorted(pd.unique(self.clusterDF.Cluster))
        else:
            clt = sorted(pd.unique(self.clusterDF.Type))
        if self.cmapRadio.isChecked() or len(self.rgba) == 0:
            cmap = cm.get_cmap(self.cmapBox.currentText())
            self.alphas = [1.0]*len(clt)
            self.rgba = cmap(np.linspace(0,1,len(clt)))
        for y, c, a in zip(clt, self.rgba, self.alphas):
            if self.colorByCluster.isChecked():
                part = self.clusterDF.loc[self.clusterDF.Cluster == y]
            else:
                part = self.clusterDF.loc[self.clusterDF.Type == y]
            axes.scatter(part[self.xAxisBox.currentText()], part[self.yAxisBox.currentText()], part[self.zAxisBox.currentText()], color=c, alpha = a)
            #axes.scatter(self.clusterDF[self.xAxisBox.currentText()], self.clusterDF[self.yAxisBox.currentText()], self.clusterDF[self.zAxisBox.currentText()],c=self.clusterDF['Cluster'], cmap = self.cmapBox.currentText())
        if self.centersCheck.isChecked():
            x = self.xAxisBox.currentIndex()
            y = self.yAxisBox.currentIndex()
            z = self.zAxisBox.currentIndex()
            centroids = self.kmeans.cluster_centers_
            C = np.pad(centroids, ((0,0),(0,len(self.summaries.keys())-(centroids.shape[1]))))
            print(C[:,x])
            print(C[:,y])
            print(C[:,z])
            axes.scatter(C[:,x], C[:,y], C[:,z], c = 'k', marker = 'D', s = 1000)
    
    def showMatrixCLUSTER(self):
        fig = Figure()
        static_canvas = FigureCanvas(fig)
        try:
            self.clusterViewFrame.layout().takeAt(0).widget().deleteLater()
        except:
            self.clusterViewFrame.layout().takeAt(0)
        self.clusterViewFrame.layout().addWidget(static_canvas)
        axes = fig.add_subplot(111)
        self.axes = axes
        conv = {'-': 0, '+': 1, 'o':2}
        if self.colorByCluster.isChecked():
            pd.plotting.scatter_matrix(self.clusterDF[list(self.summaries.keys())], ax =axes, alpha=0.2, figsize=(9, 9), marker = "o", c = self.clusterDF['Cluster'], cmap = self.cmapBox.currentText())
        else:
            colorList = [conv[b] for b in self.clusterDF['Type']]
            pd.plotting.scatter_matrix(self.clusterDF[list(self.summaries.keys())], ax =axes, alpha=0.2, figsize=(9, 9), marker = "o", c = colorList, cmap = self.cmapBox.currentText())
    
    def highlightCLUSTER(self):
        if self.clusterNumberBox.isEnabled():
            self.compoundTable.setRowCount(0)
            if self.clusterNumberBox.currentText() == 'All':
                self.compoundTable.setRowCount(len(pd.unique(self.clusterDF.index)))
                self.clusterColorButton.setStyleSheet("background-color: #bebebe")
                i = 0
                for compound in pd.unique(self.clusterDF.index):
                    col1 = QtWidgets.QTableWidgetItem(compound)
                    col2 = QtWidgets.QTableWidgetItem(str(self.clusterDF.loc[self.clusterDF.index == compound , 'Type'][0]))
                    col3 = QtWidgets.QTableWidgetItem('100')
                    self.compoundTable.setItem(i,0, col1)
                    self.compoundTable.setItem(i,1, col2)
                    self.compoundTable.setItem(i,2, col3)
                    i += 1   
            else:
                repeats = self.clusterDF.loc[self.clusterDF.index == pd.unique(self.clusterDF.index)[0]].count()[0]
                try:
                    n = int(self.clusterNumberBox.currentText())
                except:
                    n = ''
                try:
                    selectedColor =  colors.to_hex(self.rgba[int(n)])
                except:
                    selectedColor = '#bebebe'
                self.clusterColorButton.setStyleSheet("background-color: " + selectedColor)
                selected = self.clusterDF.loc[self.clusterDF.Cluster.isin([str(n), n])]
                self.compoundTable.setRowCount(len(pd.unique(selected.index)))
                i = 0
                for compound in pd.unique(selected.index):
                    percent = selected.loc[selected.index == pd.unique(selected.index)[i]].count()[0]/repeats*100
                    col1 = QtWidgets.QTableWidgetItem(compound)
                    col2 = QtWidgets.QTableWidgetItem(str(self.clusterDF.loc[self.clusterDF.index == compound , 'Type'][0]))
                    col3 = QtWidgets.QTableWidgetItem(str(int(np.rint(percent))))
                    self.compoundTable.setItem(i,0, col1)
                    self.compoundTable.setItem(i,1, col2)
                    self.compoundTable.setItem(i,2, col3)
                    i += 1            
    
    def recolorCLUSTER(self):
        cDialog = QtWidgets.QColorDialog()
        new_color = cDialog.getColor()
        cl = colors.to_rgba(new_color.name())
        if self.colorsRadio.isChecked() and len(self.rgba) > 0 and self.clusterNumberBox.currentText() != 'All':
            self.clusterColorButton.setStyleSheet("background-color: " + new_color.name())
            self.alphas[int(self.clusterNumberBox.currentText())] = self.alphaSpinBox.value()
            self.rgba[int(self.clusterNumberBox.currentText())] = cl
            self.showCLUSTER()
        
    def showCentersCLUSTER(self):
        self.showCLUSTER()
        
    def changeTypeCLUSTER(self):
        if self.view3Dradio.isChecked():
            self.showCLUSTER()
            self.clusterNumberBox.setEnabled(False)
            self.clusterNumberBox.clear()
            self.selectClusters.clear() ###################
            self.clusterNumberBox.addItem('All')
            if self.colorByCluster.isChecked():
                items = [str(n) for n in pd.unique(self.clusterDF.Cluster)]
                self.clusterNumberBox.addItems(items)
            else:
                self.clusterNumberBox.addItems(pd.unique(self.clusterDF.Type))
            self.clusterNumberBox.setEnabled(True)
        else:
            self.showMatrixCLUSTER()
        
    def updateCLUSTER(self):
        if self.isEnabled():
            self.showCLUSTER()
        
    def showDistCLUSTER(self):
        dialog = distDialog(self.clusterDF)
        functions = [item for item in self.clusterDF if item not in ['Ligand', 'Cluster', 'Type']]
        dialog.distFunctionBox.clear()
        dialog.distFunctionBox.addItems(functions)
        dialog.exec_()
        
    def reCLUSTER(self):
        fileDialog = QtWidgets.QFileDialog(self, "Save recluster file", filter = '*.csv')
        if fileDialog.exec():
            scores_file = fileDialog.selectedFiles()[0]
            if len(self.summaries.keys()) > 0:
                Scores = self.clusterDF
                clusterList = [int(item.text()) for item in self.selectClusters.selectedItems()]
                functions = [item.text() for item in self.reclusterFunctionList.selectedItems()]
                if self.selePosOnlyRadio.isChecked():  
                    alt = ['+']
                    forb = ['-']
                elif self.seleNegOnlyRadio.isChecked():
                    alt = ['-']
                    forb = ['+']
                elif self.seleBothRadio.isChecked():
                    alt = ['+', '-']
                    forb = []
                else:
                    alt = []
                    forb = ['+', '-']
                if self.eliminateBox.isChecked():
                    Scores = Scores.loc[(Scores.Cluster.isin(clusterList) & ~Scores.Type.isin(forb))] #| Scores.Type.isin(alt) 
                else:
                    Scores = Scores.loc[(Scores.Cluster.isin(clusterList) | Scores.Type.isin(alt))]
                n = self.newClusterN.value()
                km = KMeans(n_clusters = n)
                #cScores = km.fit_predict(Scores[list(self.summaries.keys())])
                cScores = km.fit_predict(Scores[functions])
                Scores['Cluster'] = cScores
                self.clusterDF = Scores
                self.clusterNumberBox.clear()
                self.selectClusters.clear()
                self.clusterNumberBox.addItem('All')
                self.clusterNumberBox.setEnabled(True)
                self.clusterColorButton.setStyleSheet("background-color: #bebebe")
                for i in range(n):
                    self.clusterNumberBox.addItem(str(i))
                    self.selectClusters.addItem(str(i))
                self.compoundTable.setRowCount(0) 
                self.compoundTable.setRowCount(len(pd.unique(self.clusterDF.index)))
                i = 0
                for compound in pd.unique(self.clusterDF.index):
                    col1 = QtWidgets.QTableWidgetItem(compound)
                    col2 = QtWidgets.QTableWidgetItem(str(self.clusterDF.loc[self.clusterDF.index == compound , 'Type'][0]))
                    col3 = QtWidgets.QTableWidgetItem('100')
                    self.compoundTable.setItem(i,0, col1)
                    self.compoundTable.setItem(i,1, col2)
                    self.compoundTable.setItem(i,2, col3)
                    i += 1
                self.kmeans = km
                joblib.dump(km, scores_file[:-4] + '.kmeans')
                self.clusterDF.to_csv(scores_file)

class distDialog(QtWidgets.QDialog):
    def __init__(self, data):
        super(distDialog, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('distDiag.ui', self) # Load the .ui file
        self.dataFrame = data
        # Add action to the drop box
        self.distFunctionBox.currentTextChanged.connect(self.showDist)
        
    def showDist(self):
        try:
            self.distFrame.layout().takeAt(0).widget().deleteLater()
        except:
            pass
        fig = Figure()
        static_canvas = FigureCanvas(fig)
        self.distFrame.layout().addWidget(static_canvas)
        axes = fig.add_subplot(111)
        axes.set_title(self.distFunctionBox.currentText())
        axes.set_ylabel('Probability')
        axes.set_xlabel('Docking score')
        f = self.distFunctionBox.currentText()
        posdf = self.dataFrame.loc[self.dataFrame.Type=='+', f]
        negdf = self.dataFrame.loc[self.dataFrame.Type=='-', f]
        lower = min(self.dataFrame[f])
        upper = max(self.dataFrame[f])
        x = np.linspace(lower,upper,100)
        stdp = posdf.std()
        meanp = posdf.mean()
        yp = stats.norm(loc = meanp, scale = stdp)
        stdn = negdf.std()
        meann = negdf.mean()
        yn = stats.norm(loc = meann, scale = stdn)
        kl = sum(scipy.special.rel_entr(yp.pdf(x),yn.pdf(x)))
        self.statsBar.setText('KL-Divergence (100 pts) = ' + str(np.round(kl,5)))
        
        if self.histogramCheckbox.isChecked():
            axes.hist(posdf, density=True, color = '#90C8FC', alpha=0.5)
            axes.hist(negdf, density=True, color = '#FCD78D', alpha=0.5)
            
        if self.distributionCheckBox.isChecked():
            # for positive controls
            
            axes.plot(x,yp.pdf(x), color='#003B73', label='Positive controls')
            # for negative controls
            axes.plot(x,yn.pdf(x), color = '#FF9900', label='Negative controls')
            axes.legend()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Interface()
    window.show()
    sys.exit(app.exec_())