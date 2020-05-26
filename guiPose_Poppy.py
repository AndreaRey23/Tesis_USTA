from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QFileDialog, QWidget, QMessageBox, QTableWidget, QTableWidgetItem
from PyQt5.uic import loadUi
from PyQt5.QtCore import QTimer, QRegExp, Qt, QEvent, QRect
from PyQt5.QtGui import QPixmap, QImage, QRegExpValidator, QMovie
from PyQt5 import QtGui

import os, sys
import numpy as np
import cv2
import threading
import time
import tensorflow as tf
from collections import deque

import Pose_Real_Time_Prediccion as pose
import Processing_Data as data
import Prediction_Models as models

import signal


###Clases de archivos importados###
PoseEstimation = pose.Pose()
Data_process = data.getData()
Predict_Model = models.Models_P()

# parent process
COUNTDOWN_TIME = 3
cTime = COUNTDOWN_TIME

class mainwindow(QMainWindow):
    def __init__(self,pipelineStdin,pipelineStdout,childProcess):
        #Inicializacion interfax
        super(mainwindow,self).__init__(None)
        loadUi('mainwindow.ui',self)
        self.stop_threads=False
        self.time_record=5
        self.setFixedWidth(990)
        self.setFixedHeight(630)
        self.Start_Pose.setText("Empezar Pose")
        self.Stop_Pose.setText("Detener Pose")
        self.Stop_Pose.setEnabled(False)
        self.Type_Prediction.addItem('')
        self.Type_Prediction.addItem('SVM')
        self.Type_Prediction.addItem('RNN-Many_to_One')
        self.Img_Pause.hide()
        self.Img_Play.hide()
        self.countdownTime.hide()
        self.Img_Record.show()
        self.Img_Record.setStyleSheet('color: #FE9E89')
        self.Move_BackGround.setEnabled(False)
        
        self.UmbralSVM.setRange(10,99)
        self.UmbralSVM.setValue(55)
        self.UmbralRNN.setRange(10,99)
        self.UmbralRNN.setValue(98)

        #Animacion Espera
        self.load.setText("")
        self.movie = QMovie('img/loading.gif')
        self.load.setMovie(self.movie)
        self.Background_Load.hide()

        #Manejador Tuberias
        self.pipelineStdin = pipelineStdin
        self.pipelineStdout = pipelineStdout
        self.childProcess = childProcess
        self.pipeCode = {'No_ROS':43, 'Ex_Move':23, 'Disp':56, 'End':62, 'Error':37}

        #Inicializacion Hilos
        self.Start_Pose.pressed.connect(self.startCountdown)
        self.Stop_Pose.pressed.connect(self.Stop_Thread_Real_Time)
        
        #Inicializacion contador
        self.COUNTDOWN_TIME = None
        
        #Inicializacion Cola
        self.maxlen_que = 66
        self.lastPointsPose = deque([],self.maxlen_que)	

        #Variable maximo frames
        self.ratePredict= 10

        #Inicializacion Servicion de ROS
        self.Start_Thread_Ros_Service()
        self.Thread_Availability = threading.Thread()

    def live_video(self,camera_port=0):
        self.Img_Play.show()
        self.Img_Play.setStyleSheet('color: #39E2B7')
        self.Img_Record.hide()
        self.countdownTime.hide()
        self.Stop_Pose.setEnabled(True)
        #Capturar del puerto
        video_capture = cv2.VideoCapture(camera_port)
        countFrames = 0
        while True:
            if self.stop_threads:
                print("stop threads")
                break
            # Captura frame-por-frame
            try: 
                ret, frame = video_capture.read()
                countFrames = countFrames+1
            except:
                print("error the take a image")
                ret = False
                self.Stop_Thread_Real_Time()
                break

            if ret == True:
                try:
                    self.show_capture_video(frame)
                    framePose = frame.copy()
                    points_Frame=PoseEstimation.PoseFrame(framePose)
                    if not Data_process.verify_Person(points_Frame,9):
                        self.Move.setText("No Person")
                    else:
                        if len(self.lastPointsPose) >= self.maxlen_que:
                            if countFrames >= self.ratePredict:
                                prediccion = self.Prediction()
                                print('Repeat prediccion: ',Predict_Model.repeatPredict(prediccion))
                                if Predict_Model.repeatPredict(prediccion) and not self.Thread_Availability.is_alive():
                                    self.Move.setText(Data_process.classes[prediccion])
									#Iniciar hilo Movimiento
                                    self.Thread_Availability = self.Start_rosPoppy_Availability(prediccion)
                                countFrames = 0
                            #Desencolar frame
                            self.lastPointsPose.pop()
                        self.lastPointsPose.insert(0,points_Frame)
                    self.show_pose_video(framePose)
                    #video_capture.release()
                except:
                    print("Error during the convertion")
                    self.Stop_Thread_Real_Time()
                    #self.Start_Thread_Real_Time()
            else:
                self.Stop_Thread_Real_Time()
                print("error the take a image")
                break
            
    def plotPose(self,img_rgb=[]):
        #MOVIMIENTO REAL
        #Imagen en blanco
        img = np.ones([900,900,3])
        r,g,b = cv2.split(img)
        img_bgr = cv2.merge([b,g,r])
        data=pose.getData()
        points_pose =  np.array([frame[0] for frame in np.asarray(self.lastPointsPose)])
        points_average=PoseEstimation.avgCentered(points_pose,14)
        pointsPose_AVG=Data_process.getOneSampleVideo(points_average)
        #Puntos Dataset
        for frame in pointsPose_AVG:
            #print(frame,'frame')
            px,py=Data_process.getPoints(frame,400)
            #print(px,py)
            #plt.plot(py,px,'o')
            #Draw Points
            img_bgr = img_rgb.copy()
            Data_process.DrawPoints(img_bgr,px,py)
            self.show_pose_video(img_bgr)
            time.sleep(0.1)

    def show_capture_video(self, img_rgb=[]):
        #Captura de la imagen y visualizacion en la interfaz
        img = cv2.resize(img_rgb,(int(400),int(350)))
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.Sequence.setPixmap(pix)

    def show_pose_video(self,img_pose=[]):
        #Captura de la imagen y visualizacion en la interfaz
        img = cv2.resize(img_pose,(int(400),int(350)))
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.View_Pose.setPixmap(pix)

    def Start_Thread_Real_Time(self):
        #Hilo de visualizacion en tiempo real
        t=threading.Thread(name='hilo',target=self.live_video)
        t.start()
        self.stop_threads = False
        print("Real Time Thread(#4362)")

    def Stop_Thread_Real_Time(self):
        #Detener hilo de tiempo real
        self.stop_threads = True
        self.Start_Pose.setEnabled(True)
        self.Stop_Pose.setEnabled(False)
        self.Img_Record.show()
        self.Img_Record.setStyleSheet('color: #FE9E89')
        self.Img_Play.hide()
        print("Stop Real Time Thread(#4362)")

    def startCountdown(self):
        #Contador para iniciar la grabacion y visualizacion del pose
        global cTime
        cTime = COUNTDOWN_TIME
        self.countdownTime.show()
        self.Img_Play.hide()
        self.Img_Pause.hide()
        self.Img_Record.hide()
        self.timerCountdown = QTimer(self)
        self.timerCountdown.timeout.connect(self.update_countdown)
        self.timerCountdown.start(1)
        self.Start_Pose.setEnabled(False)

    def update_countdown(self):
        #Acualizacion del contador
        global cTime
        self.countdownTime.setText(str(cTime))
        cTime -= 1
        if cTime == -1:
            self.countdownTime.hide()
            self.Img_Record.show()
            self.Img_Play.hide()
            self.Img_Pause.hide()
            self.timerCountdown.stop()
            self.countdownTime.setText("")
            self.Start_Thread_Real_Time()
            self.Stop_Pose.setEnabled(True)
        else:
            self.timerCountdown.start(1000)

    def Prediction(self):
        #Move Predicction
        prediction_Move = 20
        #threshold_Algoritmos
        threshold_svm=self.UmbralSVM.value()/100
        #threshold_rnn=self.UmbralRNN.value()/1000000
        threshold_rnn=self.UmbralRNN.value()

        #Procesamiento puntos del Pose para la prediccion
        points_pose =  np.array([frame[0] for frame in np.asarray(self.lastPointsPose)])[::-1]
        points_sample = Data_process.getOneSampleVideo(points_pose)
        batch_x = Data_process.avgCenterSample(points_sample)
        if(self.Type_Prediction.currentText() == 'SVM' and len(batch_x) >= Data_process.timesteps):
            svm_prediction = Predict_Model.SVM_Model(batch_x,threshold_svm)
            prediction_Move =  svm_prediction
            # self.Move.setText(Data_process.classes[svm_prediction])
            print('movimiento prediccion SVM:',Data_process.classes[svm_prediction])
        elif(self.Type_Prediction.currentText() == 'RNN-Many_to_One' and len(batch_x) >= Data_process.timesteps):
            #prediccion=Predict_Model.RNN_Model(batch_x,0.9999+threshold_rnn)
            prediccion=Predict_Model.RNN_Model(batch_x,threshold_rnn)
            prediction_Move = prediccion
            #self.Move.setText(Data_process.classes[prediccion])
            print('movimiento prediccion rnn: ',Data_process.classes[prediccion])
        else:
            self.Move.setText("No Algorithm")

        return prediction_Move

    def closeEvent(self, event):
    
        box = QMessageBox()
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle('Cerrar')
        box.setText('Esta seguro de cerrar la aplicacion?')
        box.setStandardButtons(QMessageBox.Yes|QMessageBox.No)
        buttonAccept = box.button(QMessageBox.Yes)
        buttonAccept.setText('Si')
        buttonRefuse = box.button(QMessageBox.No)
        buttonRefuse.setText('No')
        box.move(self.pos().x()+self.width()/3,self.pos().y()+self.height()/2)
        box.exec_()

        if box.clickedButton() == buttonAccept:
            self.Stop_Thread_Real_Time()
            os.kill(self.childProcess, signal.SIGKILL)
            event.accept()
        elif box.clickedButton() == buttonRefuse:
            event.ignore()

    def Start_Thread_Ros_Service(self):
        #Hilo de servicios de ROS
        t = threading.Thread(target=self.rosService_Poppy, name='Daemon')
        t.setDaemon(True)
        self.Start_Pose.setEnabled(False)
        self.Background_Load.show()
        self.load.show()
        self.load_text.show()
        self.movie.start()
        print("Start ROS Models Thread (#1150)")
        t.start()

    def rosService_Poppy(self):
        isInit = False
        while True:
            try:
                code = os.read(self.pipelineStdin, 3).decode("utf-8")
                print("[#8956] Lectura Tuberia Exitosa : ",code)
            except:
                print('Error Comunicacion Simulador (#1150)')
            
            if code[0] == '#':
                #Remover '#' para verificar codigo
                code = code[1::]
                #Comprar Estado del Simulador
                if int(code) == self.pipeCode['No_ROS'] and not isInit:
                    isInit = True
                    print('Inicializando Servicios ROS (#1150)')
                elif int(code) == self.pipeCode['Error']:
                    self.load.setPixmap(QtGui.QPixmap("img/fail.png"))
                    self.load_text.hide()
                    self.Background_Load.hide()
                    print("Error with ROS Execute (#1150)")
                    break
                elif int(code) == self.pipeCode['Disp'] and isInit:
                    print('Servicios ROS Inicializados Correctamente (#1150)')
                    break
            else:
                print('Error Codigo Comunicacion (#1150)')
    
        self.Start_Pose.setEnabled(True)
        self.Background_Load.hide()
        self.load.hide()
        self.load_text.hide()
        #Ejecutar Movimiento Inicial
        self.Start_rosPoppy_Availability(Movement=0)
        print("Stop ROS Models Thread(#1150)")

    def Start_rosPoppy_Availability(self, Movement):
        #Tuberias Ejecutar Movimiento
        code = '#'+'{0:0=2d}'.format(Movement)
        os.write(self.pipelineStdout, str.encode(code))

        #Hilo de verificacion disponibilidad de ROS
        dispThread = threading.Thread(target=self.rosPoppy_Availability, name='Daemon')
        dispThread.setDaemon(True)
        print("Start ROS Poppy Availability Thread (#4152)")
        dispThread.start()

        return dispThread

    def rosPoppy_Availability(self):
        #Poppy_Sim.loadModels_Poppy()

        #Iniciar Indicador de Movimiento
        self.movie.start()
        self.Background_Load.show()
        self.load.show()
        self.load_text.setText('Ejecutando Movimiento')
        self.load_text.show()

        #Movimiento Simulador
        while True:
            try:
                state = os.read(self.pipelineStdin, 3).decode("utf-8")
            except:
                print('[#8459] Error en Comunicacion - Aplicacion')

            if state[0] == '#':
                state = state[1::]
                #Comprar Estado del Simulador
                if int(state) == self.pipeCode['Ex_Move']:
                    print('Servicio Ejecutando Movimiento (#4152)')
                elif int(state) == self.pipeCode['Disp']:
                    print('Servicios ROS Disponibles (#4152)')
                    break
            else:
                print('Error Codigo Comunicacion - Estado (#4152)')

        #Terminar Indicador de Movimiento
        self.load.hide()
        self.load_text.hide()
        self.Background_Load.hide()

        print("Stop ROS Poppy Availability Thread (#4152)")


def uiExecute_Pipeline(inPipeline,outPipeline,managerThread):

    app = QApplication(sys.argv)
    ui = mainwindow(inPipeline, outPipeline, managerThread)
    ui.setWindowTitle('Classification Real Time')
    ui.show()
    sys.exit(app.exec_())
