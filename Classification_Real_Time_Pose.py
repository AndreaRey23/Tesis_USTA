import sys
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QFileDialog, QWidget, QMessageBox, QTableWidget, QTableWidgetItem
from PyQt5.uic import loadUi
from PyQt5.QtCore import QTimer, QRegExp, Qt, QEvent, QRect
from PyQt5.QtGui import QPixmap, QImage, QRegExpValidator, QMovie
from PyQt5 import QtGui
import numpy as np
import cv2
import threading
import logging
import time
import tensorflow as tf
from collections import deque
from joblib import dump, load

import Pose_Real_Time_Prediccion as pose
import Processing_Data as data
import Prediction_Models as models
import Poppy_Predict_Class as poppy

from datetime import datetime

import os


COUNTDOWN_TIME = 3
cTime = COUNTDOWN_TIME

###Clases de archivos importados###
PoseEstimation = pose.Pose()
Data_process = data.getData()
Predict_Model = models.Models_P()
Poppy_Sim = poppy.simPoppy()

class mainwindow(QMainWindow):
	def __init__(self):
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
		#self.load.setGeometry(QRect(0, 0, 841, 511))
		self.load.setText("")
		self.movie = QMovie('img/loading.gif')
		self.load.setMovie(self.movie)
		self.Background_Load.hide()

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
		self.Thread_Move = threading.Thread()

	def live_video(self,camera_port=0):
		#Poppy_Sim.loadModels_Poppy()
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
								if Predict_Model.repeatPredict(prediccion) and not self.Thread_Move.is_alive():
									self.Move.setText(Data_process.classes[prediccion])
									#Iniciar hilo Movimiento
									self.Thread_Move = self.Start_Thread_rosPoppy_Movement(prediccion)
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
		box.move(ui.pos().x()+ui.width()/3,ui.pos().y()+ui.height()/2)
		box.exec_()

		if box.clickedButton() == buttonAccept:
			self.Stop_Thread_Real_Time()
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
		t.start()
		print("Start ROS Models Thread(#1150)")

	def rosService_Poppy(self):
		try: 
			#Poppy_Sim.loadModels_Poppy()
			Poppy_Sim.init_RosService()
			self.Start_Pose.setEnabled(True)
			self.Background_Load.hide()
			self.load.hide()
			self.load_text.hide()
		except:
			self.load.setPixmap(QtGui.QPixmap("img/fail.png"))
			self.load_text.hide()
			print("Error with ROS Execute")
		print("Stop ROS Models Thread(#1150)")

	def Start_Thread_rosPoppy_Movement(self,Movement):
		#Hilo de servicios de ROS
		moveThread = threading.Thread(target=self.rosPoppy_Movement, name='Daemon',args=(Movement,))
		moveThread.setDaemon(True)
		print("Start ROS Poppy Movement Thread(#2105)")
		moveThread.start()

		return moveThread

	def rosPoppy_Movement(self,Prediction):
		#Poppy_Sim.loadModels_Poppy()

		#Iniciar Indicador de Movimiento
		self.movie.start()
		self.Background_Load.show()
		self.load.show()
		self.load_text.setText('Ejecutando Movimiento')
		self.load_text.show()

		#MOvimiento Simulador
		predict = Poppy_Sim.getMovement_predict(Prediction)
		Poppy_Sim.poppy_pred(predict,Poppy_Sim.loadModel_Poppy(predict))

		#Terminar Indicador de Movimiento
		self.load.hide()
		self.load_text.hide()
		self.Background_Load.hide()

		print("Stop ROS Poppy Movement Thread(#2105)")


if __name__ == "__main__":
	app = QApplication(sys.argv)
	ui = mainwindow()
	ui.setWindowTitle('Classification Real Time')
	ui.show()
	sys.exit(app.exec_())
	
