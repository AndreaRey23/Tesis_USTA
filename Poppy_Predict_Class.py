# Librerias
import os
import sys
import math
import tensorflow as tf
from keras.models import Sequential
from tensorflow.python.keras.models import load_model
from keras.layers import LSTM, Dense,Dropout
from keras.backend.tensorflow_backend import set_session
import numpy as np
import time
from datetime import datetime
import random

# Librerias para ROS
import rospy
from library_poppy_ros import poppy_control_utils
from poppy_torso_control.srv import *



class simPoppy():
    def __init__(self):
        self.FPS = 120
        #AngulosSimulador
        self.PoppyGoalAngles={'Up':[np.radians(90),np.radians(70),0,0],'Center':[np.radians(-90),np.radians(80),np.radians(-35),np.radians(120)],'Sides':[0,0,0,0],'Front':[0,math.radians(90),0,0],'Down':[np.radians(-90),np.radians(90),0,0]}
        #Movimientos_Prediccion
        self.moves = {'Up':[3,5,8,19],'Down':[0,4,6,9],'Sides':[2,16,17,18],'Front':[1,7,14,15],'Center':[10,11,12,13]}
        #Modelos_movimientos
        self.Models = {}
        #Config Maximum Memory GPU
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        set_session(tf.Session(config=config))
        
        
    def loadModels_Poppy(self):
        #Load Models
        print("[#ROS23] Esperando Cargar Modelos Poppy")
        self.Models['Up'] = load_model('model/Simulator/modelo_up_1024_big.h5')
        #self.Models['Up']._make_predict_function()
        self.Models['Down'] = load_model('model/Simulator/modelo_down_1024_big.h5')
        self.Models['Center'] = load_model('model/Simulator/modelo_center_1024_big.h5')
        self.Models['Sides'] = load_model('model/Simulator/modelo_sides_1024_big.h5')
        self.Models['Front'] = load_model('model/Simulator//modelo_front_1024_big.h5')
        print("[#ROS23] Modelos de Poppy Cargados")

    def loadModel_Poppy(self,Movement):
        #Load Models
        print("Esperando Cargar Modelos Poppy")
        self.Models[Movement] = load_model('model/Simulator/modelo_up_1024_big_2.h5')
        print("Modelos de Poppy Cargados")

    def init_RosService(self):
        # ROS
        print("[#ROS11] Esperando servicios de ROS")

        # Espera hasta que los servicios de ROS esten corriendo
        rospy.wait_for_service('/poppy_plan_movement')
        rospy.wait_for_service('/poppy_offset_movement')
        rospy.wait_for_service('/poppy_forward_kinematics')
        rospy.wait_for_service('/poppy_collision_distance')
        
        # Enruta las conexiones para realizar los llamados
        self.plan_movement = rospy.ServiceProxy('/poppy_plan_movement', PlanMovement)
        self.offset_movement = rospy.ServiceProxy('/poppy_offset_movement', OffsetMovement)
        self.forward_kinematics = rospy.ServiceProxy('/poppy_forward_kinematics', ForwardKinematics)
        self.collision_distance = rospy.ServiceProxy('/poppy_collision_distance', CollisionDistance)
        print("[#ROS11] Servicios de ROS listos")

    def pos_init(self):
        global current_angles, current_pos
        resp_plan_0 = self.plan_movement('r_arm',False,True,[],True,[],False,False,False,self.FPS)
        if resp_plan_0.error==0:
            while not resp_plan_0.plans:
                resp_plan_0 = self.plan_movement('r_arm',False,True,[],True,[],False,False,True,self.FPS)
            start= np.array(resp_plan_0.start_pos)
            return (start)
        print("Invalid angles")
        return None

    def goto(self,pos):
        global current_angles, current_pos
        resp_plan_0 = self.plan_movement('Arms',False,True,[],False,[-pos[0],pos[1],-pos[2],-pos[3],pos[0],pos[1],pos[2],pos[3]],True,True,True,self.FPS)
        if resp_plan_0.error==0:
            resp_plan_0 = self.plan_movement('Arms',False,True,[],False,[-pos[0],pos[1],-pos[2],-pos[3],pos[0],pos[1],pos[2],pos[3]],True,True,True,self.FPS)
            current_angles = np.array(resp_plan_0.target_pos)
            current_pos = np.array(self.forward_kinematics('r_arm', current_angles).end_pos)
            return (current_angles,current_pos)
        print("Invalid angles")
        return None

    def poppy_pred(self, Move, model):
        print('(#RPo2) El movimiento a ejecutar es: ',Move)
        test_input=self.pos_init()
        if test_input == [] :
            test_input =[0,0,0,0]
        elif type(test_input) == type(None):
            test_input = self.pos_init()
        print("(#RPo2) Posicion Inicial Poppy:" ,test_input)
        test_input = test_input.reshape((1,4,1))
        test_output=model.predict(test_input)
        total_dist_euc=0
        dist_euc=((test_output[0][:,-1]-self.PoppyGoalAngles[Move])**2)

        for i in dist_euc:
            total_dist_euc=total_dist_euc +i
        total_dist_euc=total_dist_euc/4
        
        out=[]
        inter_val=6
        for i in range (4):
            fp1=test_output[0][i]
            xp1=np.linspace(0,len(fp1)-1,len(fp1))
            v_interp1= np.linspace(xp1[0],xp1[len(xp1)-1],inter_val)
            v1=np.interp(v_interp1, xp1, fp1)
            out.append(list(v1))
            out_2=np.transpose(out)

        
        for j in range(inter_val):
            self.goto(out_2[::][j])
        print("(#RPo2) Error por Distancia Euclidiana: ", total_dist_euc)
        return None

    def getMovement_predict(self, predict):

        for key,values in self.moves.items():
            for value in values:
                if predict == value:
                    return(key)
        return 'Error'
    

def rosPoppy_Pipeline(parentStdin,parentStdout):

	#Crear Objeto Simulador
	Poppy_Sim = simPoppy()

	#Movimientos_Prediccion
	pipeCode = {'No_ROS':43, 'Ex_Move':23, 'Disp':56, 'End':62, 'Error':37}

	#Tuberias Iniciando Servicio
	code = '#'+'{0:0=2d}'.format(pipeCode['No_ROS'])
	os.write(parentStdout, str.encode(code))

	try:
		#Carganndo Modelos
		Poppy_Sim.loadModels_Poppy()
		#Iniciando Servicio de ROS
		Poppy_Sim.init_RosService()
	except:
		#Tuberias Iniciando Servicio
		code = '#'+'{0:0=2d}'.format(pipeCode['Error'])
		os.write(parentStdout, str.encode(code))
		print('[#7425] Servicio de ROS no Disponible')
		return

	#Tuberias Iniciando Servicio
	code = '#'+'{0:0=2d}'.format(pipeCode['Disp'])
	os.write(parentStdout, str.encode(code))

	while True:
		try:
			move = os.read(parentStdin, 3).decode("utf-8")
		except:
			#Movimiento Default
			move = '#02'
			print('[#1354] Error en Comunicacion - Aplicacion')
			print('[#5487] Restableciendo a Posicion Inicial')

		try:
			
			if(move[0] == '#'):
                #Tuberias Iniciando Servicio
				code = '#'+'{0:0=2d}'.format(pipeCode['Ex_Move'])
				os.write(parentStdout, str.encode(code))
				predict = Poppy_Sim.getMovement_predict(int(move[1::]))
				Poppy_Sim.poppy_pred(predict,Poppy_Sim.Models[predict])
				#Tuberias Liberando Servicio
				code = '#'+'{0:0=2d}'.format(pipeCode['Disp'])
				os.write(parentStdout, str.encode(code))
				print('[#6895] Movimiento Finalizado')
			else:
				print('[#3217] Movimiento Invalido')
		except:
			print('[#0120] Error al Ejecutar Movimiento')
			break

if __name__ == "__main__":
    
    poppysim=simPoppy()
    poppysim.loadModels_Poppy()
    poppysim.init_RosService()

    while True:
        movement = random.randint(0,4)
        movement = (movement, 10)[movement == 4]
        predict = poppysim.getMovement_predict(movement)
        act = datetime.now()
        poppysim.poppy_pred(predict,poppysim.Models[predict])
        print('Tiempo de ejecucion: ',datetime.now()-act)



    

