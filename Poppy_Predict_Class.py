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

# Librerias para ROS
import rospy
from library_poppy_ros import poppy_control_utils
from poppy_torso_control.srv import *



class simPoppy():
    def __init__(self):
        self.FPS = 10
        #AngulosSimulador
        self.PoppyGoalAngles={'Up':[np.radians(90),np.radians(70),0,0],'Center':[np.radians(-90),np.radians(80),np.radians(-35),np.radians(120)],'Sides':[0,0,0,0],'Front':[0,math.radians(90),0,0],'Down':[np.radians(-90),np.radians(90),0,0]}
        #Movimientos_Prediccion
        self.moves = {'Up':[3,5,8,19],'Down':[0,4,6,9],'Sides':[2,16,17,18],'Front':[1,7,14,15],'Center':[10,11,12,13]}
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.7
        set_session(tf.Session(config=config))


    def loadModel_Poppy(self,Movement):
        #Load Models
        print("Esperando Cargar Modelo Poppy")
        model = load_model('model/Simulator/modelo_'+Movement.lower()+'_1024_big.h5')
        print("Modelos de Poppy Cargado")
        return model

    def init_RosService(self):
        # ROS
        print("Esperando servicios de ROS")

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
        print("Servicios de ROS listos")

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

    def poppy_pred(self,Move,model):
        print('(#RPo2) El movimiento a ejecutar es: ',Move)
        test_input=self.pos_init()
        if test_input == []:
            test_input =[0,0,0,0]
        test_input = test_input.reshape((1,4,1))
        test_output=model.predict(test_input)
        total_dist_euc=0
        dist_euc=((test_output[0][:,-1]-self.PoppyGoalAngles[Move])**2)
        for i in dist_euc:
            total_dist_euc=total_dist_euc +i
        total_dist_euc=total_dist_euc/4
        for i in range(18):
            self.goto(test_output[0][:,i])
        print("error por distancia euclidiana: ", total_dist_euc)
        return None

    def getMovement_predict(self, predict):

        for key,values in self.moves.items():
            for value in values:
                if predict == value:
                    return(key)
        return 'Error'
    


if __name__ == "__main__":
    
    poppysim=simPoppy()
    poppysim.init_RosService()

    while True:
        predict = poppysim.getMovement_predict(3)
        model = poppysim.loadModel_Poppy(predict)
        poppysim.poppy_pred(predict, model)



    

