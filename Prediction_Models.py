from joblib import dump, load
import tensorflow as tf
import numpy as np

class Models_P():
    def __init__(self): 
        #Carga de modelo SVM 
        self.svm_model = load('./trainings/svm_gm0.1_klpoly_c0.5_dg5.joblib')
        #Sesion para cargar modelo RNN
        self.sess=tf.Session()
        #Restauracion modelo RNN
        self.new_saver = tf.train.import_meta_graph('trainings/real_many_to_one_nr256_lr0.01_bt1600_st500_outSigmoid_dtOrigInv.meta')
        self.new_saver.restore(self.sess,tf.train.latest_checkpoint('./trainings'))
        self.graph = tf.compat.v1.get_default_graph()
        #Obtener parametros del modelo restaurado
        self.input_vec = self.graph.get_tensor_by_name("input_vec:0")
        self.prediction = self.graph.get_tensor_by_name("prediction:0")
        #Offset SVM
        self.offset_svm=[0.33, 0.58, 0, -0.18, 0.37, 0.2, 0, 0.5, 0, 0, 0.2, 0.38, -0.1, 0, 0.41, -0.2, 0, -0.13, -0.16, 0]
        #Offset RNN        
        self.offset_rnn=[35, 9.1, -0.5, -0.5, 4, -0.4, 9.2, 5, 8, 0, 2, 2, 3, 0, 12, 8, 2, 3, 0, 2]
        
        #Send Last Prediction
        self.num_max_predict = 2
        self.last_predict = len(self.offset_svm)
        self.num_predict = 0


    def SVM_Model(self, batch, threshold = 0.55):
        #Realizar prediccion SVM
        svm_prediction = self.svm_model.predict_proba([batch.ravel()])[0]
        prediccion = np.argmax(svm_prediction)
        #Prediccion con probabilidades
        probaPrediction = svm_prediction[prediccion] + self.offset_svm[prediccion]
        prediccion_norm = ((probaPrediction * 100)**2)/10000
        print("Prediction : ",prediccion,"Probability : ",prediccion_norm,"Umbral",threshold)

        if prediccion_norm > threshold:
            print("Movimiento Detectado...")
        else:
            prediccion = len(svm_prediction)

        return prediccion

    def RNN_Model(self, batch, threshold = 0.9): 
        #Carga de parametros del modelo restaurado
        feed_dict = {self.input_vec: [batch]}
        #Prediccion con probabilidad
        predict_probs = self.sess.run(self.prediction, feed_dict=feed_dict)[0]

        #Probabilidad de SidesUp
        predict_probs[8] =  predict_probs[8] + 9.9*(10**-6)
        predict_probs[6] =  predict_probs[6] - 99.9*(10**-5)
        predict_probs[0] =  predict_probs[0] + 14.9*(10**-5)
        predict_probs[1] =  predict_probs[1] - 9.9*(10**-5)

        #Indice del movimiento con la maxima probabilidad
        idx_prediction= np.argmax(predict_probs)
        #Umbralizacion
        prediction = (predict_probs[idx_prediction]*(10 ** 5))-99900+self.offset_rnn[idx_prediction]

        #Normalizacion prediccion
        if prediction < 0 : prediction = 0
        prediction_norm = ((prediction * 10)**2)/10000

        

        print("Prediction : ",idx_prediction,"Probability : ",prediction,"Umbral",threshold,"Normalization : ",prediction_norm)
        
        if idx_prediction == 5 and  prediction_norm >= (threshold+0.6):
            prediction_norm = 60

        if prediction_norm > threshold:
            print("Movimiento Detectado...")
        else:
            idx_prediction = len(predict_probs)

        
        return idx_prediction


    def repeatPredict(self, current_predict):
        #Number of Movements
        lenMovements = len(self.offset_svm)

        if current_predict == lenMovements:
            self.num_predict = 0
        elif current_predict == self.last_predict:
            self.num_predict = self.num_predict+1
        else:
            self.num_predict = 0
        
        #Update Last Prediction
        self.last_predict = current_predict
        
        if self.num_predict >= self.num_max_predict:
            return True
        return False