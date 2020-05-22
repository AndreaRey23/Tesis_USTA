import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report
from joblib import dump, load
from progress.bar import Bar
import collections


class Pose():
    def __init__(self):
        #Inicializacion Parametros Pose
        self.n_art = 15
        self.POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14]]
        self.inWidth = 368
        self.inHeight = 368
        self.threshold = 0.1
        self.f = 30 #numero de frames
        self.new_time = 2.2 #Tiempo a interpolar
        self.frame_s=int(self.f*self.new_time) #Nuevo numero frame
        self.xy_factor = 66.0/182.0
        self.z_factor = 45.0/74.67
        #Carga del modelo del Pose
        self.graph = self.load_graph('model/Pose/pose.pb')
        self.net_input = self.graph.get_tensor_by_name('prefix/image:0')
        self.net_output = self.graph.get_tensor_by_name('prefix/concat_stage7:0')
        #Inicializar Sesion 
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.633)
        self.sess = tf.compat.v1.Session(graph=self.graph,config=tf.ConfigProto(gpu_options=gpu_options))

    def load_graph(self,frozen_graph_filename):
        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we can use again a convenient built-in function to import a graph_def into the
        # current default Graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="prefix",
                op_dict=None,
                producer_op_list=None
            )
        return graph

    def PoseFrame(self,frame):
        #matrix to save x,y and z points
        points_xyz = np.empty([0,self.n_art,2])
        if len(frame)<= 0:
            print("Descartando Frame...")
            return []
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (self.inWidth, self.inHeight),
                                  (0, 0, 0), swapRB=False, crop=False)
        inpBlob = np.swapaxes(inpBlob,1,3)
        inpBlob = np.swapaxes(inpBlob,1,2)
        output = self.sess.run(self.net_output, feed_dict={ self.net_input: inpBlob })
        output = np.swapaxes(output,1,3)
        output = np.swapaxes(output,2,3)
        H = output.shape[2]
        W = output.shape[3]
        # Empty list to store the detected keypoints
        points = []
        
        points_xyz_thisframe = np.empty([15,2])
        points_xyz_thisframe[:] = np.nan

        for i in range(self.n_art):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]
            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            # Scale the point to fit on the original image
            x = ((frameWidth * point[0]) / W)
            y = ((frameHeight * point[1]) / H)
            #take the point x, y in z and round the value of 3 pixels and do the mean
            if prob > self.threshold :
                points_xyz_thisframe[i] = [x, y]
                points.append((int(x), int(y)))
                    # Add the point to the list if the probability is greater than the threshold
            else :
                points.append(None)
                 # Draw Skeleton
        for pair in self.POSE_PAIRS:
            if points[pair[0]] and points[pair[1]]:
                cv2.line(frame, points[pair[0]], points[pair[1]], (0, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[pair[0]], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[pair[1]], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        points_xyz = np.append(points_xyz, [points_xyz_thisframe], axis=0)
        return points_xyz

   