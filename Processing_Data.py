import cv2
import time
import numpy as np

datasets = {'Orig': 'dataset/complete_avgtotal_centered_pixels_nodepth.npy',
            'OrigInv': 'dataset/complete_avgtotal_centered_pixels_nodepth-with_inv.npy'}

dataset = 'dataset/complete_avgtotal_centered_pixels_nodepth-with_inv.npy'
class getData():
    def __init__(self):
        #Inicializacion Parametros
        self.mpi_points = [0,1,2,3,4,5,6,7,14] #only shoulder, elbow and wrist
        self.timesteps = 66
        self.num_inputs = len(self.mpi_points)*2
        self.input_size  = self.num_inputs
        self.output_size   = 20
        self.classes = ['CenterDown','CenterFront','CenterSides','CenterUp','FrontDown','FrontUp','SidesDown','SidesFront','SidesUp','UpDown','DownCenter','FrontCenter','SidesCenter','UpCenter','DownFront','UpFront','DownSides','FrontSides','UpSides','DownUp','No Prediction']

    def getOneSampleVideo(self, pointsPose):
        #Inicializacion Parametros
        self.timesteps = len(pointsPose)
        #Arreglo con dimensiones (66,?)
        sample_x = np.empty((self.timesteps, 0))
        for p in self.mpi_points:
            #Toma de muestra del eje x
            x = pointsPose[:,p,0]
            mask = np.isnan(x)
            #Null X Points in Frame
            if not  mask.all() : 
                x[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), x[~mask])
            else : 
                x = np.zeros(len(mask))
                
            #Toma de muestra del eje y
            y = pointsPose[:,p,1]
            mask = np.isnan(y)
            #Null X Points in Frame
            if not  mask.all() : 
                y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])
            else : 
                y = np.zeros(len(mask))

            #Añadir muestra en x_point & y_point, en el arreglo
            sample_x = np.append(sample_x, x.reshape(self.timesteps,1), axis=1)
            sample_x = np.append(sample_x, y.reshape(self.timesteps,1), axis=1)
        return sample_x

    def getPoints(self,frame,offset):
        #Obtener puntos en un frame
        pos_x= [x+offset  for idx,x in enumerate(frame) if idx % 2 == 0]
        pos_y= [y+offset  for idx,y in enumerate(frame) if (idx+1) % 2 == 0]
        return pos_x, pos_y
   
    def DrawPoints(self,img,px,py):
        #inicializacion Parametros
        radius=5
        #Uniones del MPI
        uniones=[[0,1],[1,14],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7]]
        #Dibujar articulaciones
        for points in range(len(px)):
            cv2.circle (img, (int(px[points]),int(py[points])), radius, (0,0,255), 2)
        #Dibujar lineas de union de las articulaciones
        for union in uniones:
            init=self.mpi_points.index(union[0])
            end=self.mpi_points.index(union[1])
            cv2.line (img, (int(px[init]),int(py[init])), (int(px[end]),int(py[end])), (255,0,0), 2)

    def avgCentered(self,points_xyz,centerPoint):
        #Obtencion x & y del punto central
        x = points_xyz[:,centerPoint,0]
        y = points_xyz[:,centerPoint,1]
        #Promedio de puntos con respecto al punto central
        avg = [np.average(x),np.average(y)]
        pointsCentered = points_xyz - avg
        return pointsCentered

    def avgCenterSample(self,points_sample):
        #Obtencion x & y del punto central
        x_center=points_sample[:,-2]
        y_center=points_sample[:,-1]
        #Promedio de puntos con respecto al punto central
        avg = np.array([np.average(x_center),np.average(y_center)])
        points_avg=[]
        for idx in range(int(points_sample.shape[1]/2)):
            points_avg=np.append(points_avg,avg)
        pointsCentered = points_sample-points_avg
        
        return pointsCentered

    def verify_Person(self, points_Frame=[], noPoints = 9):
        #Verificar si los puntos procedentes del pose, no se encuentran completos
        nan_Points = np.isnan(points_Frame)
        nan_Count = 0
        #Contador de puntos vacios
        for point in nan_Points[0]:
            nan_Count = nan_Count + (0, 1)[point[0] or point[1]]
        #El numero de puntos vacios, debe ser menor a 9 (cantidad de puntos del MPI)
        if nan_Count > noPoints:
            return False
        
        return True