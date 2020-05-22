Andrea Catalina Rey Ramirez


In this program you can see the pose estimation in a Real time and predict the movement made (Pose estimation and prediction in Real Time).

#REQUERIMENTS
This program was tested on Ubuntu 16 and Ubuntu 18 on a server with a CPU of with GPU of

#INSTALLATION

Use the package manager pip to install the dependencies.

-pip install PyQt5
-pip install numpy
-pip install cv2
-pip install time
-pip install thread
-pip install joblib
-pip install tensorflow
-pip install collections

This program needs the models obtained by the recurrent neural(RNN) network and the SVM
wich are in the folder trainings (get https://drive.google.com/open?id=1nX9qHJezh5lu2EiIORiqdBpgcfwptVwZ).
-RNN: real_many_to_one_nr256_lr0.001_bt15_st500_outSigmoid_dtOrigInv.meta
-SVM: svm_gm0.1_klpoly_c0.5_dg6.joblib

So that the interface can be executed the file is necessary is the file that Qt create in the project:
-QT: mainwindow.ui

Finally for the estimation pose it is necessary to have the model of folder model and 
the file where the pose, interpolation and centralization of the pointswith respect the hip is located.
-MODEL OF POSE: pose.pb (get from https://drive.google.com/open?id=1XeSRrQmSGd281bf14R1XpmngmLoRVw2q).
-FILE POSE: Pose_Real_Time_Prediccion_Class.py


#USER MANUAL:

The user should open terminal and write:

	$ cd /path/where/dowland/is
	$ python Classification_Real_Time_Pose.py

In Classification_Real_Time_Pose.py the interface is found.


