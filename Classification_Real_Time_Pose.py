
import os, sys
import time
import signal



if __name__ == "__main__":

	parentStdin, childStdout  = os.pipe() 
	childStdin,  parentStdout = os.pipe() 
	pid = os.fork()
	
	if pid:
		#Import User Interface
		import guiPose_Poppy as ui

		#Close Pipeline Child
		os.close(childStdout)
		os.close(childStdin)
		#Execute User Interface
		ui.uiExecute_Pipeline(parentStdin,parentStdout,pid)
		

	else:
		#Import Poppy ROS
		import tensorflow as tf
		import Poppy_Predict_Class as poppy

		#Close Pipeline Parent
		os.close(parentStdin)
		os.close(parentStdout)
		#Execute Poppy ROS
		poppy.rosPoppy_Pipeline(childStdin,childStdout)
		#Kill Subprocess
		os.kill(pid, signal.SIGKILL)
		

	
	
