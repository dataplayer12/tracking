FPS = 24
F_0 = 0.5
AUDIO_RATE = 44100
FOURCC= [*'mp4v']
FIND_OSCILLATING=True

with open('temp/template_path.txt','r') as f:
	last_template=f.read()

BASEDIR='/home/sandhulab/Documents/' #a new folder should be created here for analysis

biosensing_flag='ok' #after uploading videos to a folder, 
					#create a folder named 'ok' (case insensitive) to begin analysis

gui_flag=True #set to Flse if your computer does not have a monitor connected to it

timeout=900 #in seconds

NUM_FRAMES_IN_HISTORY=2 #first 2 frames are used to initialize the objects being tracked
MAX_KALMAN_LEARNING_TIME=30 #how much time to allow kalman filter to learn motion model


cropwindow=(1920,1080)
delay=5 #time interval at which folders will be checked

stopfile='./service_running.txt' #this file will be created automatically.
#If you want to stop the server, delete this file. This will make sure that
#if some analysis job is running, it is finished successfully before stopping the server