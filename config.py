FPS = 24
F_0 = 0.5
AUDIO_RATE = 44100
FOURCC= [*'mp4v']

with open('temp/template_path.txt','r') as f:
	last_template=f.read()


BASEDIR='/home/sandhulab/Documents/' #a new folder should be created here for analysis

biosensing_flag='ok' #after uploading videos to a folder, 
					#create a folder named 'ok' to begin analysis

gui_flag=True #set to Flse if your computer does not have a monitor connected to it

timeout=900 #in seconds

delay=5 #time interval at which folders will be checked if running headlessly

stopfile='./service_running.txt' #this file will be created automatically.
#If you want to stop the server, delete this file. This will make sure that
#if some analysis job is running, it is finished successfully before stopping the server