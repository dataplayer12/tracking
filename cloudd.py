import os
import tutils
import config as cfg #import BASEDIR, biosensing_flag, gui_flag, timeout, delay, stopfile
from multiprocessing import Process, cpu_count
import time

class ServiceMonitor(object):
    def __init__(self, monitor_d):
        self.monitor_d=monitor_d
        self.ignored=self.index_existing_dirs()
        self.candidates=[]
        self.stopfile=cfg.stopfile

        with open(self.stopfile,'w') as f:
            f.write('Delete this file to stop analysis server')

    def run(self):
        while os.path.exists(self.stopfile):
            self.refresh_candidates()
            dirs_to_analyze=self.analyze_or_ignore()
            for d in dirs_to_analyze:
                self.analyze_biosensing_folder(d)
            time.sleep(cfg.delay)

    def index_existing_dirs(self):
        return [f for f in os.listdir(self.monitor_d) if os.path.isdir(self.monitor_d)]

    def refresh_candidates(self):
        alldirs=self.index_existing_dirs()
        allcandidates= [f for f in alldirs if f not in self.ignored]
        now=time.time()
        oldcandidates=[f[1] for f in self.candidates]
        newcandidates=[(now,f) for f in allcandidates if f not in oldcandidates]
        if len(newcandidates):
            self.candidates.extend(newcandidates)

    def analyze_or_ignore(self):
        dirs_to_analyze=[]
        remove_candidates=[]
        for idx,(first_found,d) in enumerate(self.candidates):
            if (time.time()-first_found) > cfg.timeout:
                self.ignored.append(d)
                remove_candidates.append(idx)
                if cfg.biosensing_flag in [f.lower() for f in os.listdir(os.path.join(self.monitor_d,d)) if os.path.isdir(os.path.join(self.monitor_d,d,f))]: #one last check
                    dirs_to_analyze.append(os.path.join(self.monitor_d,d))

            else:
                if cfg.biosensing_flag in [f.lower() for f in os.listdir(os.path.join(self.monitor_d,d)) if os.path.isdir(os.path.join(self.monitor_d,d,f))]:
                    dirs_to_analyze.append(os.path.join(self.monitor_d,d)) #add to list of folders to analyze
                    self.ignored.append(d) #don't check this folder again, as we have already listed it for analysis
                    remove_candidates.append(idx) #remove from candidates

        for idx in remove_candidates: #if something was found, remove it from candidates
            self.candidates.pop(idx)

        return dirs_to_analyze

    def analyze_biosensing_folder(self,folder):
        trimmed_videos=[]
        outfiles = tutils.extract_videos_for_processing(folder+"/")
        trimmed_videos.extend(outfiles)
        if os.path.exists(os.path.join(folder,'temp1.jpg')):
            temp=os.path.join(folder,'temp1.jpg')
            self.store_template()
        else:
            temp = self.get_default_template()

        th = 0.8

        max_procs=cpu_count()-1
        total_n_videos=len(trimmed_videos)
        for run in range(1+total_n_videos//max_procs):
            processes=[]
            for pidx in range(min(max_procs,total_n_videos-run*max_procs)):
                vid=trimmed_videos[max_procs*run+pidx]
                processes.append(Process(target=tutils.track_video,\
                    args=(vid,temp,th,cfg.gui_flag)))
                processes[-1].start()
            
            [p.join() for p in processes]

    def store_template(self,path):
        with open('temp/template_path.txt', 'w') as f:
            f.write(path)

    def get_default_template(self):
        with open('temp/template_path.txt', 'r') as f:
            path=f.read()
        return path

if __name__ == '__main__':
    service=ServiceMonitor(cfg.BASEDIR)
    service.run()
    print('Stop notice received. Stopping server...')