import tkinter as tk
from multiprocessing import Process, cpu_count, set_start_method
import os
import config as cfg
import tutils
import tkinter.filedialog as filed


def app():
    window = tk.Tk()

    window.title("Tracking Software")
    window.geometry("750x800")
    window.resizable(width="true", height="true")

    # three frames on top of each other
    frame_header = tk.Frame(window, borderwidth=2, pady=2)
    frame_header.grid(row=0, column=0)

    center_frame = tk.Frame(window, borderwidth=5, pady=5)
    center_frame.grid(row=1, column=0)

    bottom_frame = tk.Frame(window, borderwidth=2, pady=5)
    bottom_frame.grid(row=5, column=0)

    footer_frame = tk.Frame(window, borderwidth=2, pady=5)
    footer_frame.grid(row=7, column=0)

    mainframe = tk.Frame(window, borderwidth=5, pady=5)
    mainframe.grid(row=4, column=0)  # sticky=(N,W,E,S)

    def add_dropdown1(var, frame, choices, heading, callback, labelrow, row):
        var.set(choices[0])  # set the default option
        popupMenu = tk.OptionMenu(frame, var, *choices)
        tk.Label(frame, text=heading).grid(row=labelrow, column=1)
        popupMenu.grid(row=row, column=1)
        var.trace('w', callback)

    # add_dropdown1(tkvar1, mainframe, production_houses,
    #             "Production House: ", change_dropdown1, 1, 2)

    # label header to be placed in the frame_header
    header = tk.Label(frame_header, text="Track Master 6000", bg='black',
                      fg='white', height='3', width='50', font=("Helvetica 16 bold"))
    header.grid(row=0, column=0)
    footer = tk.Label(footer_frame, text='', bg='black',
                      fg='white', height='3', width='50', font=("Helvetica 16 bold"))
    footer.grid(row=2, column=0)

    def get_folder(*args):
        filename = filed.askopenfilename(
            initialdir=cfg.BASEDIR, title='Please select the video')
        variables['folder'].set(filename)
        entries['folder']['text'] = filename

    def get_template(*args):
        filename = filed.askopenfilename(
            title='Please select the template')
        variables['template'].set(filename)
        entries['template']['text'] = filename

    frames = {
        'f1': tk.Frame(center_frame, borderwidth=2, relief='raised'),
        'f2': tk.Frame(center_frame, borderwidth=2, relief='raised'),
        'f3': tk.Frame(center_frame, borderwidth=2, relief='raised'),
        'f4': tk.Frame(center_frame, borderwidth=2, relief='raised'),
        'f5': tk.Frame(center_frame, borderwidth=2, relief='raised'),
        'f6': tk.Frame(center_frame, borderwidth=2, relief='raised'),
        'f7': tk.Frame(center_frame, borderwidth=2, relief='raised'),
        'f8': tk.Frame(center_frame, borderwidth=2, relief='raised'),
        'f9': tk.Frame(center_frame, borderwidth=2, relief='raised'),
        'f10': tk.Frame(center_frame, borderwidth=2, relief='raised'),
        'f11': tk.Frame(center_frame, borderwidth=2, relief='raised'),
        'f12': tk.Frame(center_frame, borderwidth=2, relief='raised'),
        'f13': tk.Frame(center_frame, borderwidth=2, relief='raised'),
        'f14': tk.Frame(center_frame, borderwidth=2, relief='raised')
    }

    labels = {
        'folder': tk.Label(frames['f1'], text="File: "),
        'template': tk.Label(frames['f2'], text="Template: "),
        'mode': tk.Label(frames['f3'], text="Tracking mode: "),
        'folderjob': tk.Label(frames['f4'], text="Analyze all videos in this folder?"),
        'trim': tk.Label(frames['f5'], text='Trim Video? '),
        'trims': tk.Label(frames['f5'], text='Start time(s): '),
        'trime': tk.Label(frames['f5'], text='End time(s): '),
        'em': tk.Label(frames['f6'], text='Trim at every minute? '),
        'crop': tk.Label(frames['f7'], text='Crop Video? '),
        'threshold': tk.Label(frames['f8'], text='Template matching threshold: '),
        'sb': tk.Label(frames['f9'], text='Analyze stopped beads? '),
        'hm': tk.Label(frames['f10'], text='Analyze harmonic motion?'),
        'parallel': tk.Label(frames['f11'], text='Use parallel processing? '),
        'gpu': tk.Label(frames['f12'], text='Use GPU? '),
        'skip': tk.Label(frames['f13'], text='Subsampling factor: '),
        'proxy': tk.Label(frames['f14'], text='Settings: ')
    }

    # http://effbot.org/tkinterbook/variable.htm
    variables = {
        'folder': tk.StringVar(),
        'template': tk.StringVar(),
        'mode': tk.StringVar(),
        'folderjob': tk.StringVar(),
        'trim': tk.StringVar(),
        'crop': tk.StringVar(),
        'em': tk.StringVar(),
        'trims': tk.IntVar(),
        'trime': tk.IntVar(),
        'threshold': tk.DoubleVar(),
        'sb': tk.StringVar(),
        'hm': tk.StringVar(),
        'parallel': tk.StringVar(),
        'gpu': tk.StringVar(),
        'skip': tk.DoubleVar(),
        'status': tk.StringVar(),
        'affa': []
    }

    def set_defaults():
        variables['folder'].set('')
        variables['template'].set(cfg.last_template)
        variables['mode'].set('Biosensing')
        variables['folderjob'].set('No')
        variables['trim'].set('No')
        variables['trims'].set(50)
        variables['trime'].set(60)
        variables['crop'].set('No')
        variables['em'].set('Yes')
        variables['hm'].set('Yes')
        variables['sb'].set('Yes')
        variables['threshold'].set(0.8)
        variables['skip'].set(1.0)
        variables['parallel'].set('No')
        variables['gpu'].set('No')
        variables['affa'] = []

    template_text = 'Browse...' if not cfg.last_template else cfg.last_template

    entries = {
        'folder': tk.Button(frames['f1'], text='Browse...', command=get_folder, bg='white',
                            fg='black', relief='raised', width=50, height=2, font=('Helvetica 9 bold')),
        'template': tk.Button(frames['f2'], text=template_text, command=get_template, bg='white',
                              fg='black', relief='raised', width=50, height=2, font=('Helvetica 9 bold')),
        'mode': tk.OptionMenu(frames['f3'], variables['mode'], *['Biosensing', 'TEM']),
        'folderjob': tk.OptionMenu(frames['f4'], variables['folderjob'], *['Yes', 'No']),
        'trim': tk.OptionMenu(frames['f5'], variables['trim'], *['Yes', 'No']),
        'trims': tk.Entry(frames['f5'], textvariable=variables['trims'], width=4),
        'trime': tk.Entry(frames['f5'], textvariable=variables['trime'], width=4),
        'em': tk.OptionMenu(frames['f6'], variables['em'], *['Yes', 'No']),
        'crop': tk.OptionMenu(frames['f7'], variables['crop'], *['Yes', 'No']),
        'threshold': tk.Entry(frames['f8'], textvariable=variables['threshold'], width=4),
        'sb': tk.OptionMenu(frames['f9'], variables['sb'], *['Yes', 'No']),
        'hm': tk.OptionMenu(frames['f10'], variables['hm'], *['Yes', 'No']),
        'parallel': tk.OptionMenu(frames['f11'], variables['parallel'], *['Yes', 'No']),
        'gpu': tk.OptionMenu(frames['f12'], variables['gpu'], *['Yes', 'No']),
        'skip': tk.Entry(frames['f13'], textvariable=variables['skip'], width=4),
        'proxy': tk.Button(frames['f14'], text='Change settings', command=settings_app, bg='white',
                            fg='black', relief='raised', width=30, height=2, font=('Helvetica 9 bold')),
    }

    def store_template(*args):
        with open('temp/template_path.txt', 'w') as f:
            f.write(variables['template'].get())

    def change_mode(*args):
        if variables['mode'].get()=='Biosensing':
            variables['status'].set('Mode set to Biosensing')
            variables['folderjob'].set('Yes')
            variables['trim'].set('Yes')
            variables['trims'].set(50)
            variables['trime'].set(60)
            variables['crop'].set('No')
            variables['em'].set('Yes')
            variables['hm'].set('Yes')
            variables['sb'].set('Yes')
            variables['threshold'].set(0.8)
            variables['parallel'].set('Yes')
            variables['gpu'].set('Yes')
            variables['skip'].set(1.0)

        elif variables['mode'].get()=='TEM':
            variables['status'].set('Mode set to TEM')
            variables['folderjob'].set('No')
            variables['trim'].set('No')
            variables['trims'].set(50)
            variables['trime'].set(60)
            variables['crop'].set('No')
            variables['em'].set('No')
            variables['hm'].set('No')
            variables['sb'].set('No')
            variables['threshold'].set(0.6)
            variables['parallel'].set('No')
            variables['gpu'].set('No')
            variables['skip'].set(1.0)
            

    footer.configure(textvariable=variables['status'])
    variables['template'].trace('w', store_template)
    variables['mode'].trace("w", change_mode)
    set_defaults()
    change_mode()

    # pack UI elements
    for k in labels.keys():
        labels[k].pack(side='left', padx=1)
        # for k in entries.keys():
        entries[k].pack(side='left', padx=1)

    for f in frames.values():
        f.pack(fill='x', pady=2)

    def close_app():
        window.destroy()

    isyes = lambda v: variables[v].get() == 'Yes'

    def try_create(f): os.mkdir(f) if not os.path.exists(f) else None

    def trim_crop_video_fn():
        trimmed_videos = []
        folder = variables['folder'].get()
        extension = folder[folder.rfind('.'):]

        def trim_helper(folder, em, filemode,variables):
            if em:
                outfiles = tutils.extract_videos_for_processing(
                    folder, True, filemode, guivar=[variables['status'], window])

                trimmed_videos.extend(outfiles)

            else:
                files = [folder] if filemode else [
                    folder + f for f in os.listdir(folder) if f.endswith(extension)]
                print(files)
                for idx, f in enumerate(files):
                    srcfile = f[f.rfind('/') + 1:]
                    parent_dir=f[:f.rfind('/') + 1]
                    analysis_subfolder = parent_dir + \
                        'tracking/{}/'.format(srcfile[:srcfile.rfind(".")])
                    try_create(analysis_subfolder)
                    outfile = analysis_subfolder+srcfile[:srcfile.rfind(
                        '.')] + '_trim' + extension
                    source=tutils.cv2.VideoCapture(f)
                    tutils.trim_video(source, outfile, variables[
                                      'trims'].get(), variables['trime'].get())
                    trimmed_videos.append(outfile)
                    tutils.extract_template_frames(outfile, name='temp1.jpg')
                    if isyes("crop"):
                        newfile = tutils.crop_and_trim(f)
                        trimmed_videos.append(newfile)
                    variables['status'].set(
                        'Trimmed video {} of {}'.format(idx + 1, len(files)))
                    source.release()

        if isyes('trim'):
            if isyes('folderjob'):
                folder = folder[:folder.rfind('/') + 1]
                try_create(folder + 'tracking/')
                trim_helper(folder, isyes('em'), False,variables)
            else:
                try_create(folder[:folder.rfind('/') + 1] + 'tracking/')
                trim_helper(folder, isyes('em'), True,variables)

            if isyes('crop'):
                prev_points = []
                for idx,f in enumerate(trimmed_videos):
                    cropped, prev_points = tutils.crop_and_trim(f, prev_points)
                    variables['affa'].append(cropped)
                    variables['status'].set(
                        'Cropped video {} of {}'.format(idx + 1, len(trimmed_videos)))
            else:
                variables['affa'] = trimmed_videos[:]

            variables['status'].set('Trimming and cropping finished')

        elif isyes('crop'):
            # no trim, only crop
            if isyes('folderjob'):
                folder = folder[:folder.rfind('/') + 1]
                try_create(folder + 'tracking/')
                trimmed_videos = os.listdir(folder)
                prev_points = []
                for idx, f in enumerate(trimmed_videos):
                    cropped, prev_points = tutils.crop_and_trim(f, prev_points)
                    variables['affa'].append(cropped)
                    variables['status'].set(
                        'Cropped video {} of {}'.format(idx + 1, len(trimmed_videos)))
                    window.update_idletasks()
            else:
                cropped, _ = tutils.crop_and_trim(folder)
                variables['affa'].append(cropped)
                variables['status'].set('Cropped one video')
                # window.update_idletasks()
        else:
            variables['status'].set('You have not enabled trimming')

    def run_app():
        if not variables['affa'] and variables['folder'].get():
            variables['affa'].append(variables['folder'].get())

        temp = variables['template'].get()
        th = variables['threshold'].get()
        skip = int(variables['skip'].get())
        hm = isyes('hm')

        if isyes('parallel'):
            max_procs=cpu_count()-1
            total_n_videos=len(variables['affa'])
            for run in range(1+total_n_videos//max_procs):
                processes=[]
                for pidx in range(min(max_procs,total_n_videos-run*max_procs)):
                    vid=variables['affa'][max_procs*run+pidx]
                    processes.append(Process(target=tutils.track_video,\
                        args=(vid,temp,th,cfg.gui_flag,skip,hm)))
                    processes[-1].start()
                
                [p.join() for p in processes]

        else:
            for idx, vid in enumerate(variables['affa']):
                vidn = [idx + 1, len(variables['affa'])]
                tutils.track_video(vid, temp, th,cfg.gui_flag,skip,hm)

        if isyes('sb'):
            bead_radius=tutils.cv2.imread(temp,1).shape[0]//2

            if len(variables['affa'])==1:
                variables['status'].set('Analyzing video for stopped beads')
                src=tutils.cv2.VideoCapture(variables['affa'][0])
                total_frames=src.get(tutils.cv2.CAP_PROP_FRAME_COUNT)
                assert total_frames>0,'source file cannot be read'
                src.release()
                tutils.analyze_sensing_area([variables['affa'][0]],bead_radius,total_frames)
            else:
                if isyes('em'):
                    chunks=tutils.filename_chunks(variables['affa'])
                    for ix,c in enumerate(chunks):
                        variables['status'].set('Analyzing video {}/{} for stopped beads'.format(ix+1,len(chunks)))
                        src=tutils.cv2.VideoCapture(c[0])
                        total_frames=src.get(tutils.cv2.CAP_PROP_FRAME_COUNT)
                        assert total_frames>0,'source file cannot be read'
                        src.release()
                        tutils.analyze_sensing_area(c,bead_radius,total_frames)

        variables['status'].set('Finished tracking')

    def refresh_app():
        set_defaults()

    # a proper app needs some buttons too!
    button_trim = tk.Button(bottom_frame, text="Trim & Crop", command=trim_crop_video_fn,
                            bg='dark blue', fg='black', relief='raised', width=20, height=2, font=('Helvetica 12'))
    button_trim.grid(column=0, row=0, sticky='e', padx=100, pady=2)

    button_refresh = tk.Button(bottom_frame, text="Refresh", command=refresh_app,
                               bg='dark green', fg='black', relief='raised', width=20, height=2, font=('Helvetica 12'))
    button_refresh.grid(column=1, row=0, sticky='e', padx=100, pady=2)

    button_run = tk.Button(bottom_frame, text="Track video", command=run_app, bg='dark green',
                           fg='black', relief='raised', width=20, height=2, font=('Helvetica 12 bold'))
    button_run.grid(column=0, row=1, sticky='w', padx=100, pady=2)

    button_close = tk.Button(bottom_frame, text="Exit", command=close_app,
                             bg='dark red', fg='black', relief='raised', width=20, height=2, font=('Helvetica 12'))
    button_close.grid(column=1, row=1, sticky='e', padx=100, pady=2)

    window.mainloop()

def settings_app():
    settings = tk.Tk()
    settings.title("Settings")
    settings.geometry("700x600")
    settings.resizable(width="true", height="true")

    center_frame = tk.Frame(settings, borderwidth=5, pady=5)
    center_frame.grid(row=1, column=0)

    bottom_frame = tk.Frame(settings, borderwidth=2, pady=5)
    bottom_frame.grid(row=5, column=0)

    mainframe = tk.Frame(settings, borderwidth=5, pady=5)
    mainframe.grid(row=4, column=0)  # sticky=(N,W,E,S)

    frames = {'f{}'.format(idx): tk.Frame(center_frame, borderwidth=2, relief='raised') for idx in range(1,14)}

    labels = {
        'fps': tk.Label(frames['f1'], text="Frames per second: "),
        'f_0': tk.Label(frames['f2'], text="Frequency of oscillation (Hz): "),
        'ar': tk.Label(frames['f3'], text="Audio rate: "),
        'fourcc': tk.Label(frames['f4'], text="Four CC for writing videos: "),
        'fo': tk.Label(frames['f5'], text='Find oscillating beads? '),
        'basedir': tk.Label(frames['f6'], text='Base directory: '),
        'bf': tk.Label(frames['f7'], text='Folder to initiate cloud processing (flag): '),
        'gui': tk.Label(frames['f8'], text='Show GUI in cloud mode? '),
        'tmo': tk.Label(frames['f9'], text='Maximum time spent waiting for flag (min.)? '),
        'nfih': tk.Label(frames['f10'], text='Number of frames in history: '),
        'mklt': tk.Label(frames['f11'], text='Maximum learning steps for Kalman filter: '),
        'cww': tk.Label(frames['f12'], text='Cropping window: Width= '),
        'cwh': tk.Label(frames['f12'], text='Height= '),
        'delay': tk.Label(frames['f13'], text='Time interval at which basedir is checked (s): '),
    }

    setting_vars = {
        'fps': tk.DoubleVar(),
        'f_0': tk.DoubleVar(),
        'ar': tk.IntVar(),
        'fourcc': tk.StringVar(),
        'fo': tk.StringVar(),
        'basedir': tk.StringVar(),
        'bf': tk.StringVar(),
        'gui': tk.StringVar(),
        'tmo': tk.IntVar(),
        'nfih': tk.IntVar(),
        'mklt': tk.IntVar(),
        'cww': tk.IntVar(),
        'cwh': tk.IntVar(),
        'delay': tk.IntVar()
    }

    def get_basedir(*args):
        filename = filed.askdirectory(
            initialdir=cfg.BASEDIR, title='Please select the folder')
        setting_vars['basedir'].set(filename)
        entries['basedir']['text'] = filename

    default_vars = {'fps':24.0, 'f_0':0.5, 'ar':44100, 'fourcc':'mp4v',
        'fo':'Yes', 'basedir':'/Users/Jaiyam/Documents/', 'bf':'ok',
        'gui':'Yes', 'tmo':15, 'nfih':2, 'mklt':30, 'cww':1920,
        'cwh':1080, 'delay':5 }

    entries = {
        'fps': tk.Entry(frames['f1'], textvariable=setting_vars['fps'], width=8),
        'f_0': tk.Entry(frames['f2'], textvariable=setting_vars['f_0'], width=8),
        'ar': tk.Entry(frames['f3'], textvariable=setting_vars['ar'], width=8),
        'fourcc': tk.Entry(frames['f4'], textvariable=setting_vars['fourcc'], width=8),
        'fo': tk.OptionMenu(frames['f5'], setting_vars['fo'], *['Yes', 'No']),
        'basedir': tk.Button(frames['f6'], text=default_vars['basedir'], command=get_basedir, bg='white',
                            fg='black', relief='raised', width=30, height=2, font=('Helvetica 9 bold')),
        'bf': tk.Entry(frames['f7'], textvariable=setting_vars['bf'], width=4),
        'gui': tk.OptionMenu(frames['f8'], setting_vars['gui'], *['Yes', 'No']),
        'tmo': tk.Entry(frames['f9'], textvariable=setting_vars['tmo'], width=4),
        'nfih': tk.Entry(frames['f10'], textvariable=setting_vars['nfih'], width=4),
        'mklt': tk.Entry(frames['f11'], textvariable=setting_vars['mklt'], width=4),
        'cww': tk.Entry(frames['f12'], textvariable=setting_vars['cww'], width=4),
        'cwh': tk.Entry(frames['f12'], textvariable=setting_vars['cwh'], width=4),
        'delay': tk.Entry(frames['f13'], textvariable=setting_vars['delay'], width=4),
    }

    def restore_defaults(*args):
        for v in setting_vars.keys():
            setting_vars[v].set(default_vars[v])
            entries[v]['text']=default_vars[v]
            print('{}: {}'.format(v,default_vars[v]))

    def save_settings(*args):
        for v in setting_vars.keys():
            print('{}: {}'.format(v,setting_vars[v].get()))

    for k in labels.keys():
        labels[k].pack(side='left', padx=1)
        entries[k].pack(side='left', padx=1)

    for f in frames.values():
        f.pack(fill='x', pady=2)

    defaults = tk.Button(bottom_frame, text="Restore Defaults", command=restore_defaults,
                            bg='dark blue', fg='black', relief='raised', width=20, height=2, font=('Helvetica 12'))
    defaults.grid(column=0, row=0, sticky='e', padx=100, pady=2)

    save_button = tk.Button(bottom_frame, text="Save Settings", command=save_settings,
                               bg='dark green', fg='black', relief='raised', width=20, height=2, font=('Helvetica 12'))
    save_button.grid(column=1, row=0, sticky='e', padx=100, pady=2)

    settings.mainloop()

if __name__ == '__main__':
    set_start_method('spawn')
    app()
