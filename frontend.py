import tkinter as tk
import easygui
import pandas as pd
from time import strftime
from time import sleep
import subprocess
import signal
import os
import config as cfg
import tutils


def app(database):
    window = tk.Tk()  # you may also see it named as "root" in other sources

    window.title("Tracking Software")
    window.geometry("650x600")
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
    #mainframe.columnconfigure(0, weight=1)
    #mainframe.rowconfigure(1, weight=1)

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
        filename = tk.filedialog.askopenfilename(
            title='Please select the video')
        variables['folder'].set(filename)
        entries['folder']['text'] = filename

    def get_template(*args):
        filename = tk.filedialog.askopenfilename(
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
        'f8': tk.Frame(center_frame, borderwidth=0, relief='raised'),
        'f9': tk.Frame(center_frame, borderwidth=0, relief='raised'),
        'f10': tk.Frame(center_frame, borderwidth=0, relief='raised')
    }

    labels = {
        'folder': tk.Label(frames['f1'], text="File: "),
        'template': tk.Label(frames['f2'], text="Template: "),
        'folderjob': tk.Label(frames['f3'], text="Analyze all videos in this folder?"),
        'trim': tk.Label(frames['f4'], text='Trim Video? '),
        'trims': tk.Label(frames['f4'], text='Start time(s): '),
        'trime': tk.Label(frames['f4'], text='End time(s): '),
        'em': tk.Label(frames['f5'], text='Trim at every minute? '),
        'crop': tk.Label(frames['f6'], text='Crop Video? '),
        'threshold': tk.Label(frames['f7'], text='Template matching threshold: '),
        'sb': tk.Label(frames['f8'], text='Analyze stopped beads? '),
        'parallel': tk.Label(frames['f9'], text='Use parallel threads? '),
        'gpu': tk.Label(frames['f10'], text='Use GPU? '),
    }

    # http://effbot.org/tkinterbook/variable.htm
    variables = {
        'folder': tk.StringVar(),
        'template': tk.StringVar(),
        'folderjob': tk.StringVar(),
        'trim': tk.StringVar(),
        'crop': tk.StringVar(),
        'em': tk.StringVar(),
        'trims': tk.IntVar(),
        'trime': tk.IntVar(),
        'threshold': tk.DoubleVar(),
        'sb': tk.StringVar(),
        'parallel': tk.StringVar(),
        'gpu': tk.StringVar(),
        'status': tk.StringVar(),
        'affa': []
    }

    def set_defaults():
        variables['folder'].set('')
        variables['template'].set(cfg.last_template)
        variables['folderjob'].set('No')
        variables['trim'].set('No')
        variables['trims'].set(50)
        variables['trime'].set(60)
        variables['crop'].set('No')
        variables['em'].set('Yes')
        variables['sb'].set('Yes')
        variables['threshold'].set(0.8)
        variables['parallel'].set('No')
        variables['gpu'].set('No')
        variables['affa'] = []

    template_text = 'Browse...' if not cfg.last_template else cfg.last_template

    entries = {
        'folder': tk.Button(frames['f1'], text='Browse...', command=get_folder, bg='dark green',
                            fg='black', relief='raised', width=50, height=2, font=('Helvetica 9 bold')),
        'template': tk.Button(frames['f2'], text=template_text, command=get_template, bg='dark green',
                              fg='black', relief='raised', width=50, height=2, font=('Helvetica 9 bold')),
        'folderjob': tk.OptionMenu(frames['f3'], variables['folderjob'], *['Yes', 'No']),
        # tk.Checkbutton(frames['f3'], text="Analyze all videos in this folder",
        #                    variable=variables['folderjob']),
        'trim': tk.OptionMenu(frames['f4'], variables['trim'], *['Yes', 'No']),
        'trims': tk.Entry(frames['f4'], textvariable=variables['trims'], width=4),
        'trime': tk.Entry(frames['f4'], textvariable=variables['trime'], width=4),
        'em': tk.OptionMenu(frames['f5'], variables['em'], *['Yes', 'No']),
        'crop': tk.OptionMenu(frames['f6'], variables['crop'], *['Yes', 'No']),
        'threshold': tk.Entry(frames['f7'], textvariable=variables['threshold'], width=4),
        'sb': tk.OptionMenu(frames['f8'], variables['sb'], *['Yes', 'No']),
        'parallel': tk.OptionMenu(frames['f9'], variables['parallel'], *['Yes', 'No']),
        'gpu': tk.OptionMenu(frames['f10'], variables['gpu'], *['Yes', 'No']),
    }

    def store_template(*args):
        with open('temp/template_path.txt', 'w') as f:
            f.write(variables['template'].get())

    set_defaults()
    footer.configure(textvariable=variables['status'])
    variables['template'].trace('w', store_template)
    #to_city_entry.bind("<KeyRelease>", caps_to)

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

        def trim_helper(folder, em, filemode):
            if em:
                outfiles = tutils.extract_videos_for_processing(
                    folder, False, filemode, guivar=[variables['status'], window])

                trimmed_videos.extend(outfiles)
            else:
                files = [folder] if filemode else [
                    folder + f for f in os.listdir(folder) if f.endswith(extension)]

                for idx, f in enumerate(files):
                    srcfile = f[f.rfind('/') + 1:]
                    analysis_subfolder = folder + \
                        'tracking/video{}/'.format(str(idx + 1))
                    try_create(analysis_subfolder)
                    outfile = srcfile[:srcfile.rfind(
                        '.')] + '_trim' + extension
                    tutils.trim_video(f, outfile, variables[
                                      'trims'].get(), variables['trime'].get())
                    trimmed_videos.append(outfile)

                    newfile = tutils.crop_and_trim(f)
                    trimmed_videos.append(newfile)
                    variables['status'].set(
                        'Trimmed video {} of {}'.format(idx + 1, len(files)))

        if isyes('trim'):
            if isyes('folderjob'):
                folder = folder[:folder.rfind('/') + 1]
                try_create(folder + 'tracking/')
                trim_helper(folder, isyes('em'), False)
            else:
                trim_helper(folder, isyes('em'), True)

            if isyes('crop'):
                prev_points = []
                for f in trimmed_videos:
                    cropped, prev_points = tutils.crop_and_trim(f, prev_points)
                    variables['affa'].append(cropped)
                    variables['status'].set(
                        'Cropped video {} of {}'.format(idx + 1, len(trimmed_videos)))
            else:
                variables['affa'] = trimmed_videos[:]

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
            #footer['text'] = 'You have not enabled trimming'
            variables['status'].set('You have not enabled trimming')

    def run_app():
        if not variables['affa'] and variables['folder'].get():
            variables['affa'].append(variables['folder'].get())

        for idx, vid in enumerate(variables['affa']):
            temp = variables['template'].get()
            th = variables['threshold'].get()
            sb = isyes('sb')
            vidn = [idx + 1, len(variables['affa'])]
            tutils.track_video(vid, temp, th, sb, vidn=vidn,
                               guivar=variables['status'])

        variables['status'].set('Finished tracking')

    def refresh_app():
        set_defaults()

    # a proper app needs some buttons too!
    button_trim = tk.Button(bottom_frame, text="Trim & Crop", command=trim_crop_video_fn,
                            bg='dark red', fg='black', relief='raised', width=20, height=2, font=('Helvetica 12'))
    button_trim.grid(column=0, row=0, sticky='e', padx=100, pady=2)

    button_refresh = tk.Button(bottom_frame, text="Refresh", command=refresh_app,
                               bg='dark red', fg='black', relief='raised', width=20, height=2, font=('Helvetica 12'))
    button_refresh.grid(column=1, row=0, sticky='e', padx=100, pady=2)

    button_run = tk.Button(bottom_frame, text="Track video", command=run_app, bg='dark green',
                           fg='black', relief='raised', width=20, height=2, font=('Helvetica 12 bold'))
    button_run.grid(column=0, row=1, sticky='w', padx=100, pady=2)

    button_close = tk.Button(bottom_frame, text="Exit", command=close_app,
                             bg='dark red', fg='black', relief='raised', width=20, height=2, font=('Helvetica 12'))
    button_close.grid(column=1, row=1, sticky='e', padx=100, pady=2)

    window.mainloop()

if __name__ == '__main__':
    app('')
