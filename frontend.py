import tkinter as tk
import easygui
import pandas as pd
from time import strftime
from time import sleep
from config import production_houses, house_design_numbers, db_file
import subprocess
import signal
import os

def app(database):
    window = tk.Tk()  # you may also see it named as "root" in other sources

    window.title("Sakura Inventory")
    window.geometry("600x600")
    # window.minsize(width=600, height=600) # you can define the minimum size
    # of the window like this
    # change to false if you want to prevent resizing
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
    header = tk.Label(frame_header, text="Sakura Inventory Management", bg='black',
                      fg='white', height='3', width='50', font=("Helvetica 16 bold"))
    header.grid(row=0, column=0)
    footer = tk.Label(footer_frame, text='', bg='black',
                      fg='white', height='3', width='50', font=("Helvetica 16 bold"))
    footer.grid(row=2, column=0)

    def open_window(*args):
        filename = tk.filedialog.askopenfilename(
            title='Please select the image')
        variables['photo'].set(filename)
        entries['photo']['text'] = filename

    frames = {
        'f1': tk.Frame(center_frame, borderwidth=2, relief='raised'),
        'f2': tk.Frame(center_frame, borderwidth=2, relief='sunken'),
        'f3': tk.Frame(center_frame, borderwidth=2, relief='sunken'),
        'f4': tk.Frame(center_frame, borderwidth=2, relief='sunken'),
        'f5': tk.Frame(center_frame, borderwidth=2, relief='sunken'),
        'f6': tk.Frame(center_frame, borderwidth=2, relief='sunken'),
        'f7': tk.Frame(center_frame, borderwidth=2, relief='sunken'),
        'f8': tk.Frame(center_frame, borderwidth=0, relief='sunken'),
        #'f9': tk.Frame(center_frame, borderwidth=0, relief='raised')
    }

    labels = {
        'sn': tk.Label(frames['f1'], text="Serial Number: "),
        'photo': tk.Label(frames['f1'], text="Photo: "),
        'hn': tk.Label(frames['f2'], text="House Number: "),
        'hdn': tk.Label(frames['f2'], text="Design Number: "),
        'cdn': tk.Label(frames['f3'], text="Customer Design Number: "),
        'k18_PT': tk.Label(frames['f3'], text="k18_PT: "),
        'wt': tk.Label(frames['f4'], text="Weight: "),
        'size': tk.Label(frames['f4'], text="Size: "),
        'waku_price': tk.Label(frames['f5'], text="waku/price: "),
        'labour': tk.Label(frames['f5'], text="Labour: "),
        'dialot': tk.Label(frames['f6'], text="dialot: "),
        'CTS': tk.Label(frames['f6'], text="CTS: "),
        'AT': tk.Label(frames['f7'], text="@: "),
        'diacost': tk.Label(frames['f7'], text="diacost: "),
        'totalcost': tk.Label(frames['f8'], text="Total Cost: ")
    }

    # http://effbot.org/tkinterbook/variable.htm
    variables = {
        'sn': tk.StringVar(),
        'photo': tk.StringVar(),
        'hn': tk.StringVar(),
        'hdn': tk.StringVar(),
        'cdn': tk.StringVar(),
        'k18_PT': tk.StringVar(),
        'wt': tk.StringVar(),
        'size': tk.StringVar(),
        'waku_price': tk.StringVar(),
        'labour': tk.StringVar(),
        'dialot': tk.StringVar(),
        'CTS': tk.StringVar(),
        'AT': tk.StringVar(),
        'diacost': tk.StringVar(),
        'totalcost': tk.StringVar()
    }

    entries = {
        'sn': tk.Entry(frames['f1'], textvariable=variables['sn'], width=4),
        'photo': tk.Button(frames['f1'], text="Browse...", command=open_window, bg='dark green',
                           fg='black', relief='raised', width=10, font=('Helvetica 9 bold')),
        'hn': tk.OptionMenu(frames['f2'], variables['hn'], *production_houses),
        #tk.Entry(frames['f2'], textvariable=variables['hn'], width=4),
        'hdn': tk.OptionMenu(frames['f2'], variables['hdn'], *house_design_numbers[production_houses[0]]),
        #tk.Entry(frames['f2'], textvariable=variables['dn'], width=4),
        'cdn': tk.Entry(frames['f3'], textvariable=variables['cdn'], width=4),
        'k18_PT': tk.Entry(frames['f3'], textvariable=variables['k18_PT'], width=4),
        'wt': tk.Entry(frames['f4'], textvariable=variables['wt'], width=4),
        'size': tk.Entry(frames['f4'], textvariable=variables['size'], width=4),
        'waku_price': tk.Entry(frames['f5'], textvariable=variables['waku_price'], width=4),
        'labour': tk.Entry(frames['f5'], textvariable=variables['labour'], width=4),
        'dialot': tk.Entry(frames['f6'], textvariable=variables['dialot'], width=4),
        'CTS': tk.Entry(frames['f6'], textvariable=variables['CTS'], width=4),
        'AT': tk.Entry(frames['f7'], textvariable=variables['AT'], width=4),
        'diacost': tk.Entry(frames['f7'], textvariable=variables['diacost'], width=4),
        'totalcost': tk.Entry(frames['f8'], textvariable=variables['totalcost'], width=4)
    }

    def callback_hn(*args):
	    entries['hdn']=tk.OptionMenu(frames['f2'], variables['hdn'], *house_design_numbers[variables['hn'].get()])
	    variables['hdn'].set(house_design_numbers[variables['hn'].get()][0])


    variables['hn'].set(production_houses[0])
    variables['hdn'].set(house_design_numbers[production_houses[0]][0])
    variables['sn'].set(database.header-1)
    variables['hn'].trace('w', callback_hn)
    #to_city_entry.bind("<KeyRelease>", caps_to)

    # pack UI elements
    for k in labels.keys():
        labels[k].pack(side='left')
        entries[k].pack(side='left', padx=1)

    for f in frames.values():
        f.pack(fill='x', pady=2)

    def close_app():
        window.destroy()

    def print_barcode():
        if len(database.jewelleries)>0:
            footer['text']='Printing barcode...'
            file=database.jewelleries[-1].barcodefile
            with open(file,'rb') as f:
                img=f.read()
            with subprocess.Popen("/usr/bin/lpr", stdin=subprocess.PIPE) as lpr:
                lpr.stdin.write(img)
                #lpr.stdin.flush()
            footer['text']='Printing finished. Ready for next item...'

    def run_app():
        database.add_entry(variables)
        if footer['text']=='' or footer['text']=='Done!':
            footer['text'] = 'Saved!'
        elif footer['text'] == 'Saved!':
            footer['text'] = 'Done!'
        else:
            footer['text'] = 'Saved!'

    def refresh_app():
        variables['sn'].set(database.header-1)
        entries['photo']['text']='Browse...'
        variables['hn'].set(production_houses[0])
        variables['hdn'].set(house_design_numbers[production_houses[0]][0])
        variables['cdn'].set('')
        variables['k18_PT'].set('')
        variables['wt'].set('')
        variables['size'].set('')
        variables['waku_price'].set('')
        variables['labour'].set('')
        variables['dialot'].set('')
        variables['CTS'].set('')
        variables['AT'].set('')
        variables['diacost'].set('')
        variables['totalcost'].set('')


    # a proper app needs some buttons too!
    button_barcode = tk.Button(bottom_frame, text="Print Barcode", command=print_barcode,
                             bg='dark red', fg='black', relief='raised', width=10, font=('Helvetica 12'))
    button_barcode.grid(column=0, row=0, sticky='e', padx=100, pady=2)

    button_refresh = tk.Button(bottom_frame, text="Refresh", command=refresh_app,
                             bg='dark red', fg='black', relief='raised', width=10, font=('Helvetica 12'))
    button_refresh.grid(column=1, row=0, sticky='e', padx=100, pady=2)

    button_run = tk.Button(bottom_frame, text="Register", command=run_app, bg='dark green',
                           fg='black', relief='raised', width=10, font=('Helvetica 9 bold'))
    button_run.grid(column=0, row=1, sticky='w', padx=100, pady=2)
    
    button_close = tk.Button(bottom_frame, text="Exit", command=close_app,
                             bg='dark red', fg='black', relief='raised', width=10, font=('Helvetica 9'))
    button_close.grid(column=1, row=1, sticky='e', padx=100, pady=2)

    window.mainloop()

if __name__ == '__main__':
    app(db_file)
