import tkinter

window = tkinter.Tk()
# to rename the title of the window
window.title("GUI")
window.geometry("650x100")
# pack is used to show the object in the window
label = tkinter.Label(window, text = "Hello World!").pack()
window.mainloop()