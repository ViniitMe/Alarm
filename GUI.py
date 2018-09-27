from Tkinter import *
import Alarm

def open():
	Alarm.main()

Gui=Tk(className="  Sleep if U Can")
Gui.geometry('365x380')

C=Canvas(Gui,bg="blue",height=250,width=300)
pic=PhotoImage(file="/home/vinit/Desktop/2.png")
label=Label(Gui, image=pic).pack()


button=Button(text="Set Alarm", command=open ,fg='red').pack()
Gui.mainloop()