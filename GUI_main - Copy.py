import tkinter as tk
#from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk



##############################################+=============================================================
root = tk.Tk()
root.configure(background="white")
# root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Mobile Botnet Detection System")

# 43

# ++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
image2 = Image.open('slide.jpg')
image2 = image2.resize((w,h), Image.ANTIALIAS)

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)



#
label_l1 = tk.Label(root, text="Mobile Botnet Detection System",font=("Times New Roman", 30, 'bold'),
                    background="brown", fg="white", width=67, height=2)
label_l1.place(x=0, y=0)

img = Image.open('clogo.jpg')
img = img.resize((100,70), Image.ANTIALIAS)
logo_image = ImageTk.PhotoImage(img)

logo_label = tk.Label(root, image=logo_image)
logo_label.image = logo_image
logo_label.place(x=40, y=10)

frame_alpr = tk.LabelFrame(root, text=" --Details-- ", width=700, height=600, bd=5, font=('times', 14, ' bold '),bg="grey")
frame_alpr.grid(row=0, column=0, sticky='nw')
frame_alpr.place(x=800, y=150)

label_l2 = tk.Label(frame_alpr, text="Mobile Botnet Detection System \n \n \n Android is now the most widespread mobile operating system worldwide. \n Over the years the volume of malware targeting Android has continued to grow. \n \n This is because it is easier and more profitable for malware authors to target \n an operating system that is open-source, more prevalent, and does not restrict the installation of apps from any possible source.\n As a matter of fact, numerous families of malware apps that are  capable of infecting Android devices and \n turning them into malicious bots have been discovered in the wild.\n These Android bots may become part of a larger botnet that can be used to perform various types of attacks such as Distributed Denial of Service (DDoS) attacks, generation and distribution of Spam, Phishing attacks, click fraud, stealing login credentials or credit card details, etc.",font=("Times New Roman",15, 'bold'),width=50,
                    background="grey", fg="white")
label_l2.place(x=30, y=15) 


img1 = Image.open('slide1.jpg')
img1 = img1.resize((750,600), Image.ANTIALIAS)
logo_image1 = ImageTk.PhotoImage(img1)

logo_label1 = tk.Label(root, image=logo_image1)
logo_label1.image = logo_image1
logo_label1.place(x=15, y=150)







def log():
    from subprocess import call
    call(["python","GUI_main.py"])
    root.destroy()
    
def window():
  root.destroy()
  
  
def con():
    from subprocess import call
    call(["python","crop_login"])
    root.destroy()

def about():
    from subprocess import call
    call(["python","register.py"])
    root.destroy()
    
    
button1 = tk.Button(label_l1, text="LOGIN", command=con, width=10, height=1,font=('times 15 bold underline'),bd=0, bg="brown", fg="white")
button1.place(x=1180, y=50)

button2 = tk.Button(label_l1, text="REGISTER",command=about,width=13, height=1,font=('times 15 bold underline'), bd=0,bg="brown", fg="white")
button2.place(x=1290, y=50)

button4 = tk.Button(label_l1, text="EXIT", command=con, width=10, height=1,font=('times 15 bold underline'),bd=0,bg="brown", fg="white")
button4.place(x=1430, y=50)


# button1 = tk.Button(frame_alpr, text="Login", command=log, width=14, height=1,font=('times', 20, ' bold '), bg="Black", fg="white")
# button1.place(x=150, y=110)

# button2 = tk.Button(frame_alpr, text="Register",command=reg,width=14, height=1,font=('times', 20, ' bold '), bg="black", fg="white")
# button2.place(x=150, y=200)

# button3 = tk.Button(frame_alpr, text="Exit",command=window,width=14, height=1,font=('times', 20, ' bold '), bg="red", fg="white")
# button3.place(x=150, y=300)


label_l1 = tk.Label(root, text="** Mobile Botnet Detection System @2021 By ___ **",font=("Times New Roman", 10, 'bold'),
                    background="black", fg="white", width=250, height=2)
label_l1.place(x=0, y=800)


root.mainloop()