from tkinter import *
def Train():
    """GUI"""
    import tkinter as tk
    import numpy as np
    import pandas as pd

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import LabelEncoder

    root = tk.Tk()

    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry("%dx%d+0+0" % (w, h))
    root.title("Mobile Botnet")
    root.configure(background="deeppink4")
    
    tele_device = tk.IntVar()
    tele_subscriber = tk.IntVar()
    abort = tk.IntVar()
    sendsms = tk.IntVar()
    delete_pack = tk.IntVar()
    sms_received = tk.IntVar()
    ljava = tk.IntVar()
    phone_s= tk.IntVar()
    readsms = tk.IntVar()
    boot_comp = tk.IntVar()
    io_delete = tk.DoubleVar()
    chown = tk.IntVar()
    chmod = tk.IntVar()
    mount = tk.IntVar()
    apk = tk.IntVar()
    zip_file = tk.DoubleVar()
    dex_file= tk.IntVar()
    camera = tk.IntVar()
    access = tk.IntVar()
    package = tk.IntVar()
    battery_low = tk.IntVar()
    so_file = tk.DoubleVar()
    power_connec = tk.IntVar()
    load_lib = tk.IntVar()
    exe_file = tk.IntVar()
    
    #===================================================================================================================
    def Detect():
        e1=tele_device.get()
        print(e1)
        e2=tele_subscriber.get()
        print(e2)
        e3=abort.get()
        print(e3)
        #print(type(e3))
        e4=sendsms.get()
        print(e4)
        e5=delete_pack.get()
        print(e5)
        e25=phone_s.get()
        print(e5)
        e6=sms_received.get()
        print(e6)
        e7=ljava.get()
        print(e7)
        e8=readsms.get()
        print(e8)
        e9=boot_comp.get()
        print(e9)
        e10=io_delete.get()
        print(e10)
        e11=chown.get()
        print(e11)
        e12=chmod.get()
        print(e12)
        e13=mount.get()
        print(e13)
    
        e14=apk.get()
        print(e14)
        e15=zip_file.get()
        print(e15)
        e16=dex_file.get()
        print(e16)    
        e17=camera.get()
        print(e17)
        e18=access.get()
        print(e18)
        e19=package.get()
        print(e19)
        e20=battery_low.get()
        print(e20)
        
        e21=so_file.get()
        print(e21)
        e22=power_connec.get()
        print(e22)
        e23=load_lib.get()
        print(e23)
        e24=exe_file.get()
        print(e24)

        
        
        
        #########################################################################################
        
        from joblib import dump , load
        a1=load('C:/Users/Dell/OneDrive/Desktop/botnet_mobilesvm (1)/botnet_mobilesvm/botnet_MODEL.joblib')
        v= a1.predict([[e1, e2, e3, e4, e5,e25, e6, e7, e8, e9,e10, e11, e12, e13,e14, e15, e16, e17, e18, e19, e20, e21, e22,e23, e24]])
        print(v)
        if v[0]==1:
            print("Yes")
            yes = tk.Label(root,text="DDoS attack",background="red",foreground="white",font=('times', 20, ' bold '),width=15)
            yes.place(x=300,y=100)
                     
        else:
            print("No")
            no = tk.Label(root, text="No DDoS attack", background="green", foreground="white",font=('times', 20, ' bold '),width=15)
            no.place(x=300, y=100)
            


    l1=tk.Label(root,text="Telephony Get Device ID",background="olive",font=('times', 20, ' bold '),width=25)
    l1.place(x=100,y=30)
    tele_device=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=tele_device)
    tele_device.place(x=650,y=30)

    l2=tk.Label(root,text="Telephony Get Subscriber ID",background="olive",font=('times', 20, ' bold '),width=25)
    l2.place(x=100,y=80)
    tele_subscriber=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=tele_subscriber)
    tele_subscriber.place(x=650,y=80)

    l3=tk.Label(root,text="Abort Broadcast",background="olive",font=('times', 20, ' bold '),width=15)
    l3.place(x=100,y=130)
    abort=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=abort)
    abort.place(x=650,y=130)
    
    l4=tk.Label(root,text="Send SMS",background="olive",font=('times', 20, ' bold '),width=15)
    l4.place(x=100,y=180)
    sendsms=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=sendsms)
    sendsms.place(x=650,y=180)

    l5=tk.Label(root,text="Delete Packages",background="olive",font=('times', 20, ' bold '),width=15)
    l5.place(x=100,y=230)
    delete_pack=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=delete_pack)
    delete_pack.place(x=650,y=230)

    l6=tk.Label(root,text="Phone_State",background="olive",font=('times', 20, ' bold '),width=15)
    l6.place(x=100,y=280)
    phone_s=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=phone_s)
    phone_s.place(x=650,y=280)

    l7=tk.Label(root,text="SMS_Received",background="olive",font=('times', 20, ' bold '),width=15)
    l7.place(x=100,y=330)
    sms_received=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=sms_received)
    sms_received.place(x=650,y=330)

    l8=tk.Label(root,text="Ljava.net.InetSocket Address",background="olive",font=('times', 20, ' bold '),width=25)
    l8.place(x=100,y=380)
    ljava=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=ljava)
    ljava.place(x=650,y=380)

    l9=tk.Label(root,text="Read_SMS",background="olive",font=('times', 20, ' bold '),width=10)
    l9.place(x=100,y=430)
    readsms=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=readsms)
    readsms.place(x=650,y=430)

    l10=tk.Label(root,text="Android.intent.action_Boot_completed",background="olive",font=('times', 20, ' bold '),width=30)
    l10.place(x=100,y=480)
    boot_comp=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=boot_comp)
    boot_comp.place(x=650,y=480)

    l11=tk.Label(root,text="IO.File.*delete",background="olive",font=('times', 20, ' bold '),width=15)
    l11.place(x=100,y=530)
    io_delete=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=io_delete)
    io_delete.place(x=650,y=530)

    l12=tk.Label(root,text="Chown",background="olive",font=('times', 20, ' bold '),width=10)
    l12.place(x=100,y=580)
    chown=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=chown)
    chown.place(x=650,y=580)

    l13=tk.Label(root,text="Chmod",background="olive",font=('times', 20, ' bold '),width=10)
    l13.place(x=820,y=30)
    chmod=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=chmod)
    chmod.place(x=1300,y=30)


    l14=tk.Label(root,text="Mount",background="olive",font=('times', 20, ' bold '),width=10)
    l14.place(x=820,y=80)
    mount=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=mount)
    mount.place(x=1300,y=80)

    l15=tk.Label(root,text=".APK",background="olive",font=('times', 20, ' bold '),width=10)
    l15.place(x=820,y=130)
    apk=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=apk)
    apk.place(x=1300,y=130)

    l16=tk.Label(root,text=".ZIP",background="olive",font=('times', 20, ' bold '),width=10)
    l16.place(x=820,y=180)
    zip_file=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=zip_file)
    zip_file.place(x=1300,y=180)

    l17=tk.Label(root,text=".DEX",background="olive",font=('times', 20, ' bold '),width=10)
    l17.place(x=820,y=230)
    dex_file=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=dex_file)
    dex_file.place(x=1300,y=230)

    l18=tk.Label(root,text="Camera",background="olive",font=('times', 20, ' bold '),width=10)
    l18.place(x=820,y=280)
    camera=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=camera)
    camera.place(x=1300,y=280)
    
    l19=tk.Label(root,text="Access_fine_location",background="olive",font=('times', 20, ' bold '),width=15)
    l19.place(x=820,y=330)
    access=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=access)
    access.place(x=1300,y=330)

    l20=tk.Label(root,text="install_packages",background="olive",font=('times', 20, ' bold '),width=15)
    l20.place(x=820,y=280)
    package=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=package)
    package.place(x=1300,y=280)

    l21=tk.Label(root,text="Android.intent.action.Battery_low",background="olive",font=('times', 20, ' bold '),width=25)
    l21.place(x=820,y=330)
    battery_low=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=battery_low)
    battery_low.place(x=1300,y=330)

    l22=tk.Label(root,text=".SO",background="olive",font=('times', 20, ' bold '),width=10)
    l22.place(x=820,y=380)
    so_file=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=so_file)
    so_file.place(x=1300,y=380)

    l23=tk.Label(root,text="Android.intent.action.power_connected",background="olive",font=('times', 20, ' bold '),width=30)
    l23.place(x=820,y=430)
    power_connec=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=power_connec)
    power_connec.place(x=1300,y=430)
    
    l24=tk.Label(root,text="System.* Load Library",background="olive",font=('times', 20, ' bold '),width=30)
    l24.place(x=820,y=480)
    load_lib=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=load_lib)
    load_lib.place(x=1300,y=480)

    l25=tk.Label(root,text=".EXE",background="olive",font=('times', 20, ' bold '),width=10)
    l25.place(x=820,y=530)
    exe_file=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=exe_file)
    exe_file.place(x=1300,y=530)
    
    button1 = tk.Button(root,text="Submit",command=Detect,font=('times', 20, ' bold '),width=10)
    button1.place(x=600,y=700)


    root.mainloop()

Train()