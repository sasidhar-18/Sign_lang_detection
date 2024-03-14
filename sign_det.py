import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import numpy as np

# Loading the Model
from keras.models import load_model
model = load_model('C:\\Users\\SASIDHAR\\Sign_detection.h5')

# Initializing the GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Sign Detection')
top.configure(background='#CDCDCD')

# Initializing the Labels
label = Label(top, background="#CDCDCD", font=('arial', 15, "bold"))
sign_image = Label(top)

file_path = 'C:\\Users\\SASIDHAR\\Desktop\\sign_lang\\A_test.jpg'

# Definig Detect function which detects the sign in the image using the model
def Detect(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((50, 50))
    image = np.array(image)
    image = np.array([image]) / 255
    pred = model.predict(image)
    sign_label = np.argmax(pred)
    print("Predicted Sign Label:", sign_label)
    label.configure(foreground="#011638", text=f"Predicted Sign Label: {sign_label}")

# Defining Show_detect button function
def show_Detect_button(file_path):
    Detect_b = Button(top, text="Detect Sign", command=lambda: Detect(file_path), padx=10, pady=5)
    Detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    Detect_b.place(relx=0.79, rely=0.46)

# Definig Upload Image Function
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_Detect_button(file_path)
    except Exception as e:
        print("Error:", e)

upload = Button(top, text="Upload an Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand=True)
label.pack(side="bottom", expand=True)
heading = Label(top, text="Sign Detection", pady=20, font=('arial', 20, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()
top.mainloop()





