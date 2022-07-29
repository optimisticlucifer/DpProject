# %%
from tkinter import *
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter("ignore")

pickled_model = pickle.load(open('model.pkl', 'rb'))
df_s = pd.read_csv('severity.csv')
med = pd.read_csv('medicines.csv')
x = np.array(df_s['Symptom'])
y = np.array(df_s['weight'])


def predict(lis):
    l = [0]*7
    for i in range(len(lis)):
        l[i] = lis[i]
    res = [l]
    pred = pickled_model.predict(res)
    return pred


root = Tk()
root.title('Medico')

mDisplay = Entry(root, width=20, borderwidth=5)
mDisplay.grid(row=6, column=0, columnspan=3, padx=5, pady=5)


mDisplay2 = Entry(root, width=20, borderwidth=5)
mDisplay2.grid(row=6, column=2, columnspan=3, padx=5, pady=5)

lis = []


def action(number):
    current = mDisplay.get()
    mDisplay.delete(0, END)
    mDisplay2.delete(0, END)
    mDisplay.insert(0, str(current)+ str(number)+',')
    current = mDisplay.get()
    lis.append(number)


def enteraction():
    if len(lis)>0 and len(lis)<8:
        m = predict(lis)
        p = m[0]
        x1 = np.array(med['Disease'])
        y1 = np.array(med['Medicine'])
        for i in range(len(x1)):
            if p == x1[i]:
                mDisplay.delete(0, END)
                mDisplay2.delete(0, END)
                mDisplay.insert(0, str(p))
                mDisplay2.insert(0, str(y1[i]))
    else:
        mDisplay2.delete(0, END)
        if(len(lis)>=8):
            mDisplay2.insert(0, str("Please consult a doctor"))
        else:
            mDisplay2.insert(0, str("Please enter someting"))
        


def clraction():
    lis.clear()
    mDisplay.delete(0, END)
    mDisplay2.delete(0, END)

padxy=0

button=[[None]*5]*6

# Create Button widget
button[0][0] = Button(root, text=x[0], padx=padxy, pady=padxy,command=lambda: action(0)).grid(row=0, column=0)
button[0][1] = Button(root, text=x[1], padx=padxy, pady=padxy,command=lambda: action(1)).grid(row=0, column=1)
button[0][2] = Button(root, text=x[2], padx=padxy, pady=padxy,command=lambda: action(2)).grid(row=0, column=2)
button[0][3] = Button(root, text=x[3], padx=padxy, pady=padxy,command=lambda: action(3)).grid(row=0, column=3)
button[0][4] = Button(root, text=x[4], padx=padxy, pady=padxy,command=lambda: action(4)).grid(row=0, column=4)
button[1][0] = Button(root, text=x[5], padx=padxy, pady=padxy,command=lambda: action(5)).grid(row=1, column=0)
button[1][1] = Button(root, text=x[6], padx=padxy, pady=padxy,command=lambda: action(6)).grid(row=1, column=1)
button[1][2] = Button(root, text=x[7], padx=padxy, pady=padxy,command=lambda: action(7)).grid(row=1, column=2)
button[1][3] = Button(root, text=x[8], padx=padxy, pady=padxy,command=lambda: action(8)).grid(row=1, column=3)
button[1][4] = Button(root, text=x[9], padx=padxy, pady=padxy,command=lambda: action(9)).grid(row=1, column=4)
button[2][0] = Button(root, text=x[10], padx=padxy, pady=padxy,command=lambda: action(10)).grid(row=2, column=0)
button[2][1] = Button(root, text=x[11], padx=padxy, pady=padxy,command=lambda: action(11)).grid(row=2, column=1)
button[2][2] = Button(root, text=x[12], padx=padxy, pady=padxy,command=lambda: action(12)).grid(row=2, column=2)
button[2][3] = Button(root, text=x[13], padx=padxy, pady=padxy,command=lambda: action(13)).grid(row=2, column=3)
button[2][4] = Button(root, text=x[14], padx=padxy, pady=padxy,command=lambda: action(14)).grid(row=2, column=4)
button[3][0] = Button(root, text=x[15], padx=padxy, pady=padxy,command=lambda: action(15)).grid(row=3, column=0)
button[3][1] = Button(root, text=x[16], padx=padxy, pady=padxy,command=lambda: action(16)).grid(row=3, column=1)
button[3][2] = Button(root, text=x[17], padx=padxy, pady=padxy,command=lambda: action(17)).grid(row=3, column=2)
button[3][3] = Button(root, text=x[18], padx=padxy, pady=padxy,command=lambda: action(18)).grid(row=3, column=3)
button[3][4] = Button(root, text=x[19], padx=padxy, pady=padxy,command=lambda: action(19)).grid(row=3, column=4)
button[4][0] = Button(root, text=x[20], padx=padxy, pady=padxy,command=lambda: action(20)).grid(row=4, column=0)
button[4][1] = Button(root, text=x[21], padx=padxy, pady=padxy,command=lambda: action(21)).grid(row=4, column=1)
button[4][2] = Button(root, text=x[22], padx=padxy, pady=padxy,command=lambda: action(22)).grid(row=4, column=2)
button[4][3] = Button(root, text=x[23], padx=padxy, pady=padxy,command=lambda: action(23)).grid(row=4, column=3)
button[4][4] = Button(root, text=x[24], padx=padxy, pady=padxy,command=lambda: action(24)).grid(row=4, column=4)
button[5][0] = Button(root, text=x[25], padx=padxy, pady=padxy,command=lambda: action(25)).grid(row=5, column=0)
button[5][1] = Button(root, text=x[26], padx=padxy, pady=padxy,command=lambda: action(26)).grid(row=5, column=1)
button[5][2] = Button(root, text=x[27], padx=padxy, pady=padxy,command=lambda: action(27)).grid(row=5, column=2)
button[5][3] = Button(root, text=x[28], padx=padxy, pady=padxy,command=lambda: action(28)).grid(row=5, column=3)
button[5][4] = Button(root, text=x[29], padx=padxy, pady=padxy,command=lambda: action(0)).grid(row=5, column=4)


buttonclr = Button(root, text="clr", padx=padxy, pady=padxy,command=lambda: clraction()).grid(row=7, column=0, columnspan=3)
buttonenter = Button(root, text="Enter", padx=padxy, pady=padxy,command=lambda: enteraction()).grid(row=7, column=2, columnspan=3)

root.mainloop()

# %%
