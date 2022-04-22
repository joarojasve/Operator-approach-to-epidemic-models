import math as ma
import numpy as np
import pylab as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import pandas as pd
from scipy.integrate import odeint
I0=[1,2,3,4,5,6,7,8,9]
m=[]
aes=[]
color=sns.color_palette("RdYlGn_r",len(I0))
for i in range(len(I0)):
    df = pd.read_csv('I0_'+str(I0[i])+'.csv')
    df=df[df["R0"]>1]
    df= df[df['probability'] != 0]
    #print(df)
    prob=df["probability"].to_numpy()
    R0=df["R0"].to_numpy()
    plt.scatter(R0,prob,color=color[i])
    model = np.polyfit(R0, np.log10(prob), 1)
    k1=10**model[1]
    a=10**model[0]
    plt.plot(R0,k1*(a**R0),color=color[i])
    plt.xlabel(r"$\mathcal{R}_0$")
    plt.ylabel("Probability")
    m.append(model[0])
    aes.append(10**model[0])
    print(model)
plt.show()
plt.ylim(0.01,1.01)
plt.xlabel(r"$I_0$")
plt.ylabel("basis ("+r"$a$" +")")
plt.yscale('log')
plt.scatter(I0,aes)
plt.show()