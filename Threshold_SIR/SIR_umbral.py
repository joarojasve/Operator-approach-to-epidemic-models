import math as ma
import numpy as np
import pylab as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import pandas as pd
from scipy.integrate import odeint
### You may want to try with popylation size of 50 (small) to see the events
### In this case uncomment the next line
#N0=50.0
I0=1
R0=0
#beta=0.5
gamma=1.0
popsize=1000
S0=popsize-I0-R0
nreacciones=2
Tmax=50.0
tau=0.1
np.random.seed(0)
def find(condition):
    res, = np.nonzero(np.ravel(condition))
    return res
    
def stoc_eqs(INP,beta,gamma,N,nreacciones): 
    V = INP
    Rate=np.zeros((nreacciones))
    Change=np.zeros((nreacciones,len(V)))
    N=V[0]+V[1]+V[2]
    Rate[0] = beta*V[0]*V[1]/N; Change[0,:]=([-1, +1, 0])
    Rate[1] = gamma*V[1];  Change[1,:]=([0, -1, +1])
    for i in range(nreacciones):
        K_i=np.random.poisson(Rate[i]*tau)
        K_i=min([K_i, V[find(Change[i,:]<0)]])
        V=V+Change[i,:]*K_i
    return V

def Stoch_Iteration(INPUT,beta,gamma,N,nreacciones,T):
    lop=0
    S=[]
    I=[]
    R=[]
    SI=[]
    for lop in T:
        res = stoc_eqs(INPUT,beta,gamma,N,nreacciones)
        S.append(INPUT[0])
        I.append(INPUT[1])
        R.append(INPUT[2])
        if INPUT[1]==0:
            break
        INPUT=res
    return S,I,R
def one_trajectory(S0,I0,R0,beta,gamma,N,nreacciones,Tmax,tau):
    INPUT = np.array((S0,I0,R0))
    T=np.arange(0.0, Tmax, tau)
    [S,I,R]=Stoch_Iteration(INPUT,beta,gamma,N,nreacciones,T)
    return R[-1]

nsims=1000

def R_inf_dist(nsims,S0,I0,R0,beta,gamma,popsize,nreacciones,Tmax,tau):
    T=np.arange(0.0, Tmax, tau)
    matrs=[0.0 for i in range(popsize+1)]
    n=[]
    for l in range(nsims):
        Ri=one_trajectory(S0,I0,R0,beta,gamma,popsize,nreacciones,Tmax,tau)
        matrs[int(Ri)]+=1
        n.append(Ri)
        #if l%500==0:
            #print(l)
    return n


umbral=100
plt.figure(figsize=(10,6))
def p_calculator(umbral,nsims,S0,I0,R0,gamma,popsize,nreacciones,Tmax,tau,labelu,coloru):
    erreceros=[0.9+0.2*i for i in range(20)]
    erreceros_str=["{:.1f}".format(erre0) for erre0 in erreceros]
    pes=[]
    for errecero in erreceros: 
        beta=errecero*gamma
        ARs=R_inf_dist(nsims,S0,I0,R0,beta,gamma,popsize,nreacciones,Tmax,tau)
        ARsn=[n for n in ARs if n<umbral]
        pes.append(len(ARsn)/nsims)
        print(len(ARsn)/nsims)
    dic = {'R0': erreceros, 'probability': pes}
    df = pd.DataFrame(dic)
    df.to_csv(str(gamma)+".csv", index=False) 
    er0=[0.1,0.2,0.3,0.4,0.5,0.6,0.7]
    pe1=[1.0 for i in range(7)]
    plt.plot(er0+erreceros,pe1+pes,label=labelu,color=coloru)
    return None
plt.xlabel(r"$\mathcal{R}_0$", fontsize=15)
plt.ylabel("Probability of no-outbreak ", fontsize=15)
plt.xticks(rotation=45)

plt.fill_between([0.0,1.0],[1.0,1.0],color="r",alpha=0.1)
p_calculator(umbral,nsims,999,1,0,gamma,popsize,nreacciones,Tmax,tau,r"$\gamma = 0.1 $"+" "+r"$ d^{-1}$","olive")
p_calculator(umbral,nsims,998,2,0,gamma,popsize,nreacciones,Tmax,tau,r"$\gamma = 1.0$"+" "+r"$  d^{-1}$","darkorange")
p_calculator(umbral,nsims,S0,I0,R0,gamma,popsize,nreacciones,Tmax,tau,r"$\gamma = 10.0 $"+" "+r"$ d^{-1}$","darkred")
plt.legend()
plt.tight_layout()
plt.show()
