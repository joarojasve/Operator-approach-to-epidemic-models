import math as ma
import numpy as np
import pylab as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import pandas as pd
from scipy.integrate import odeint


#beta=0.5
gamma=1.0


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

nsims=4000

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



plt.figure(figsize=(10,6))
def p_calculator(errecero,nsims,I0,R0,gamma,nreacciones,Tmax,tau,labelu):
    #
    #erreceros_str=["{:.1f}".format(erre0) for erre0 in erreceros]
    pes=[]
    Ns=range(100,6000,500)
    for j in range(len(Ns)): 
        S0=Ns[j]-I0
        umbral=0.1*Ns[j]
        beta=errecero*gamma
        ARs=R_inf_dist(nsims,S0,I0,R0,beta,gamma,Ns[j],nreacciones,Tmax,tau)
        ARsn=[n for n in ARs if n<umbral]
        pes.append(len(ARsn)/nsims)

        #if len(ARsn)==0:
        #    break
        print(len(ARsn)/nsims)
    dic = {'Ns': Ns, 'probability': pes}
    df = pd.DataFrame(dic)
    df.to_csv("errecero_"+str(errecero)+".csv", index=False) 
    #er0=[0.1,0.2]
    #pe1=[1.0 for i in range(2)]
    plt.plot(Ns,pes,label=labelu)
plt.xlabel(r"$N$", fontsize=15)
plt.ylabel("Probability of no-outbreak ", fontsize=15)
plt.xticks(rotation=45)

#plt.fill_between([0.0,1.0],[1.0,1.0],color="r",alpha=0.1)
erreceros=erreceros=[1.1,2.0,3.0,4.0,5.0]
for i in range(len(erreceros)):
    p_calculator(erreceros[i],nsims,1,0,gamma,nreacciones,Tmax,tau,r"$R_0 = $"+str(erreceros[i]))
plt.legend()
plt.tight_layout()
plt.show()
