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

nsims=2000

def R_inf_dist(nsims,S0,I0,R0,beta,gamma,popsize,nreacciones,Tmax,tau):
    T=np.arange(0.0, Tmax, tau)
    matrs=[0.0 for i in range(popsize+1)]
    n=[]
    for l in range(nsims):
        Ri=one_trajectory(S0,I0,R0,beta,gamma,popsize,nreacciones,Tmax,tau)
        matrs[int(Ri)]+=1
        n.append(Ri)
        if l%500==0:
            print(l)
    return n
betmat=[]
betas=[0.1+0.1*i for i in range(40)]
betas_str=["{:.1f}".format(beta) for beta in betas]
for beta in betas: 
    betmat.append(R_inf_dist(nsims,S0,I0,R0,beta,gamma,popsize,nreacciones,Tmax,tau))
df = pd.DataFrame(np.array(betmat).T, columns=betas_str)
#print(df["0.5"])
plt.figure(figsize=(10,6))
plt.xlabel(r"$\mathcal{R}_0$", fontsize=15)
plt.ylabel("Attack Rate (R"+ r"$_\infty$"+")", fontsize=15)
sns.stripplot(data=df,size=1000/nsims)
#sns.heatmap(betmat,vmax=0.05,yticklabels=np.array(betas)/gamma,norm=LogNorm())
plt.xticks(rotation=45)
plt.tight_layout()
#plt.show()    




##################################### DETERMINISTA

def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N 
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I 
    return dSdt, dIdt, dRdt
def detRinf(y0,t,N,beta,gamma):
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T    
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    return R[-1]
y0 = S0, I0, R0
T=np.arange(0.0, Tmax, tau)
vvec=[]
betasW=np.arange(0.1,4.0,0.01)
for i in range(len(betasW)):
    erre0=betasW[i]/gamma
    vvec.append(detRinf(y0,T,popsize,betasW[i],gamma))
plt.plot(10*(np.array(betasW))-1.0,vvec,color='darkblue', linestyle='dashed',alpha=0.6)
#plt.fill_between([0.0,9.0],[1000.0,1000.0],color="r",alpha=0.1)
plt.show()
"""def plotter(matriz,title):
    fig=sns.heatmap(matriz) #vmax=0.1
    plt.title(title)
    plt.xlabel("Susceptible individuals")
    plt.ylabel("Infectious individuals")
    plt.ylim(0,popsize+1)
    plt.xlim(0,popsize+1)
    plt.show()
plotter(np.array(matrs[-1])/nsims,"t=25 dias")
"""