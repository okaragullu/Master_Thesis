from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import csv
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
from sklearn.metrics import mean_squared_error
from math import sqrt
get_ipython().run_line_magic('matplotlib', 'inline')
from smt.sampling_methods import LHS
workdir=r'C:\Users\Orkun\Untitled Folder 1\Blue_RSO_csv'
os.chdir(workdir)


# # Modified Root Mean Square Error (RMSE)



def find_nearest(array, value):
    array = np.asarray(array)   
    idx = (np.abs(array - value)).argmin()
    return array[idx]
def rms_force(x):
    df=pd.read_csv(x)
    forceMax=np.argmax(np.array(df['Force [N]']))
    df=df.iloc[:forceMax]
    Force_Pred=np.array(df['Force [N]'])
    disp=np.array(df['Displacement [mm]'])
    a=[]

    for sim in(disp):
        x= find_nearest(disp_ref,sim)
        force_actual=np.array(force_ref[a])
        a.append(np.argwhere(x == disp_ref)[0][0])
    force_actual=np.array(force_ref[a])
    rms=sqrt(mean_squared_error(force_actual, Force_Pred))
    return rms


# ## Load Experimental Data and Find RMSE Between Exp. and Simulation Data

df_ref = pd.read_csv(r'C:\Users\Orkun\Untitled Folder 1\Blue_Ref.csv')
force_max_ref=np.argmax(np.array(df_ref.Load))
df_ref=df_ref.iloc[:force_max_ref]
force_ref = np.array(df_ref.Load)
disp_ref = np.array(df_ref.Displacement)
df_samp=pd.read_csv('sro2.csv')

os.chdir(workdir)
files=sorted([x for x in os.listdir() if 'Force' in x])
df_samp['files']=files
df_samp=df_samp.set_index('files')
df_samp['rms']=0
df_samp.head()

for xi in files:
    df_samp['rms'].loc[xi]=rms_force(xi)
df_samp.to_csv('RSO_DP_2.2_fix_rmse.csv')
df_samp.head()


# #  Finding Polynomial Equation of the Surface



crss_0=np.asarray(df_samp.crss)
h0=np.asarray(df_samp.h0)
rms=np.asarray(df_samp.rms)
def modelSurfPoly(x,y,w):
    surf=w[0] + w[1]*x + w[2]*y + w[3]*x**2 + w[4]*x*y + w[5]*y**2 + w[6]*x**3 + w[7]*x**2*y     + w[8]*x*y**2 + w[9]*y**3 + w[10]*x**4 + w[11]*x**3*y + w[12]*x**2*y**2+ w[13]*x*y**3 + w[14]*y**4
    return surf

# Center and scale to have zero mean and unit variance

X = (crss_0-np.mean(crss_0))/np.std(crss_0)
Y = (h0-np.mean(h0))/np.std(h0)

poly = PolynomialFeatures(degree=4)
input_data = poly.fit_transform(np.transpose([X, Y]))

# Make the fit with ordinary least squares.
clf = LinearRegression()
clf.fit(input_data, rms)

print(clf.coef_)
print(clf.intercept_)
w = list(clf.coef_)
w2 = clf.intercept_
w[0]=w2
print(w)
ww=w


# # Creating 3D Surface 


def scale_crss(x):
    return (x-np.mean(crss_0))/np.std(crss_0)
def scale_h0(x):
    return (x-np.mean(h0))/np.std(h0)

x_grid = np.linspace(df_samp.crss.min(),df_samp.crss.max(),100)
y_grid = np.linspace(df_samp.h0.min(),df_samp.h0.max(),100)

X_gr, Y_gr = np.meshgrid(x_grid,y_grid)
Z = modelSurfPoly(scale_crss(X_gr),scale_h0(Y_gr), w)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax = fig.gca(projection='3d')
surf = ax.plot_surface( X_gr,Y_gr,Z , cmap='coolwarm',alpha=0.7, antialiased=False)
ax.scatter(crss_0,h0, df_samp['rms'], c='black')
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.set_xlabel("\u03C4$_{0}$ [MPa]", linespacing=3.2,fontsize=10)
ax.set_ylabel("\nh$_{0}$ [MPa]", linespacing=3.2,fontsize=10)
ax.set_zlabel('\nRMSE', linespacing=3.2,fontsize=10)
#ax.text2D(0.2, 0.95, "PL Steel", transform=ax.transAxes,fontsize=16)
ax.zaxis.set_major_formatter(FormatStrFormatter('%.e'))
ax.view_init(45,120)
#plt.draw()
#ax.figure.savefig("RSO_PL_2.2.png",dpi=200)
plt.show()


# # Particle Swarm Optimization


def modelSurfPoly(x):
    surf=ww[0] + ww[1]*x[0] + ww[2]*x[1] + ww[3]*x[0]**2 + ww[4]*x[0]*x[1] + ww[5]*x[1]**2     + ww[6]*x[1]**3 + ww[7]*x[1]**2*+ ww[8]*x[0]*x[1]**2 + ww[9]*x[1]**3 + ww[10]*x[0]**4     + ww[11]*x[0]**3*x[1] + ww[12]*x[0]**2*x[1]**2+ ww[13]*x[0]*x[1]**3 + ww[14]*x[1]**4
    return surf
nVar = 2
#numb of variables
ub=np.array([1.699850e+00,1.734199e+00])
lb=np.array([-1.716163e+00,-1.720669e+00])
#Upper bound & Lower bound
fobj = modelSurfPoly
#Function
noP=10
#Number of Particles
maxiter=900
#max iteration
wMax = 0.9
wMin = 0.2
#max and min weights
c1=2
#cognitive component
c2=2
#social component
vMax = (ub - lb)* 0.2
vMin  = -vMax

a={}
b={}
c={}
d={}
e={}
for k in range(noP):
    a["Swarm_Particles_X_"+str(k)]=(ub-lb)*np.random.uniform(size=nVar)+lb
    b["Swarm_Particles_V_"+str(k)]=(np.zeros(nVar))
    c["Swarm_Particles_PBest_X_"+str(k)]=np.zeros(nVar)
    d["Swarm_Particles_PBest_O_"+str(k)]=(np.infty)
Swarm_GBest_X=np.zeros(nVar)
Swarm_GBest_O=np.infty

for t in range(maxiter):
   
    for k in range(noP):
        print(Swarm_GBest_O)
        currentX =a["Swarm_Particles_X_"+str(k)]
        e["Swarm_Particles_0_"+str(k)] = fobj(currentX)
        
        if e["Swarm_Particles_0_"+str(k)]<d['Swarm_Particles_PBest_O_'+str(k)]:
            c['Swarm_Particles_PBest_X_'+str(k)]=currentX
            d['Swarm_Particles_PBest_O_'+str(k)]=e["Swarm_Particles_0_"+str(k)]

        if e["Swarm_Particles_0_"+str(k)]<Swarm_GBest_O:
            Swarm_GBest_X = currentX
            Swarm_GBest_O=e["Swarm_Particles_0_"+str(k)]

    w=wMax-t*((wMax-wMin)/maxiter)

    for k in range(noP):
         
        b['Swarm_Particles_V_'+str(k)] = w*b['Swarm_Particles_V_'+str(k)]         +c1*np.random.uniform(size=nVar)*(c['Swarm_Particles_PBest_X_'+str(k)]-a['Swarm_Particles_X_'+str(k)])         +c2*np.random.uniform(size=nVar)*(Swarm_GBest_X-a['Swarm_Particles_X_'+str(k)])
        
        a['Swarm_Particles_X_'+str(k)] =a['Swarm_Particles_X_'+str(k)]+b['Swarm_Particles_V_'+str(k)]
        
        indeX1=np.where(a['Swarm_Particles_X_'+str(k)]>ub)
        a['Swarm_Particles_X_'+str(k)][indeX1]=ub[indeX1]
        indeX2=np.where(a['Swarm_Particles_X_'+str(k)]<lb)
        a['Swarm_Particles_X_'+str(k)][indeX2]=lb[indeX2]                    
print(Swarm_GBest_O)
print(Swarm_GBest_X)
h0_PSO=Swarm_GBest_X[1]*np.std(h0)+np.mean(h0)
crss_PSO=Swarm_GBest_X[0]*np.std(crss_0)+np.mean(crss_0)
print('Results for h0 and crss_0 are respectively:',int(h0_PSO) ,'and', int(crss_PSO))

df_samp.describe()

# # Latin Hypercube Sampling 

xlimits = np.array([[h0_PSO-56,h0_PSO+56],[crss_PSO-12,crss_PSO+12]])
sampling = LHS(xlimits=xlimits,criterion='ese')
num = 30
x = sampling(num)
x =x.round(decimals=2)
plt.plot(x[:, 0], x[:, 1], ".")
plt.ylabel("\u03C4$_{0}$ [MPa]",fontsize=10)
plt.xlabel("h$_{0}$ [MPa]",fontsize=10)
plt.title('Latin Hypercube Sampling')
#plt.savefig('sro2.2_fix.png',dpi=400)
df=pd.DataFrame(x,columns=['h0','crss'])
df.to_csv('sro_blue_2.2_fix_last.csv',index=None)
df.head()
