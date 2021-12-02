#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
import random
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
from sklearn.metrics import r2_score
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import KFold
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import seaborn as sns
import math


d = {'Mod_Eval': [], 'Var_Scr': [], 'r2': [], 'RMSE': [], 'mean': "" , 'var': ""}
test = pd.DataFrame(data=d)
test.to_csv('test.csv',index=False)
for x in range(40):
    #Predict force as an output
    os.chdir(r'C:\Users\Orkun\OneDrive\Masaüstü\Master\Blue_NN\Blue_NN_Test')
    df=pd.DataFrame(pd.read_csv('ANN_Blue_Frame.csv'))
    df=df[df['Force [N]']<0.01]
    y=df['Force [N]']
    X=df.drop('Force [N]',axis=1)
    #Split data into test/train set and scale them
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33,shuffle=True)
    X_test.to_csv('X_test_'+str(x).zfill(2)+'.csv',index=False)
    y_train.to_csv('y_train_'+str(x).zfill(2)+'.csv',index=False)
    y_test.to_csv('y_test_'+str(x).zfill(2)+'.csv',index=False)
    X_train.to_csv('X_train_'+str(x).zfill(2)+'.csv',index=False)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train= scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    #Creating Model
    model = Sequential()
    model.add(Dense(5, activation='relu',kernel_initializer='normal'))
    model.add(Dense(64, activation='relu',kernel_initializer='normal'))
    model.add(Dense(32, activation='relu',kernel_initializer='normal'))
    model.add(Dense(1,kernel_initializer='normal'))
    model.compile(loss="mean_squared_error",optimizer="adam")
    #Callback/EarlyStopping in order to avoid overfitting
    es = EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='auto',restore_best_weights=True)
    #Train model
    model.fit(X_train, y_train.values,epochs=4000,batch_size=128,verbose=0,callbacks=[es],
              validation_data=(X_test, y_test.values))
    #Evaluation
    Mod_Eval = model.evaluate(X_test, y_test,verbose=0, batch_size=128)
    predictions = model.predict(X_test)
    Var_Scr= explained_variance_score(y_test,predictions)
    r2 =r2_score(y_test, predictions)
    mean = scaler.mean_
    var= scaler.var_
    RMSE=math.sqrt(mean_squared_error(y_test, predictions))
    test.loc[0] = [Mod_Eval]+[Var_Scr]+[r2]+[RMSE]+[mean]+[var]
    model.save("model_" + str(x).zfill(2) + ".h5")
    #Graph
    axes = plt.figure()
    # True Values
    plt.plot(y_test,y_test,label='True Value',color='black',lw=5)
    # Our predictions
    plt.scatter(y_test,predictions,label='AI Prediction',alpha=0.5)
    plt.legend(loc="upper left")
    plt.xlabel('Force [N]')
    plt.ylabel('Force [N]')
    #plt.xlim(0,0.015)
    #plt.ylim(0,0.012)
    axes.savefig("model_" + str(x).zfill(2) + ".png",dpi=800)
    test.to_csv('test.csv',mode='a',index=False,header=False)


# In[3]:


x=1


# In[5]:


str(x).zfill(4)


# In[ ]:




