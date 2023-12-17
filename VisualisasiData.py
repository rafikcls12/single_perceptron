#%% Load Data
import pandas as pd
Data = pd.read_csv('kanker.csv')
# %% Data Testing   
SurviveTest = Data[0:50] #Survive
DiedTest = Data[150:200] #Died
Testing = pd.concat([SurviveTest,DiedTest])
# %% Data Training
SurviveTrain = Data[50:150] #Survive
DiedTrain = Data[200:300] #Died
Training = pd.concat([SurviveTrain,DiedTrain])
# %% Menampilkan Kolom
Age = Data[['Age_of_patient_at_time_of_operation']]
# Operation = Data[['Patient_year_of_operation']]
Nodes = Data[['Number_of_positive_axillary_nodes_detected']]
TargetClass = Data['Survival_status']
# %% Visualisasi 
import matplotlib.pyplot as plt
import numpy as np

colormap = np.array(['r','g','b'])
plt.scatter(Age,Nodes, c = colormap[TargetClass], edgecolors="k")
plt.xlabel = ("Age")
# plt.ylabel = ("Operation")
plt.ylabel = ("Nodes")
# %%
import seaborn as sns
sns.pairplot(Data, hue='Survival_status', vars=['Age_of_patient_at_time_of_operation', 'Number_of_positive_axillary_nodes_detected'])
# sns.pairplot(Data, hue='Survival_status', vars=['Age_of_patient_at_time_of_operation', 'Patient_year_of_operation','Number_of_positive_axillary_nodes_detected'])
# %%
import plotly.express as px
px.scatter_3d(Data, x = 'sepal length (cm)',y = 'sepal width (cm)', z = 'petal width (cm)')
# %%
