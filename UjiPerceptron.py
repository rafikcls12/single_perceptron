#%%
import pandas as pd
kanker =  pd.read_csv('kanker.csv')
# %% Testing
import numpy as np
datasets = np.array(kanker)
Testing = pd.concat([kanker[0:10], kanker[150:160]]) # survived & died
Xtest = np.array(Testing[['Age_of_patient_at_time_of_operation','Number_of_positive_axillary_nodes_detected']])
Ytest = np.concatenate((datasets[0:10,3], datasets[150:160,3]))
# %% Training
Training = pd.concat([kanker[10:150], kanker[150:300]])
Xtrain = np.array(Training[['Age_of_patient_at_time_of_operation','Number_of_positive_axillary_nodes_detected']])
Ytrain = np.concatenate((datasets[10:150,3], datasets[150:300,3]))
# %%
from PerceptronTunggal import Perceptron

p = Perceptron(learning_rate=0.02, iterasi=1000)
p.train(Xtrain, Ytrain)
predictions= p.prediksi(Xtest)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)*100
    return accuracy
akurasi = accuracy(Ytest, predictions)
print(akurasi)
# %%
import matplotlib.pyplot as plt
Age = kanker[['Age_of_patient_at_time_of_operation']]
# Operation = Data[['Patient_year_of_operation']]
Nodes = kanker[['Number_of_positive_axillary_nodes_detected']]

plt.scatter(Xtrain[:, 0], Xtrain[:, 1], marker="o", edgecolor="black", c=Ytrain, s = 30, alpha=0.5)
plt.scatter(Xtest[:, 0], Xtest[:, 1], marker="x", c=Ytest, s = 100)
plt.xlabel = ("Age")
plt.ylabel = ("Nodes")

# x0_1 = np.amin(Xtrain[:, 1])
# x0_2 = np.amax(Xtrain[:, 0])

# x1_1 = (-p.weights[0]*x0_1 - p.bias)/p.weights[1]
# x1_2 = (-p.weights[0]*x0_2 - p.bias)/p.weights[1]

# plt.plot([x0_1, x0_2], [x1_1, x1_2], "r")
# %%
