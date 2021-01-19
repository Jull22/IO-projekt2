import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from time import perf_counter


from sklearn.neural_network import MLPClassifier

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras
from mlxtend.frequent_patterns import apriori, association_rules





df= pd.read_csv("Telco-customer-churn.csv")
del df["customerID"]  #usuwam ID

df.dtypes


df["SeniorCitizen"]=pd.Categorical(df["SeniorCitizen"])   #zmiana na kategoryczne
df["TotalCharges"]=pd.to_numeric(arg=df["TotalCharges"], errors= 'coerce')   #zmiana na float

df.isnull().sum()

df=df.fillna(0)   #NA wypełniam zerami

numerics = ['float64', 'int64']
numeric_ds = df.select_dtypes(include=numerics)
objects_ds = df.select_dtypes(exclude=numerics)

numeric_ds.describe()   

numeric_ds = pd.concat([numeric_ds,df["Churn"]],axis=1) #Add the 'Churn' variable to the numeric dataset

fig, ax = plt.subplots(figsize=(12,12))
y=numeric_ds["tenure"]
ax.hist(y)
ax.set_title("Tenure histogram",fontsize=20)
ax.set_xlabel("Amount of months that the customer has been with the company",fontsize=18)
ax.set_ylabel("Number of customers",fontsize=18)



fig, ax = plt.subplots(figsize=(12,12))

y=numeric_ds["MonthlyCharges"]
ax.hist(y)
ax.set_title("MonthlyCharges histogram", fontsize=20)
ax.set_xlabel("Monthly charge",fontsize=18)
ax.set_ylabel("Number of customers",fontsize=18)



fig, ax = plt.subplots(figsize=(12,12))

y=numeric_ds["TotalCharges"]
ax.hist(y)
ax.set_title("TotalCharges histogram", fontsize=20)
ax.set_xlabel("Total charge",fontsize=18)
ax.set_ylabel("Number of customers",fontsize=18)


objects_ds.describe().T


MonthlyCharges_bins=pd.cut(numeric_ds["MonthlyCharges"], bins=[0,35,60,130], labels=['low','medium','high'])
tenure_bins=pd.cut(numeric_ds["tenure"], bins=[0,20,60,80], labels=['low','medium','high'])
TotalCharges_bins=pd.cut(numeric_ds["TotalCharges"], bins=[0,1000,4000,10000], labels=['low','medium','high'])


fig,ax =plt.subplots(1,3,figsize=(19,5))
sns.countplot(x=tenure_bins, hue="Churn", data=numeric_ds, palette="Accent", ax=ax[0])                   
sns.countplot(x=MonthlyCharges_bins, hue="Churn", data=numeric_ds, palette="Accent", ax=ax[1])
sns.countplot(x=TotalCharges_bins, hue="Churn", data=numeric_ds, palette="Accent", ax=ax[2])
                                                                                            


bins=pd.DataFrame([tenure_bins, MonthlyCharges_bins, TotalCharges_bins]).T


fig,ax =plt.subplots(4,4,figsize=(15,15))
fig.subplots_adjust(hspace=.5)
for i in range(0,16):
    g = sns.countplot(x=objects_ds.iloc[:,i], hue=objects_ds["Churn"], ax=ax[divmod(i,4)], palette="coolwarm")
    g.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0.) if i==0 else g.legend_.remove()
for tick in ax[3,3].get_xticklabels():
    tick.set_rotation(45)

objects_ds


data=pd.concat([bins,objects_ds],axis=1)  # Concatenate bins with object variables
for i in list(data.columns):
    data[i] = pd.Categorical(data[i]) # Convert all the variables into categorical
dummy = pd.get_dummies(data) # Transform the categorical variables into binary vectors
dummy.T

features = dummy.drop(["Churn_Yes", "Churn_No"], axis=1).columns

X = dummy[features].values
Y = dummy["Churn_Yes"].values

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.3)


#  KNN k=3


from sklearn import metrics
st_time= perf_counter()
knn3 = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn3.fit(X_train, Y_train)

st_time= perf_counter()
test_pred= knn3.predict(X_validation)
end_time=perf_counter()
time_knn3=end_time-st_time
print("Time:", end_time-st_time)
accuracy_knn3=metrics.accuracy_score(Y_validation, test_pred)
print("Accuracy: ", accuracy_knn3)
print(confusion_matrix(Y_validation, test_pred))
plot_confusion_matrix(knn3,X_validation, Y_validation,cmap="PuBu")


#  KNN k=8


from sklearn import metrics

st_time=perf_counter()
knn8 = KNeighborsClassifier(n_neighbors=8, metric='euclidean')
knn8.fit(X_train, Y_train)

test_pred= knn8.predict(X_validation)
end_time=perf_counter()
time_knn8=end_time-st_time
print("Time:", end_time-st_time)

accuracy_knn8=metrics.accuracy_score(Y_validation, test_pred)
print("Accuracy: ", accuracy_knn8)
print(confusion_matrix(Y_validation, test_pred))
plot_confusion_matrix(knn8,X_validation, Y_validation,cmap="PuBu")


# # KNN k=20


from sklearn import metrics
st_time= perf_counter()
knn20 = KNeighborsClassifier(n_neighbors=20, metric='euclidean')
knn20.fit(X_train, Y_train)

test_pred= knn20.predict(X_validation)
end_time=perf_counter()
time_knn20=end_time-st_time
print("Time:", end_time-st_time)
accuracy_knn20=metrics.accuracy_score(Y_validation, test_pred)
print("Accuracy: ", accuracy_knn20)
print(confusion_matrix(Y_validation, test_pred))
plot_confusion_matrix(knn20,X_validation, Y_validation,cmap="PuBu")


#  Drzewo decyzyjne

DT =  DecisionTreeClassifier()
st_time= perf_counter()
DT.fit(X_train, Y_train)

test_pred= DT.predict(X_validation)
end_time=perf_counter()
time_DT=end_time-st_time
print("Time:", end_time-st_time)
accuracy_DT=metrics.accuracy_score(Y_validation, test_pred)
print("Accuracy: ", accuracy_DT)
print(confusion_matrix(Y_validation, test_pred))
plot_confusion_matrix(DT,X_validation, Y_validation,cmap="PuBu")


#  NAIVE BAYES


NB =  GaussianNB()
st_time= perf_counter()
NB.fit(X_train, Y_train)

test_pred= NB.predict(X_validation)

end_time=perf_counter()
time_NB=end_time-st_time
print("Time:", end_time-st_time)

accuracy_NB=metrics.accuracy_score(Y_validation, test_pred)

print("Accuracy:", accuracy_NB)
print(confusion_matrix(Y_validation, test_pred))
plot_confusion_matrix(NB,X_validation, Y_validation,cmap="PuBu")


#  Artificial network-  multi-layer perceptron

st_time= perf_counter()
ann = MLPClassifier(hidden_layer_sizes=(20,40,20),max_iter=900,activation='relu',
                    solver='adam',alpha=0.0001,random_state=42).fit(X_validation,Y_validation)

test_pred= ann.predict(X_validation)

end_time=perf_counter()
time_ann=end_time-st_time
print("Time:", end_time-st_time)
accuracy_MLP= accuracy_score(Y_validation, test_pred)
print("Accuracy:", accuracy_MLP)
print(confusion_matrix(Y_validation, test_pred))

plot_confusion_matrix(ann,X_validation, Y_validation,cmap="PuBu")


#  Random forest


RF = RandomForestClassifier(n_estimators=100).fit(X_validation,Y_validation)
st_time= perf_counter()

test_pred= RF.predict(X_validation)
end_time=perf_counter()
time_RF=end_time-st_time
print("Time:", end_time-st_time)
accuracy_RF= accuracy_score(Y_validation, test_pred)
print("Accuracy:", accuracy_RF)
print(confusion_matrix(Y_validation, test_pred))
plot_confusion_matrix(RF,X_validation, Y_validation,cmap="PuBu")


# # SVM


from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=1)
st_time= perf_counter()
svm.fit(X_train, Y_train)

test_pred= svm.predict(X_validation)

end_time=perf_counter()
time_svm=end_time-st_time
print("Time:", end_time-st_time)

accuracy_svm= accuracy_score(Y_validation, test_pred)
print("Accuracy:", accuracy_svm)
print(confusion_matrix(Y_validation, test_pred))
plot_confusion_matrix(svm,X_validation, Y_validation,cmap="PuBu")


# # Gradient boosting


gbc = GradientBoostingClassifier(n_estimators=100).fit(X_validation,Y_validation)
st_time= perf_counter()
test_pred= gbc.predict(X_validation)
end_time=perf_counter()
time_gbc=end_time-st_time
print("Time:", end_time-st_time)
accuracy_gbc= accuracy_score(Y_validation, test_pred)
print("Accuracy:", accuracy_gbc)
print(confusion_matrix(Y_validation, test_pred))
plot_confusion_matrix(gbc,X_validation, Y_validation,cmap="PuBu")


# # Neural network 


model = keras.Sequential()
model.add(keras.layers.Dense(100,activation="selu",kernel_initializer= "he_normal",input_dim = X_train.shape[1]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(52,activation="selu",kernel_initializer= "he_normal"))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.Dense(20,activation="selu",kernel_initializer= "he_normal"))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(2,activation="softmax"))


model.compile(optimizer='adam',loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.summary()



st_time=perf_counter()
history = model.fit(X_train, Y_train, epochs=20, batch_size= 50, validation_split=0.2)
end_time=perf_counter()
time_NN=end_time-st_time
print(time_NN)

score1 = model.evaluate(x=X_validation, y=Y_validation)

print('Loss:', score1[0])
print('Accuracy:', score1[1])
accuracy_NN=score1[1]


predictions = model.predict(X_validation)
predictions
y_pred = (predictions > 0.5)
matrix = metrics.confusion_matrix(Y_validation, y_pred.argmax(axis=1))
matrix

pd.DataFrame(history.history).plot(figsize=(10, 7))
plt.grid(True)
plt.show()


# # CZAS


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize =(23, 15)) 
klasyfikator = ['Naive Bayes', 'KNN3', 'KNN8', 'KNN20', 'Drzewo decyzyjne', "Multi-layer perceptron", "Random Forest" , "NN", "Gradient Boosting", "Support Vector Machines"]
result = [time_NB, time_knn3, time_knn8, time_knn20,time_DT, time_ann,time_RF, time_NN, time_gbc, time_svm]

x = np.arange(len(klasyfikator))  # the label locations
width = 0.7  # the width of the bars
rects= ax.bar(klasyfikator, result, width , color='maroon')

for tick in ax.get_xticklabels():
    tick.set_rotation(45)
    
for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.4f}s'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 6),  
                    textcoords="offset points",
                    ha='center', va='center', fontsize=18)
# plt.rc('xtick', labelsize=20)
ax.set_title('Czas', fontsize= 30)
plt.xlabel("Klasyfikatory", fontsize=18) 
plt.ylabel("Sekundy", fontsize=18) 
plt.show()


# # DOKŁADNOŚCI


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize =(23, 15)) 
klasyfikator = ['Naive Bayes', 'KNN3', 'KNN8', 'KNN20', 'Drzewo decyzyjne', "Multi-layer perceptron", "Random Forest", "NN", "Gradient Boosting","Support Vector Machines"]
result = [accuracy_NB,accuracy_knn3,accuracy_knn8,accuracy_knn20,accuracy_DT,accuracy_MLP,accuracy_RF, accuracy_NN, accuracy_gbc, accuracy_svm]

x = np.arange(len(klasyfikator))  # the label locations
width = 0.7  # the width of the bars
rects= ax.bar(klasyfikator, result, width , color='maroon')

for tick in ax.get_xticklabels():
    tick.set_rotation(45)
    
for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}%'.format(height*100),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 6),  
                    textcoords="offset points",
                    ha='center', va='center', fontsize=23)


plt.rc('xtick', labelsize=20)

ax.set_title('Dokładności klasyfikatorów', fontsize= 30)
plt.xlabel("Klasyfikatory", fontsize=18) 
plt.ylabel("Wynik", fontsize=18) 
plt.show()


# # ASSOCIATION

apriori(dummy, min_support=0.5, use_colnames=True, max_len=None, verbose=0, low_memory=False)
freq_items = apriori(dummy, min_support=0.6, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.7)







