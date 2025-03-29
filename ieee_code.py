"""
IEEE Workshop
Christos Andrianos 
"""

#------------------Import Libraries-------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from scipy import stats
import seaborn as sns

#-----------------------------------------------------

#--------------------Load Dataset---------------------

file="heart_disease.csv"

print("Dataset Information:")
data=pd.read_csv(file)
print(data.info())

print("\nMissing values per column:")
print(data.isnull().sum())

#-----------------------------------------------------

#------------------Dataset Cleaning-------------------

def clean(data):
    
    data = data.copy()
    data = data.dropna()
    
    data['Gender'] = data['Gender'].replace({'M':0, 'F':1})
    data['ChestPainType'] = data['ChestPainType'].replace({
        'TA':0, 'ATA':1, 'NAP':2, 'ASY':3 })
    data['RestingECG'] = data['RestingECG'].replace({
        'Normal':0, 'ST':1, 'LVH':2})
    data['ExerciseAngina'] = data['ExerciseAngina'].replace({
        'Y':0, 'N':1})
    data['ST_Slope'] = data['ST_Slope'].replace({
        'Up':0, 'Flat':1, 'Down':2 })
    
    return data

data=clean(data)
print(data.info())
print(data['Gender'])

#-----------------------------------------------------

#-----------------Data Transformation-----------------

labels=data['HeartDisease']
labels=labels.astype(int)
labels=np.array(labels)
#print("\n")
#print(labels)


data=data.drop(columns=['HeartDisease'])
Names=data.columns
Names=np.array(Names)

print("\nNames of features:")
print(Names)

X = np.array(data)
print(X.shape)

#-----------------------------------------------------

#-----------Data Distribution Visualization-----------

low_risk=0
high_risk=0

for i in range(len(labels)):
    if labels[i]==0:
        low_risk=low_risk+1
    
    elif labels[i]==1:
        high_risk=high_risk+1

classes=['Low-Risk', 'High-Risk']
risk_values=[low_risk,high_risk]

plt.bar(classes, risk_values, color=['#42A5F5'], width=0.6)
plt.title('Distribution of Classes')
plt.xlabel('Risk Level')
plt.ylabel('Number of Individuals')
plt.show()

#-----------------------------------------------------

#------------------Balance Restoration----------------

undersample = RandomUnderSampler(sampling_strategy='majority', random_state=40)
X, y =undersample.fit_resample(X, labels)

low_risk_new=0
high_risk_new=0

for i in range(len(y)):
    if y[i]==0:
        low_risk_new=low_risk_new+1
    
    elif y[i]==1:
        high_risk_new=high_risk_new+1

classes=['Low-Risk', 'High-Risk']
risk_values_new=[low_risk_new, high_risk_new]

plt.bar(classes, risk_values_new, color=[ '#0D47A1'], width=0.6)
plt.title('Distribution after Undersampling')
plt.xlabel('Risk Level')
plt.ylabel('Number of Individuals')
plt.show()

#-----------------------------------------------------

#------------------Data Normalization-----------------

from sklearn import preprocessing
X=preprocessing.normalize(X,axis=0)

#-----------------------------------------------------

#-----------------Feature Correlation-----------------

X_df=pd.DataFrame(data)
corr_matrix = X_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Correlation Matrix of Features')
plt.show()

#-----------------------------------------------------

#------------------Feature Selection------------------

def perform_statistical_test(class1, class2, test_type=0):
    
    if test_type == 0:
        _, p_value = stats.ttest_ind(class1, class2, equal_var=True)
    elif test_type == 1:
        _, p_value = stats.mannwhitneyu(class1, class2)
    else:
        raise ValueError(f"Unsupported test_type: {test_type}. Use 0 (t-test) or 1 (Mann-Whitney U).")

    return p_value
    

print('\nNumber of features:')
print(len(Names))

group0=[]
group1=[]

for i in range(len(X)):
    if y[i]==0:
        group0.append(X[i])
        
    elif y[i]==1:
        group1.append(X[i])

group0 = np.array(group0)
group1 = np.array(group1)

#   --> 0: T-Test --> 1:Mann-Whitney Test
test_type=1

selected_features = []
featsSSD=[]
names_featsSSD=[]
ic=0
    
for j in range(X.shape[1]):
    
    x0= group0[:,j]
    x1= group1[:,j]
    
    p = perform_statistical_test(x0, x1, test_type)
    

    if p <= 1e-15 :
        featsSSD.append(j)
        names_featsSSD.append(Names[j])
        ic += 1  # Αύξηση μετρητή για στατιστικά σημαντικά χαρακτηριστικά
        selected_features.append(j)
        
        fz = 8
        fig = plt.figure(figsize=(fz, fz))
        plt.tick_params(labelsize=16)
        
        bp = plt.boxplot([x0, x1], labels=['Low Risk', 'High Risk'], patch_artist=True)
        
        colors = ['#42A5F5', '#EF5350']  # Blue for Low Risk, Red for High Risk
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
                
        # Τίτλος με το p-value
        sTitle = ('Feature: "%s", \np = %4.3e' % (Names[j], p))
        plt.ylabel(f'Values of feature: "{Names[j]}"', fontsize=20)
        print(sTitle)
        plt.title(sTitle, fontsize=20)

X_sel = X[:, selected_features]
print(X_sel.shape)

print("\nSelected Features based on p-value:")
print(names_featsSSD)
print("Shape of X_sel:", X_sel.shape)

#-----------------------------------------------------

#-------------------Dataset Split---------------------

from sklearn.model_selection import train_test_split

def data_split(X, y, test_size=0.25, random_state=40):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = data_split(X_sel, y, test_size=0.25, random_state=40)

print(X_train.shape)

#-----------------------------------------------------

#-------------------Model Training--------------------

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

model=0

if model==0:
    model= KNeighborsClassifier(n_neighbors=5)

elif model==1:
    
    model = RandomForestClassifier(random_state=40)

elif model==2:
    model = Perceptron(tol=1e-3, random_state=40)

elif model == 3:
    model = SVC(kernel='linear', random_state=40)

elif model == 4:
    model = GaussianNB()

elif model == 5:
    model = LogisticRegression(random_state=40, max_iter=1000) 
    
model.fit(X_train, y_train)
y_pred_rf = model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Model Accuracy: {accuracy_rf:.4f}")

def plot_confusion_matrix(y_test, y_pred_rf):
    cm = confusion_matrix(y_test, y_pred_rf)
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

plot_confusion_matrix(y_test, y_pred_rf)

#-----------------------------------------------------

#------------------Exaustive Search-------------------

from itertools import combinations

def exhaustive_search(X, y, selected_features, model, max_comb):
    
    best_accuracy = 0
    best_features = []
    best_model = []
    
    for i in range(1, max_comb + 1):
        for comb in combinations(range(X.shape[1]), i):
            
            X_subset = X[:, comb]
            
            X_train, X_test, y_train, y_test = data_split(X_subset, y, test_size=0.25, random_state=40)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Tested combination: {comb}, Accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_features = comb
                best_model = model
                best_y_test = y_test
                best_y_pred = y_pred

    print("\nBest combination of features:", best_features)
    print(f"Best Accuracy: {best_accuracy:.4f}")
    plot_confusion_matrix(best_y_test, best_y_pred)
            
    return best_features, best_accuracy, best_model

best_features, best_accuracy, best_model = exhaustive_search(X_sel, y, selected_features, model, max_comb=5)

