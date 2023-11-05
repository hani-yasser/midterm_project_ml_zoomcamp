#!/usr/bin/env python
# coding: utf-8

# # **. Import Libraries:**

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report,confusion_matrix
import copy




C = 1.0
n_splits = 5
output_file = f'model_C={C}.bin'


# # **. Load Dataset:**

df = pd.read_csv('heart.csv')


# # **. Data Exploration:**

df.info()


# All Colimns are Numeric type

df.describe().T


# Statics shows the count of each column along with their mean value, standard deviation, minimum and maximum values.

# Display first 5 rows of the dataset


df.head()


# # **Checking Missing Value**


df.isnull().sum()


# No column has any missing value

# There only Binary Classification (0 and 1)

# # Heat Map and importance of features
# ## 

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.set(font_scale=1.2)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=.5)
plt.title("Correlation Heatmap")
plt.show()

# Calculate the correlation coefficients between features and the label 'output'
correlation_with_output = df.corr()['target'].drop('target')

# Plotting using Seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation_with_output.index, y=correlation_with_output.values, palette="mako")

# Adding labels and title
plt.xlabel("Features")
plt.ylabel("Correlation with Output")
plt.title("Correlation between Features and Target")

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)

# Display the plot
plt.show()


# # **Plot chances of heart attack**

import seaborn as sns
import matplotlib.pyplot as plt

# Create the countplot
ax = sns.countplot(x="target", data=df, palette="cubehelix")

# Calculate the number of occurrences for each category
countNoDisease = len(df[df.target == 0])
countHaveDisease = len(df[df.target == 1])

# Set labels for the columns
ax.set_xticklabels(["No Heart Disease", "Heart Disease"])

# Show the plot
plt.show()

# Rest of your code
df.target.value_counts()
print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / len(df.target)) * 100))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / len(df.target)) * 100))
df.target.value_counts()


# # **Check gender distribution for the study**

sns.countplot(x='sex', data=df, palette="mako_r")
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()
df.sex.value_counts()


# # Correlation between sex and target

# Group and count data
grouped = df.groupby('target')['sex'].value_counts()

# Convert grouped data to a DataFrame for easy plotting
grouped_df = grouped.unstack()

# Map target labels
target_mapping = {0: "No Disease", 1: "Have Disease"}
grouped_df.index = grouped_df.index.map(target_mapping)

# Plotting using Seaborn
sns.set(style="whitegrid")
ax = grouped_df.plot(kind="bar", stacked=True, colormap="viridis")

# Adding labels and title
plt.xlabel("Target")
plt.ylabel("Count")
plt.title("Counts of Sex by Target")

# Adding annotations for each bar
for i, col in enumerate(grouped_df.columns):
    for j, value in enumerate(grouped_df[col]):
        ax.text(i, value, str(value), ha='center', va='bottom', fontsize=10)

# Modify the legend labels
plt.legend(["Female", "Male"])

# Display the plot
plt.show()


# # Heart Desease VS Age


pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease VS Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# # **Correlation between SLP and output**

pd.crosstab(df.slope,df.target).plot(kind="bar",figsize=(15,6),color=['#DAF7A6','#FF5733' ])
plt.title('Heart Disease VS Speech and Language Disorders')
plt.xlabel(' Slope ')
plt.xticks(rotation = 0)
plt.ylabel('Frequency')
plt.show()


# 
# # **Correlation between FBS and output bold text**

pd.crosstab(df.fbs,df.target).plot(kind="bar",figsize=(15,6),color=['#00FF00','#FF0000' ])
plt.title('Heart Disease Frequency VS FBS')
plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation = 0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency of Disease')
plt.show


# # **Correlation between Chest pain type and output**


pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(15,6),color=['#800080','#FFA500' ])
plt.title('Heart Disease VS Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation = 0)
plt.ylabel('Frequency of Disease ')
plt.show()


# Calculating and Sorting Correlations with Target Variable in a DataFrame


correlation_df = df.corrwith(df['target']).to_frame('correlation')

# Sort the DataFrame by the 'correlation' column in descending order
correlation_df_sorted = correlation_df.sort_values(by='correlation', ascending=False)

# Display the sorted DataFrame
print(correlation_df_sorted)


# ## Split data as 60%  will be train data and 20% will be Data_Validation and 20% test data.


from sklearn.model_selection import train_test_split


df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)


df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=11)


len(df_train),len(df_val),len(df_test),len(df_train_full)


y_train = df_train.target.values
y_val = df_val.target.values
y_test = df_test.target.values


del df_train['target']
del df_val['target']
del df_test['target']


df.columns


numerical = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']


len(y_train),len(y_val),len(y_test)


# ##  Train Logestic Regression

from sklearn.linear_model import LogisticRegression


model = LogisticRegression(solver='liblinear', random_state=1)


X_train = df_train
X_val = df_val
X_test = df_test



X_test


model.fit(X_train, y_train)


model.predict_proba(X_val)

y_pred = model.predict_proba(X_val)[:, 1]


target = y_pred > 0.55

(y_val == target).mean()


# ### AUC for Data Validation

y_pred = model.predict_proba(X_val)[:, 1]

auc = roc_auc_score(y_val, y_pred)
auc


# ### AUC for Data test

y_pred_LR_test = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_LR_test)
auc


# ### Train Logestic Regression

thresholds = np.linspace(0, 1, 21)

scores = []

for t in thresholds:
    score = accuracy_score(y_val, y_pred >= t)
    print('%.2f %.3f' % (t, score))
    scores.append(score)


plt.plot(thresholds, scores)


# ### Hyper parameter tunin usining Cross Validation

# ### Define Train and Predict functions

def train1(df_train, y_train, C=1.0):
    X_train= df_train[numerical]
    
    model = LogisticRegression(C=C, max_iter=5000)
    model.fit(X_train, y_train)
    
    return  model


def predict1(df,model):
    X = df[numerical]

    
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred



from tqdm.auto import tqdm

from sklearn.model_selection import KFold

len(df_train),len( y_train) , len(df_val),len(y_val)


n_splits = 5

for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    scores = []

    for train_idx, val_idx in kfold.split(df_train_full):
        df_train = df_train_full.iloc[train_idx]
        df_val = df_train_full.iloc[val_idx]

        y_train = df_train.target.values
        y_val = df_val.target.values

        model = train1(df_train, y_train, C=C)
        y_pred = predict1(df_val, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

scores


# ### Train final model for logestic regression

model = train1(df_train, y_train, C=10.0)
y_pred_LR = predict1(df_val, model)

auc = roc_auc_score(y_val, y_pred_LR)
auc

len(df_test),len(y_test)


model = train1(df_train, y_train, C=10.0)
y_pred_LR = predict1(df_test, model)

auc = roc_auc_score(y_test, y_pred_LR)
auc


# ## Prepare Data for Deceion tree

df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)


df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=11)


y_train = df_train.target.values
y_val = df_val.target.values
y_test = df_test.target.values


del df_train['target']
del df_val['target']
del df_test['target']


len(df_train), len(df_val) ,len(df_test) , len(y_train), len(y_val) ,len(y_test)


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()


X_train= df_train
X_val=df_val
X_test=df_test




# ## Train the Decision Tree classifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)



y_pred = dt.predict_proba(X_train)[:, 1]
roc_auc_score(y_train, y_pred)


y_pred_dt = dt.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, y_pred_dt)


y_pred = dt.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, y_pred)


# ## Prepare Data for Random Forest


df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)


df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=11)


y_train = df_train.target.values
y_val = df_val.target.values
y_test = df_test.target.values


del df_train['target']
del df_val['target']
del df_test['target']


X_train= df_train
X_val=df_val
X_test=df_test


# ## Random Forest Classifier Before tuning

rf = RandomForestClassifier(n_estimators=1, random_state=3)
rf.fit(X_train, y_train)
y_pred = rf.predict_proba(X_val)[:, 1]

roc_auc_score(y_val, y_pred)


y_pred_rf_test = rf.predict_proba(X_test)[:, 1]
roc_auc_score(y_val, y_pred_rf_test)


# ## Tuning parameters

# ### Number of estimators required (Number of trees)

aucs = []

for i in range(10, 201, 10):
    rf = RandomForestClassifier(n_estimators=i, random_state=3)
    rf.fit(X_train, y_train)
    y_pred = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print('%s -> %.3f' % (i, auc))
    aucs.append(auc)


# #### n_estimator =60 is the best value


plt.figure(figsize=(6, 4))

plt.plot(range(10, 201, 10), aucs, color='black')
plt.xticks(range(0, 201, 50))

plt.title('Number of trees vs AUC')
plt.xlabel('Number of trees')
plt.ylabel('AUC')

# plt.savefig('ch06-figures/06_random_forest_n_estimators.svg')

plt.show()


# ### Tuninig the max_depth parameter:


all_aucs = {}

for depth in [5, 10, 20]:
    print('depth: %s' % depth)
    aucs = []

    for i in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=i, max_depth=depth, random_state=1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        print('%s -> %.3f' % (i, auc))
        aucs.append(auc)
    
    all_aucs[depth] = aucs
    print()


plt.figure(figsize=(6, 4))

num_trees = list(range(10, 201, 10))

plt.plot(num_trees, all_aucs[5], label='depth=5', color='black', linestyle='dotted')
plt.plot(num_trees, all_aucs[10], label='depth=10', color='black', linestyle='dashed')
plt.plot(num_trees, all_aucs[20], label='depth=20', color='black', linestyle='solid')
    
plt.xticks(range(0, 201, 50))
plt.legend()

plt.title('Number of trees vs AUC')
plt.xlabel('Number of trees')
plt.ylabel('AUC')

# plt.savefig('ch06-figures/06_random_forest_n_estimators_depth.svg')

plt.show()


# ### Tuning the min_samples_leaf parameter

all_aucs = {}

for m in [3, 5, 10]:
    print('min_samples_leaf: %s' % m)
    aucs = []

    for i in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=i, max_depth=10, min_samples_leaf=m, random_state=1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        print('%s -> %.3f' % (i, auc))
        aucs.append(auc)
    
    all_aucs[m] = aucs
    print()


plt.figure(figsize=(6, 4))

num_trees = list(range(10, 201, 10))

plt.plot(num_trees, all_aucs[3], label='min_samples_leaf=3', color='black', linestyle='dotted')
plt.plot(num_trees, all_aucs[5], label='min_samples_leaf=5', color='black', linestyle='dashed')
plt.plot(num_trees, all_aucs[10], label='min_samples_leaf=10', color='black', linestyle='solid')
    
plt.xticks(range(0, 201, 50))
plt.legend()

plt.title('Number of trees vs AUC')
plt.xlabel('Number of trees')
plt.ylabel('AUC')

# plt.savefig('ch06-figures/06_random_forest_n_estimators_sample_leaf.svg')

plt.show()


# ### Training the final model:


rf_final = RandomForestClassifier(n_estimators=10, max_depth=5, min_samples_leaf=3, random_state=1)
rf_final.fit(X_train, y_train)


y_pred_rf = rf_final.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, y_pred_rf)


y_pred_rf_test = rf_final.predict_proba(X_test)[:, 1]
roc_auc_score(y_val, y_pred_rf_test)


from sklearn.metrics import roc_curve


plt.figure(figsize=(5, 5))


fpr, tpr, _ = roc_curve(y_val, y_pred_rf)
plt.plot(fpr, tpr, color='red')

fpr, tpr, _ = roc_curve(y_val, y_pred_dt)
plt.plot(fpr, tpr, color='blue', linestyle='dashed')

plt.plot([0, 1], [0, 1], color='black', lw=0.7, linestyle='dashed', alpha=0.5)

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('ROC curve')

plt.show()


## XGBoost

import xgboost as xgb


dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=None)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=None)


xgb_params = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'nthread': 8,
    'seed': 1
}


model = xgb.train(xgb_params, dtrain, num_boost_round=10)


y_pred = model.predict(dval)
y_pred[:10]


roc_auc_score(y_val, y_pred)


watchlist = [(dtrain, 'train'), (dval, 'val')]


xgb_params = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1
}


# # #%%capture output

# # model = xgb.train(xgb_params, dtrain,
# #                  num_boost_round=100,
# #                   evals=watchlist, verbose_eval=5)
# import sys
# from contextlib import redirect_stdout

# # ...

# # Capture the output of the XGBoost training
# with open('xgboost_output.txt', 'w') as f:
#     with redirect_stdout(f):
#         model = xgb.train(xgb_params, dtrain,
#                           num_boost_round=100,
#                           evals=watchlist, verbose_eval=5)

# # ...



# # In[274]:


# def parse_xgb_output(output):
#     tree = []
#     aucs_train = []
#     aucs_val = []

#     for line in output.stdout.strip().split('\n'):
#         it_line, train_line, val_line = line.split('\t')

#         it = int(it_line.strip('[]'))
#         train = float(train_line.split(':')[1])
#         val = float(val_line.split(':')[1])

#         tree.append(it)
#         aucs_train.append(train)
#         aucs_val.append(val)

#     return tree, aucs_train, aucs_val


# # In[275]:


# tree, aucs_train, aucs_val = parse_xgb_output(output)


# # In[276]:


# plt.figure(figsize=(6, 4))

# plt.plot(tree, aucs_train, color='black', linestyle='dashed', label='Train AUC')
# plt.plot(tree, aucs_val, color='black', linestyle='solid', label='Validation AUC')
# plt.xticks(range(0, 101, 25))


# plt.legend()

# plt.title('XGBoost: number of trees vs AUC')
# plt.xlabel('Number of trees')
# plt.ylabel('AUC')

# # plt.savefig('ch06-figures/06_xgb_default.svg')

# plt.show()


# # In[277]:


# len(df_train) , len(y_test)


# # In[278]:


#training the final model

print('training the final model')
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_pred = dt.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred)
auc

print(f'auc={auc}')


#Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((model), f_out)

print(f'the model is saved to {output_file}')


# ### A variety of models were leveraged in the training process, supplemented with rigorous hyperparameter tuning methodologies. The performance of these models was rigorously evaluated based on key metrics AUC (Area Under Curve) , it emerged that the Decesion tree model after second round of parameter tuning  outperformed as compared to other model as AUC reached 97%
