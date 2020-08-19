import warnings
warnings.filterwarnings("ignore")
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import math
from sklearn.impute import SimpleImputer
from scipy.stats import randint as sp_randint
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from prettytable import PrettyTable
from sklearn.calibration import CalibratedClassifierCV

# %% [markdown]
# ## 3.1 Train Data

# %%
train = pd.read_csv("aps_failure_training_set.csv", na_values="na")
train.head()

# %%
train.describe()


# %%
train['class'].value_counts()

# %% [markdown]
# **This is a highly imbalanced dataset.**

# %%
# Checking missing values
train.isnull().any().value_counts()

# %%
# Checking missing values
train.isnull().sum(axis = 0)

# %%
y_pos = np.arange(2) 
# Create bars
plt.bar(y_pos, list(train['class'].value_counts()))
 
# Create names on the x-axis
plt.xticks(y_pos, ["Negative", "Positive"])
 
# Show graphic
plt.show()

# %%
sns.distplot(train['aa_000'])

# %%
sns.distplot(train['ag_001'])

# %%
sns.distplot(train['ag_002'])

# %% [markdown]
# ## 3.2. Test Data

# %%
test = pd.read_csv("aps_failure_test_set.csv", na_values="na")
test.head()

# %%
test.describe()

# %%
test['class'].value_counts()

# %% [markdown]
# **This is a highly imbalanced dataset.**

# %%
# Checking missing values
test.isnull().any().value_counts()

# %%
# Checking missing values
test.isnull().sum(axis = 0)

# %%
y_pos = np.arange(2) 
# Create bars
plt.bar(y_pos, list(test['class'].value_counts()))
 
# Create names on the x-axis
plt.xticks(y_pos, ["Negative", "Positive"])
 
# Show graphic
plt.show()

# %%
sns.distplot(test['aa_000'])

# %%
sns.distplot(test['ag_001'])

# %%
sns.distplot(test['ag_002'])

# %% [markdown]
#  - Here we have two many columns i.e. 170 columns 
#  - We cant visualize 170 columns.
#  - From columns's distributions we can't interpreat any useful info.
# %% [markdown]
# # 4. Preprocessing Data

# %%
# mapping class column pos to 1 and neg to -1
train['class'] = train['class'].apply(lambda x: 1 if x=='pos' else 0)

# %%
# mapping class column pos to 1 and neg to -1
test['class'] = test['class'].apply(lambda x: 1 if x=='pos' else 0)

# %%
print(train['class'].value_counts())
print(test['class'].value_counts())

# %%
y_train = train[['class']]
train = train.drop(['class'], axis=1)

# %%
y_test = test[['class']]
test = test.drop(['class'], axis=1)

# %%
train, cv, y_train, y_cv = train_test_split(train, y_train, stratify=y_train, test_size=0.15, random_state=42)

# %%
print(y_train['class'].value_counts())
print(y_cv['class'].value_counts())
print(y_test['class'].value_counts())

# %%
y_train.to_csv('y_train.csv', index=False)
y_cv.to_csv('y_cv.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# %% [markdown]
# # 3.2. Impute Missing Data
# %% [markdown]
# ### Impute technique used:
#  - Mean Impute

# %% [markdown]
# ### 3.2.1. Mean Impute

# %%
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit(train)
train_mean = pd.DataFrame(imp_mean.transform(train), columns=train.columns)
cv_mean = pd.DataFrame(imp_mean.transform(cv), columns=train.columns)
test_mean = pd.DataFrame(imp_mean.transform(test), columns=test.columns)

# %% [markdown]
# train_mean.isnull().any().value_counts()

# %%
cv_mean.isnull().any().value_counts()

# %%
test_mean.isnull().any().value_counts()

# %%
train_mean.to_csv('train_mean.csv', index=False)
train_mean = pd.read_csv('train_mean.csv')

cv_mean.to_csv('cv_mean.csv', index=False)
cv_mean = pd.read_csv('cv_mean.csv')

test_mean.to_csv('test_mean.csv', index=False)
test_mean = pd.read_csv('test_mean.csv')

train_mean.head(2)

# %%
scaler = StandardScaler().fit(train_mean)
train_mean_std = pd.DataFrame(scaler.transform(train_mean), columns=train.columns)
cv_mean_std = pd.DataFrame(scaler.transform(cv_mean), columns=train.columns)
test_mean_std = pd.DataFrame(scaler.transform(test_mean), columns=test.columns)

# %%
train_mean_std.to_csv('train_mean_std.csv', index=False)
train_mean_std = pd.read_csv('train_mean_std.csv')

cv_mean_std.to_csv('cv_mean_std.csv', index=False)
cv_mean_std = pd.read_csv('cv_mean_std.csv')

test_mean_std.to_csv('test_mean_std.csv', index=False)
test_mean_std = pd.read_csv('test_mean_std.csv')

train_mean_std.head(2)

# %% [markdown]
# # 4. ML Models

# %%
def plot_grid_search(clf, X_train, y_train, C):
    train_auc= clf.cv_results_['mean_train_score']
    train_auc_std= clf.cv_results_['std_train_score']
    cv_auc = clf.cv_results_['mean_test_score'] 
    cv_auc_std= clf.cv_results_['std_test_score']

    plt.plot(C, train_auc, label='Train Recall Score')
    # this code is copied from here: https://stackoverflow.com/a/48803361/4084039
    plt.gca().fill_between(C,train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')

    plt.plot(C, cv_auc, label='Test Recall Score')
    # this code is copied from here: https://stackoverflow.com/a/48803361/4084039
    plt.gca().fill_between(C,cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')
    plt.legend()
    plt.xlabel("log(C): hyperparameter")
    plt.ylabel("Recall Score")
    plt.title("ERROR PLOTS")
    plt.grid()
    plt.show()

# This function plots the confusion matrices given y_i, y_i_hat.
def plot_confusion_matrix(train_y, train_predict_y, test_y, test_predict_y):
    # Confusion Matrix
    confusion_matrix_train = confusion_matrix(train_y, train_predict_y)
    
    # Ploting heatmap of confusion matrix
    # https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
    class_names = ['negative','positive']
    confusion_matrix_train = pd.DataFrame(confusion_matrix_train,index=class_names, columns=class_names)
    heatmap = sns.heatmap(confusion_matrix_train, annot=True, fmt='g')

    plt.xlabel('Predicted Class',size=14)
    plt.ylabel('Actual Class',size=14)
    plt.title("Train Confusion Matrix\n",size=24)
    plt.show()
    
    C = confusion_matrix(test_y, test_predict_y)
    # C = 9,9 matrix, each cell (i,j) represents number of points of class i are predicted class j
    
    tn, fp, fn, tp = confusion_matrix(train_y, train_predict_y).ravel()
    print("Train Cost: ", (fp*10+fn*500))
    tn, fp, fn, tp = confusion_matrix(test_y, test_predict_y).ravel()
    print("Test Cost: ", (fp*10+fn*500))

    A =(((C.T)/(C.sum(axis=1))).T)
    #divid each element of the confusion matrix with the sum of elements in that column
    
    # C = [[1, 2],
    #     [3, 4]]
    # C.T = [[1, 3],
    #        [2, 4]]
    # C.sum(axis = 1)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array
    # C.sum(axix =1) = [[3, 7]]
    # ((C.T)/(C.sum(axis=1))) = [[1/3, 3/7]
    #                           [2/3, 4/7]]

    # ((C.T)/(C.sum(axis=1))).T = [[1/3, 2/3]
    #                           [3/7, 4/7]]
    # sum of row elements = 1
    
    B =(C/C.sum(axis=0))
    #divid each element of the confusion matrix with the sum of elements in that row
    # C = [[1, 2],
    #     [3, 4]]
    # C.sum(axis = 0)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array
    # C.sum(axix =0) = [[4, 6]]
    # (C/C.sum(axis=0)) = [[1/4, 2/6],
    #                      [3/4, 4/6]] 
    plt.figure(figsize=(20,5))
    plt.suptitle("Test Confusion, Presicion & Recall Matrix", fontsize=24)
   
    labels = ['negative','positive']
    # representing A in heatmap format
    plt.subplot(1, 3, 1)
    sns.heatmap(C, annot=True, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Confusion matrix")
    
    plt.subplot(1, 3, 2)
    sns.heatmap(B, annot=True, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Precision matrix")
    
    plt.subplot(1, 3, 3)
    # representing B in heatmap format
    sns.heatmap(A, annot=True, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Recall matrix")
    
    plt.show()
    
    
def plot_precision_recall_costs(model, data, y_true):
#     X_train, X_test, y_train, y_test = train_test_split(data, y_true, stratify=y_true, \
#                                                         test_size=0.266666666666666667, random_state=42)
    X_test = data
    y_test = y_true
    y_pred = calib.predict_proba(data)[:,1]
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    thresholds = np.append(thresholds,1)

    costs = []

    for threshold in tqdm(thresholds):
        y_hat = y_pred > threshold
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        cost = fp*10+fn*500
        costs.append(cost)
    plt.figure(figsize=(20,6))

    plt.subplot(121)
    plt.plot(thresholds, precision, label='Precision')
    plt.plot(thresholds, recall, label='Recall')
    plt.legend()
    plt.xlabel("Thresholds")
    plt.ylabel("Precision_Recall")
    plt.title("Precision_Recall-Threshold Plot")
    plt.grid()

    plt.subplot(122)
    plt.plot(thresholds, costs, label='Costs')
    plt.legend()
    plt.xlabel("Thresholds")
    plt.ylabel("Costs")
    plt.title("Cost-Threshold Plot")
    plt.grid()

    plt.show()
    print("Best threshold: {:.4f}".format(thresholds[np.argsort(costs)[0]]))
    print("Min cost: {:.2f}".format(costs[np.argsort(costs)[0]]))
    return thresholds[np.argsort(costs)[0]], costs[np.argsort(costs)[0]]


# %% [markdown]
# # 4.1 Using Imbalance data

# %%
metric = []
base = 10
cv = 5

# %% [markdown]
# ## 4.1.1. Mean Impute

# %%
X_train = pd.read_csv('train_mean.csv')
X_cv = pd.read_csv('cv_mean.csv')
X_test = pd.read_csv('test_mean.csv')

X_train_std = pd.read_csv('train_mean_std.csv')
X_cv_std = pd.read_csv('cv_mean_std.csv')
X_test_std = pd.read_csv('test_mean_std.csv')

y_train = pd.read_csv('y_train.csv')['class']
y_cv = pd.read_csv('y_cv.csv')['class']
y_test = pd.read_csv('y_test.csv')['class']

# %%
y_train.value_counts()

# %%
y_cv.value_counts()

# %%
y_test.value_counts()

# %% [markdown]
# ### 4.1.1.1. Logistic Regression

# %%
C = [math.pow(base,i) for i in range(-6,6)]
# H = [round(math.log(i,10)) for i in C]

tuned_parameters = [{'C': C}, {'penalty':['l1','l2']}, {'class_weight':[None,'balanced']}]

C = [round(math.log(i,base)) for i in C]
clf = GridSearchCV(LogisticRegression(),                    tuned_parameters, cv=cv, scoring='recall', n_jobs=-1, verbose=10)
clf.fit(X_train_std, y_train)

# plot_grid_search(clf, X_train, y_train, C)
print(clf.best_estimator_)
print(clf.best_params_)
best_estimator = clf.best_estimator_

calib = CalibratedClassifierCV(best_estimator, cv=cv, method='sigmoid')
calib.fit(X_train_std, y_train)
plot_confusion_matrix(y_train, calib.predict(X_train_std), y_test, calib.predict(X_test_std))


# %%
threshold, cost = plot_precision_recall_costs(calib, X_cv_std, y_cv)

# %%
y_train_pred = calib.predict_proba(X_train_std)[:,1] > threshold
y_test_pred = calib.predict_proba(X_test_std)[:,1] > threshold
tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
train_cost = fp*10+fn*500
print("Train Cost: ", train_cost)
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
test_cost = fp*10+fn*500
print("Test Cost: ", test_cost)
metric.append(["Mean","Logistic Reg.", train_cost, cost, test_cost])

# %%
plot_confusion_matrix(y_train, y_train_pred, y_test, y_test_pred)

# %% [markdown]
# ### 4.1.1.2. Support Vector Machine

# %%
C = [math.pow(base,i) for i in range(-6,6)]
# H = [round(math.log(i,10)) for i in C]

tuned_parameters = [{'alpha': C}, {'penalty':['l1','l2']}, {'class_weight':[None,'balanced']}]

C = [round(math.log(i,base)) for i in C]
clf = GridSearchCV(SGDClassifier(loss="hinge",max_iter=1000, n_jobs=-1),                    tuned_parameters, cv=cv, scoring='recall', n_jobs=-1)
clf.fit(X_train_std, y_train)

# plot_grid_search(clf, X_train, y_train, C)
print(clf.best_estimator_)
print(clf.best_params_)
best_estimator = clf.best_estimator_

calib = CalibratedClassifierCV(best_estimator, cv=cv, method='sigmoid')
calib.fit(X_train_std, y_train)
plot_confusion_matrix(y_train, calib.predict(X_train_std), y_test, calib.predict(X_test_std))

# %%
threshold, cost = plot_precision_recall_costs(calib, X_cv_std, y_cv)

# %%
_stdy_train_pred = calib.predict_proba(X_train_std)[:,1] > threshold
y_test_pred = calib.predict_proba(X_test_std)[:,1] > threshold
tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
train_cost = fp*10+fn*500
print("Train Cost: ", train_cost)
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
test_cost = fp*10+fn*500
print("Test Cost: ", test_cost)
metric.append(["Mean","Lr. SVM", train_cost, cost, test_cost])

# %%
plot_confusion_matrix(y_train, y_train_pred, y_test, y_test_pred)

# %% [markdown]
# ### 4.1.1.3. Random Forest

# %%
tuned_parameters = {"max_depth": [2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50],
              "n_estimators": [10, 20, 30, 40, 50, 80, 100, 150, 200],
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
                "max_features": ['auto', 'sqrt'],
                "class_weight": ['balanced', 'balanced_subsample', None]
             }
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
clf = RandomizedSearchCV(rf, tuned_parameters, cv=cv, scoring='recall', n_jobs=-1, verbose=10)
clf.fit(X_train, y_train)

print(clf.best_estimator_)

best_estimator = clf.best_estimator_

calib = CalibratedClassifierCV(best_estimator, cv=cv, method='sigmoid')
calib.fit(X_train, y_train)
plot_confusion_matrix(y_train, calib.predict(X_train), y_test, calib.predict(X_test))

# %%
threshold, cost = plot_precision_recall_costs(calib, X_cv, y_cv)

# %%
y_train_pred = calib.predict_proba(X_train)[:,1] > threshold
y_test_pred = calib.predict_proba(X_test)[:,1] > threshold
tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
train_cost = fp*10+fn*500
print("Train Cost: ", train_cost)
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
test_cost = fp*10+fn*500
print("Test Cost: ", test_cost)
metric.append(["Mean","Random Forest", train_cost, cost, test_cost])

# %%
plot_confusion_matrix(y_train, y_train_pred, y_test, y_test_pred)

# %% [markdown]
# ### 4.1.1.4. XGBoost

# %%
# tuned_parameters = {"max_depth": [1, 2, 3, 5, 8, 10, 20, 50, 100],
#               "n_estimators": [10, 20, 50, 100, 200, 300, 500, 800, 1000],
#                 'learning_rate':[0.01,0.03,0.05,0.1,0.15,0.2],
#               'colsample_bytree':[0.1,0.3,0.5,1],
#             'subsample':[0.1,0.3,0.5,1]}

tuned_parameters = {"n_estimators": [10, 20, 30, 40, 50],
                   "max_depth" : [2, 3, 5, 10, 15, 20, 25, 30],
                    'colsample_bytree':[0.1,0.3,0.5,1],
                   'subsample':[0.1,0.3,0.5,1]}


xgbc = xgb.XGBClassifier(n_jobs = -1, random_state=42)
clf = RandomizedSearchCV(xgbc, tuned_parameters, cv=cv, scoring='recall', n_jobs = -1, verbose=10)
clf.fit(X_train, y_train)

print(clf.best_estimator_)
best_estimator = clf.best_estimator_

calib = CalibratedClassifierCV(best_estimator, cv=cv, method='sigmoid')
calib.fit(X_train, y_train)
plot_confusion_matrix(y_train, calib.predict(X_train), y_test, calib.predict(X_test))


# %%
threshold, cost = plot_precision_recall_costs(calib, X_cv, y_cv)

# %%
threshold, cost = plot_precision_recall_costs(calib, X_test, y_test)

# %%
# threshold = 0.007

# %%
y_train_pred = calib.predict_proba(X_train)[:,1] > threshold
y_test_pred = calib.predict_proba(X_test)[:,1] > threshold
tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
train_cost = fp*10+fn*500
print("Train Cost: ", train_cost)
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
test_cost = fp*10+fn*500
print("Test Cost: ", test_cost)
metric.append(["Mean","XGBoost", train_cost, cost, test_cost])

# %%
plot_confusion_matrix(y_train, y_train_pred, y_test, y_test_pred)


# %% [markdown]
# # 5. Conclusion
# %% [markdown]
# ## 5.1. Imbalanced Data

# %%
# metric = pickle.load(open("metric1.pkl","rb"))

# %%

    
x = PrettyTable()

x.field_names = ["Impute", "Model", "Train Cost", "CV Cost", "Test Cost"]

for i in metric:
    x.add_row(i)
print(x)


