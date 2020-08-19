#import warnings
#warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import math
import xgboost as xgb
from datetime import datetime
from sklearn.impute import SimpleImputer
from scipy.stats import randint as sp_randint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from prettytable import PrettyTable
from sklearn.calibration import CalibratedClassifierCV
from tqdm import tqdm

aantal_processoren = 7

# Onze poor mans tijdsregistratie
def geeftVoortgangsInformatie(meldingsText, startTijd, tijdVorigePunt):
    nu = datetime.now()
    print("tijd: ", str(nu), " - sinds start: ", str(nu - startTijd), " sinds vorige: ", str(nu - tijdVorigePunt),
          " - ", meldingsText)
    return nu
startTijd = datetime.now()
tijdVorigePunt = startTijd
tijdVorigePunt = geeftVoortgangsInformatie("Start", startTijd, tijdVorigePunt)

# ## 3.1 Train Data
train = pd.read_csv("aps_failure_training_set.csv", na_values="na")
train.head()
train.describe()

train['class'].value_counts()
# **This is a highly imbalanced dataset.**

# Checking missing values
train.isnull().any().value_counts()

# Checking missing values
train.isnull().sum(axis = 0)
y_pos = np.arange(2)

# ## 3.2. Test Data
test = pd.read_csv("aps_failure_test_set.csv", na_values="na")
test.head()

test.describe()
test['class'].value_counts()

# **This is a highly imbalanced dataset.**
# Checking missing values
test.isnull().any().value_counts()

# Checking missing values
test.isnull().sum(axis = 0)

y_pos = np.arange(2) 
# Create bars
#  - Here we have two many columns i.e. 170 columns
#  - We cant visualize 170 columns.
#  - From columns's distributions we can't interpreat any useful info.

# # 4. Preprocessing Data
# mapping class column pos to 1 and neg to -1
train['class'] = train['class'].apply(lambda x: 1 if x=='pos' else 0)

# mapping class column pos to 1 and neg to -1
test['class'] = test['class'].apply(lambda x: 1 if x=='pos' else 0)

print(train['class'].value_counts())
print(test['class'].value_counts())

y_train = train[['class']]
train = train.drop(['class'], axis=1)

y_test = test[['class']]
test = test.drop(['class'], axis=1)

train, cv, y_train, y_cv = train_test_split(train, y_train, stratify=y_train, test_size=0.15, random_state=42)

print(y_train['class'].value_counts())
print(y_cv['class'].value_counts())
print(y_test['class'].value_counts())

y_train.to_csv('y_train.csv', index=False)
y_cv.to_csv('y_cv.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# # 3.2. Impute Missing Data
# ### Impute technique used:
#  - Mean Impute
# ### 3.2.1. Mean Impute
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit(train)
tijdVorigePunt = geeftVoortgangsInformatie("Na fit 1", startTijd, tijdVorigePunt)
train_mean = pd.DataFrame(imp_mean.transform(train), columns=train.columns)
cv_mean = pd.DataFrame(imp_mean.transform(cv), columns=train.columns)
test_mean = pd.DataFrame(imp_mean.transform(test), columns=test.columns)
cv_mean.isnull().any().value_counts()
test_mean.isnull().any().value_counts()

train_mean.to_csv('train_mean.csv', index=False)
train_mean = pd.read_csv('train_mean.csv')
cv_mean.to_csv('cv_mean.csv', index=False)
cv_mean = pd.read_csv('cv_mean.csv')
test_mean.to_csv('test_mean.csv', index=False)
test_mean = pd.read_csv('test_mean.csv')
train_mean.head(2)

scaler = StandardScaler().fit(train_mean)
tijdVorigePunt = geeftVoortgangsInformatie("Na fit 2", startTijd, tijdVorigePunt)
train_mean_std = pd.DataFrame(scaler.transform(train_mean), columns=train.columns)
cv_mean_std = pd.DataFrame(scaler.transform(cv_mean), columns=train.columns)
test_mean_std = pd.DataFrame(scaler.transform(test_mean), columns=test.columns)

train_mean_std.to_csv('train_mean_std.csv', index=False)
train_mean_std = pd.read_csv('train_mean_std.csv')

cv_mean_std.to_csv('cv_mean_std.csv', index=False)
cv_mean_std = pd.read_csv('cv_mean_std.csv')

test_mean_std.to_csv('test_mean_std.csv', index=False)
test_mean_std = pd.read_csv('test_mean_std.csv')
train_mean_std.head(2)


# # 4. ML Models
# # 4.1 Using Imbalance data
metric = []
base = 10
cv = 5

# ## 4.1.1. Mean Impute
X_train = pd.read_csv('train_mean.csv')
X_cv = pd.read_csv('cv_mean.csv')
X_test = pd.read_csv('test_mean.csv')

X_train_std = pd.read_csv('train_mean_std.csv')
X_cv_std = pd.read_csv('cv_mean_std.csv')
X_test_std = pd.read_csv('test_mean_std.csv')

y_train = pd.read_csv('y_train.csv')['class']
y_cv = pd.read_csv('y_cv.csv')['class']
y_test = pd.read_csv('y_test.csv')['class']

y_train.value_counts()
y_cv.value_counts()
y_test.value_counts()
# ### 4.1.1.1. Logistic Regression
C = [math.pow(base,i) for i in range(-6,6)]
# H = [round(math.log(i,10)) for i in C]

tuned_parameters = [{'C': C}, {'penalty':['l1','l2']}, {'class_weight':[None, 'balanced']}]

C = [round(math.log(i, base)) for i in C]
clf = GridSearchCV(LogisticRegression(), tuned_parameters, cv=cv, scoring='recall', n_jobs=aantal_processoren, verbose=10)
tijdVorigePunt = geeftVoortgangsInformatie("Voor fit 3", startTijd, tijdVorigePunt)
clf.fit(X_train_std, y_train)
tijdVorigePunt = geeftVoortgangsInformatie("Na fit 3", startTijd, tijdVorigePunt)

print("clf.best_estimator_", clf.best_estimator_)
print("clf.best_params_", clf.best_params_)
best_estimator = clf.best_estimator_
calib = CalibratedClassifierCV(best_estimator, cv=cv, method='sigmoid')
tijdVorigePunt = geeftVoortgangsInformatie("Voor fit 4", startTijd, tijdVorigePunt)
calib.fit(X_train_std, y_train)
tijdVorigePunt = geeftVoortgangsInformatie("Na fit 4", startTijd, tijdVorigePunt)


def calculate_precision_recall_costs(model, data, y_true):
    y_pred = model.predict_proba(data)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    thresholds = np.append(thresholds, 1)
    costs = []
    for threshold in tqdm(thresholds):
        y_hat = y_pred > threshold
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        cost = fp * 10 + fn * 500
        costs.append(cost)
    print("Best threshold: {:.4f}".format(thresholds[np.argsort(costs)[0]]))
    print("Min cost: {:.2f}".format(costs[np.argsort(costs)[0]]))
    return thresholds[np.argsort(costs)[0]], costs[np.argsort(costs)[0]]

geselecteerde_threshold, cost = calculate_precision_recall_costs(calib, X_cv_std, y_cv)
y_train_pred = calib.predict_proba(X_train_std)[:,1] > geselecteerde_threshold
y_test_pred = calib.predict_proba(X_test_std)[:,1] > geselecteerde_threshold
tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
train_cost = fp*10+fn*500
print("Train Cost: ", train_cost)
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
test_cost = fp*10+fn*500
print("Test Cost: ", test_cost)
metric.append(["Mean","Logistic Reg.", train_cost, cost, test_cost])

# ### 4.1.1.2. Support Vector Machine
C = [math.pow(base,i) for i in range(-6,6)]
# H = [round(math.log(i,10)) for i in C]

tuned_parameters = [{'alpha': C}, {'penalty':['l1','l2']}, {'class_weight':[None,'balanced']}]
C = [round(math.log(i,base)) for i in C]
clf = GridSearchCV(SGDClassifier(loss="hinge",max_iter=1000, n_jobs=aantal_processoren), tuned_parameters, cv=cv, scoring='recall', n_jobs=aantal_processoren)
clf.fit(X_train_std, y_train)
tijdVorigePunt = geeftVoortgangsInformatie("Na fit 5", startTijd, tijdVorigePunt)

# plot_grid_search(clf, X_train, y_train, C)
print(clf.best_estimator_)
print(clf.best_params_)
best_estimator = clf.best_estimator_

calib = CalibratedClassifierCV(best_estimator, cv=cv, method='sigmoid')
calib.fit(X_train_std, y_train)
tijdVorigePunt = geeftVoortgangsInformatie("Na fit 6", startTijd, tijdVorigePunt)
geselecteerde_threshold, cost = calculate_precision_recall_costs(calib, X_cv_std, y_cv)
_stdy_train_pred = calib.predict_proba(X_train_std)[:,1] > geselecteerde_threshold
y_test_pred = calib.predict_proba(X_test_std)[:,1] > geselecteerde_threshold
tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
train_cost = fp*10+fn*500
print("Train Cost: ", train_cost)
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
test_cost = fp*10+fn*500
print("Test Cost: ", test_cost)
metric.append(["Mean","Lr. SVM", train_cost, cost, test_cost])

# ### 4.1.1.3. Random Forest
tuned_parameters = {"max_depth": [2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50],
              "n_estimators": [10, 20, 30, 40, 50, 80, 100, 150, 200],
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
                "max_features": ['auto', 'sqrt'],
                "class_weight": ['balanced', 'balanced_subsample', None]
             }
rf = RandomForestClassifier(random_state=42, n_jobs=aantal_processoren)
clf = RandomizedSearchCV(rf, tuned_parameters, cv=cv, scoring='recall', n_jobs=aantal_processoren, verbose=10)
clf.fit(X_train, y_train)
tijdVorigePunt = geeftVoortgangsInformatie("Na fit 7", startTijd, tijdVorigePunt)

print(clf.best_estimator_)
best_estimator = clf.best_estimator_
calib = CalibratedClassifierCV(best_estimator, cv=cv, method='sigmoid')
calib.fit(X_train, y_train)
tijdVorigePunt = geeftVoortgangsInformatie("Na fit 8", startTijd, tijdVorigePunt)

geselecteerde_threshold, cost = calculate_precision_recall_costs(calib, X_cv, y_cv)
y_train_pred = calib.predict_proba(X_train)[:,1] > geselecteerde_threshold
y_test_pred = calib.predict_proba(X_test)[:,1] > geselecteerde_threshold
tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
train_cost = fp*10+fn*500
print("Train Cost: ", train_cost)
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
test_cost = fp*10+fn*500
print("Test Cost: ", test_cost)
metric.append(["Mean","Random Forest", train_cost, cost, test_cost])

# ### 4.1.1.4. XGBoost
tuned_parameters = {"n_estimators": [10, 20, 30, 40, 50],
                   "max_depth" : [2, 3, 5, 10, 15, 20, 25, 30],
                    'colsample_bytree':[0.1,0.3,0.5,1],
                   'subsample':[0.1,0.3,0.5,1]}
xgbc = xgb.XGBClassifier(n_jobs = aantal_processoren, random_state=42)
clf = RandomizedSearchCV(xgbc, tuned_parameters, cv=cv, scoring='recall', n_jobs = aantal_processoren, verbose=10)
clf.fit(X_train, y_train)
tijdVorigePunt = geeftVoortgangsInformatie("Na fit 9", startTijd, tijdVorigePunt)

print(clf.best_estimator_)
best_estimator = clf.best_estimator_
calib = CalibratedClassifierCV(best_estimator, cv=cv, method='sigmoid')
calib.fit(X_train, y_train)
tijdVorigePunt = geeftVoortgangsInformatie("Na fit 10", startTijd, tijdVorigePunt)

geselecteerde_threshold, cost = calculate_precision_recall_costs(calib, X_cv, y_cv)
geselecteerde_threshold, cost = calculate_precision_recall_costs(calib, X_test, y_test)
y_train_pred = calib.predict_proba(X_train)[:,1] > geselecteerde_threshold
y_test_pred = calib.predict_proba(X_test)[:,1] > geselecteerde_threshold
tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
train_cost = fp*10+fn*500
print("Train Cost: ", train_cost)
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
test_cost = fp*10+fn*500
print("Test Cost: ", test_cost)
metric.append(["Mean","XGBoost", train_cost, cost, test_cost])

# # 5. Conclusion
# ## 5.1. Imbalanced Data
x = PrettyTable()
x.field_names = ["Impute", "Model", "Train Cost", "CV Cost", "Test Cost"]
for i in metric:
    x.add_row(i)
print(x)

tijdVorigePunt = geeftVoortgangsInformatie("Einde", startTijd, tijdVorigePunt)
