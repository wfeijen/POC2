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

aantal_processoren = 1
# De volgend code is gekozen om een zeker load te genereren.
# Ter indicatie de tijden die op een ASUS Intel® Core™ i7-6700HQ CPU @ 2.60GHz × 8 GeForce GTX 1060/PCIe/SSE2 en 16 Gb geheugen gemaakt zijn:
# 8 core:
# ('tijd: 2020-08-20 08:33:35.875232 - sinds start: 0:00:00.000003 sinds vorige: 0:00:00.000003 - ', 'Start')
# ('tijd: 2020-08-20 08:33:38.187570 - sinds start: 0:00:02.312341 sinds vorige: 0:00:02.312338 - ', 'Na fit 1')
# ('tijd: 2020-08-20 08:33:48.831335 - sinds start: 0:00:12.956106 sinds vorige: 0:00:10.643765 - ', 'Na fit 2')
# ('tijd: 2020-08-20 08:34:11.132180 - sinds start: 0:00:35.256951 sinds vorige: 0:00:22.300845 - ', 'Voor fit 3')
# ('tijd: 2020-08-20 08:34:45.133252 - sinds start: 0:01:09.258023 sinds vorige: 0:00:34.001072 - ', 'Na fit 3')
# ('tijd: 2020-08-20 08:34:45.134705 - sinds start: 0:01:09.259476 sinds vorige: 0:00:00.001453 - ', 'Voor fit 4')
# ('tijd: 2020-08-20 08:34:52.869939 - sinds start: 0:01:16.994710 sinds vorige: 0:00:07.735234 - ', 'Na fit 4')
# ('tijd: 2020-08-20 08:38:15.550280 - sinds start: 0:04:39.675051 sinds vorige: 0:03:22.680341 - ', 'Na fit 5')
# ('tijd: 2020-08-20 08:38:19.098901 - sinds start: 0:04:43.223672 sinds vorige: 0:00:03.548621 - ', 'Na fit 6')
# ('tijd: 2020-08-20 08:42:41.786067 - sinds start: 0:09:05.910838 sinds vorige: 0:04:22.687166 - ', 'Na fit 7')
# ('tijd: 2020-08-20 08:42:48.433510 - sinds start: 0:09:12.558281 sinds vorige: 0:00:06.647443 - ', 'Na fit 8')
# ('tijd: 2020-08-20 08:51:37.457754 - sinds start: 0:18:01.582525 sinds vorige: 0:08:49.024244 - ', 'Na fit 9')
# ('tijd: 2020-08-20 08:51:48.157266 - sinds start: 0:18:12.282037 sinds vorige: 0:00:10.699512 - ', 'Na fit 10')
# ('tijd: 2020-08-20 08:53:04.610884 - sinds start: 0:19:28.735655 sinds vorige: 0:01:16.453618 - ', 'Einde')
# 4 core (maar niet heus: alle cores werden gebruikt)
# ('tijd: 2020-08-20 09:09:34.468478 - sinds start: 0:00:00.000004 sinds vorige: 0:00:00.000004 - ', 'Start')
# ('tijd: 2020-08-20 09:09:36.777738 - sinds start: 0:00:02.309264 sinds vorige: 0:00:02.309260 - ', 'Na fit 1')
# ('tijd: 2020-08-20 09:09:47.399251 - sinds start: 0:00:12.930777 sinds vorige: 0:00:10.621513 - ', 'Na fit 2')
# ('tijd: 2020-08-20 09:10:09.353017 - sinds start: 0:00:34.884543 sinds vorige: 0:00:21.953766 - ', 'Voor fit 3')
# ('tijd: 2020-08-20 09:10:48.157148 - sinds start: 0:01:13.688674 sinds vorige: 0:00:38.804131 - ', 'Na fit 3')
# ('tijd: 2020-08-20 09:10:48.158608 - sinds start: 0:01:13.690134 sinds vorige: 0:00:00.001460 - ', 'Voor fit 4')
# ('tijd: 2020-08-20 09:10:56.108821 - sinds start: 0:01:21.640347 sinds vorige: 0:00:07.950213 - ', 'Na fit 4')
# ('tijd: 2020-08-20 09:14:17.643366 - sinds start: 0:04:43.174892 sinds vorige: 0:03:21.534545 - ', 'Na fit 5')
# ('tijd: 2020-08-20 09:14:21.404985 - sinds start: 0:04:46.936511 sinds vorige: 0:00:03.761619 - ', 'Na fit 6')
# ('tijd: 2020-08-20 09:17:40.212085 - sinds start: 0:08:05.743611 sinds vorige: 0:03:18.807100 - ', 'Na fit 7')
# ('tijd: 2020-08-20 09:17:58.732838 - sinds start: 0:08:24.264364 sinds vorige: 0:00:18.520753 - ', 'Na fit 8')
# ('tijd: 2020-08-20 09:21:53.272619 - sinds start: 0:12:18.804145 sinds vorige: 0:03:54.539781 - ', 'Na fit 9')
# ('tijd: 2020-08-20 09:22:03.092721 - sinds start: 0:12:28.624247 sinds vorige: 0:00:09.820102 - ', 'Na fit 10')
# ('tijd: 2020-08-20 09:24:47.554556 - sinds start: 0:15:13.086082 sinds vorige: 0:02:44.461835 - ', 'Einde')
# 2 core (maar niet heus: alle cores werden gebruikt)
# ('tijd: 2020-08-20 10:16:58.642929 - sinds start: 0:00:00.000003 sinds vorige: 0:00:00.000003 - ', 'Start')
# ('tijd: 2020-08-20 10:17:00.950331 - sinds start: 0:00:02.307405 sinds vorige: 0:00:02.307402 - ', 'Na fit 1')
# ('tijd: 2020-08-20 10:17:11.627979 - sinds start: 0:00:12.985053 sinds vorige: 0:00:10.677648 - ', 'Na fit 2')
# ('tijd: 2020-08-20 10:17:33.440810 - sinds start: 0:00:34.797884 sinds vorige: 0:00:21.812831 - ', 'Voor fit 3')
# ('tijd: 2020-08-20 10:18:28.610582 - sinds start: 0:01:29.967656 sinds vorige: 0:00:55.169772 - ', 'Na fit 3')
# ('tijd: 2020-08-20 10:18:28.612034 - sinds start: 0:01:29.969108 sinds vorige: 0:00:00.001452 - ', 'Voor fit 4')
# ('tijd: 2020-08-20 10:18:36.550094 - sinds start: 0:01:37.907168 sinds vorige: 0:00:07.938060 - ', 'Na fit 4')
# ('tijd: 2020-08-20 10:22:24.096731 - sinds start: 0:05:25.453805 sinds vorige: 0:03:47.546637 - ', 'Na fit 5')
# ('tijd: 2020-08-20 10:22:27.877170 - sinds start: 0:05:29.234244 sinds vorige: 0:00:03.780439 - ', 'Na fit 6')
# ('tijd: 2020-08-20 10:26:42.787516 - sinds start: 0:09:44.144590 sinds vorige: 0:04:14.910346 - ', 'Na fit 7')
# ('tijd: 2020-08-20 10:26:49.954981 - sinds start: 0:09:51.312055 sinds vorige: 0:00:07.167465 - ', 'Na fit 8')
# ('tijd: 2020-08-20 10:29:13.973050 - sinds start: 0:12:15.330124 sinds vorige: 0:02:24.018069 - ', 'Na fit 9')
# ('tijd: 2020-08-20 10:29:45.918847 - sinds start: 0:12:47.275921 sinds vorige: 0:00:31.945797 - ', 'Na fit 10')
# ('tijd: 2020-08-20 10:35:16.415002 - sinds start: 0:18:17.772076 sinds vorige: 0:05:30.496155 - ', 'Einde')
# 1 core
# ('tijd: 2020-08-20 10:58:16.874877 - sinds start: 0:00:00.000004 sinds vorige: 0:00:00.000004 - ', 'Start')
# ('tijd: 2020-08-20 10:58:19.213087 - sinds start: 0:00:02.338214 sinds vorige: 0:00:02.338210 - ', 'Na fit 1')
# ('tijd: 2020-08-20 10:58:30.119115 - sinds start: 0:00:13.244242 sinds vorige: 0:00:10.906028 - ', 'Na fit 2')
# ('tijd: 2020-08-20 10:58:51.879431 - sinds start: 0:00:35.004558 sinds vorige: 0:00:21.760316 - ', 'Voor fit 3')
# ('tijd: 2020-08-20 11:00:16.745300 - sinds start: 0:01:59.870427 sinds vorige: 0:01:24.865869 - ', 'Na fit 3')
# ('tijd: 2020-08-20 11:00:16.746611 - sinds start: 0:01:59.871738 sinds vorige: 0:00:00.001311 - ', 'Voor fit 4')
# ('tijd: 2020-08-20 11:00:24.386736 - sinds start: 0:02:07.511863 sinds vorige: 0:00:07.640125 - ', 'Na fit 4')
# ('tijd: 2020-08-20 11:05:06.502583 - sinds start: 0:06:49.627710 sinds vorige: 0:04:42.115847 - ', 'Na fit 5')
# ('tijd: 2020-08-20 11:05:10.209494 - sinds start: 0:06:53.334621 sinds vorige: 0:00:03.706911 - ', 'Na fit 6')
# ('tijd: 2020-08-20 11:21:25.714644 - sinds start: 0:23:08.839771 sinds vorige: 0:16:15.505150 - ', 'Na fit 7')
# ('tijd: 2020-08-20 11:22:39.839917 - sinds start: 0:24:22.965044 sinds vorige: 0:01:14.125273 - ', 'Na fit 8')
# ('tijd: 2020-08-20 11:32:22.961177 - sinds start: 0:34:06.086304 sinds vorige: 0:09:43.121260 - ', 'Na fit 9')
# ('tijd: 2020-08-20 11:33:56.446972 - sinds start: 0:35:39.572099 sinds vorige: 0:01:33.485795 - ', 'Na fit 10')
# ('tijd: 2020-08-20 11:39:43.944503 - sinds start: 0:41:27.069630 sinds vorige: 0:05:47.497531 - ', 'Einde')

# Onze poor mans tijdsregistratie
def geeftVoortgangsInformatie(meldingsText, startTijd, tijdVorigePunt, meldRegels):
    nu = datetime.now()
    meldRegel = "tijd: "+ str(nu) + " - sinds start: " + str(nu - startTijd) + " sinds vorige: " + str(nu - tijdVorigePunt) + " - ", meldingsText
    print(meldRegel)
    meldRegels.append(meldRegel)
    return nu, meldRegels


startTijd = datetime.now()
tijdVorigePunt = startTijd
tijdVorigePunt, meldRegels = geeftVoortgangsInformatie("Start", startTijd, tijdVorigePunt, [])

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
tijdVorigePunt, meldRegels = geeftVoortgangsInformatie("Na fit 1", startTijd, tijdVorigePunt, meldRegels)
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
tijdVorigePunt, meldRegels = geeftVoortgangsInformatie("Na fit 2", startTijd, tijdVorigePunt, meldRegels)
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
tijdVorigePunt, meldRegels = geeftVoortgangsInformatie("Voor fit 3", startTijd, tijdVorigePunt, meldRegels)
clf.fit(X_train_std, y_train)
tijdVorigePunt, meldRegels = geeftVoortgangsInformatie("Na fit 3", startTijd, tijdVorigePunt, meldRegels)

print("clf.best_estimator_", clf.best_estimator_)
print("clf.best_params_", clf.best_params_)
best_estimator = clf.best_estimator_
calib = CalibratedClassifierCV(best_estimator, cv=cv, method='sigmoid')
tijdVorigePunt, meldRegels = geeftVoortgangsInformatie("Voor fit 4", startTijd, tijdVorigePunt, meldRegels)
calib.fit(X_train_std, y_train)
tijdVorigePunt, meldRegels = geeftVoortgangsInformatie("Na fit 4", startTijd, tijdVorigePunt, meldRegels)


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
tijdVorigePunt, meldRegels = geeftVoortgangsInformatie("Na fit 5", startTijd, tijdVorigePunt, meldRegels)

# plot_grid_search(clf, X_train, y_train, C)
print(clf.best_estimator_)
print(clf.best_params_)
best_estimator = clf.best_estimator_

calib = CalibratedClassifierCV(best_estimator, cv=cv, method='sigmoid')
calib.fit(X_train_std, y_train)
tijdVorigePunt, meldRegels = geeftVoortgangsInformatie("Na fit 6", startTijd, tijdVorigePunt, meldRegels)
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
tijdVorigePunt, meldRegels = geeftVoortgangsInformatie("Na fit 7", startTijd, tijdVorigePunt, meldRegels)

print(clf.best_estimator_)
best_estimator = clf.best_estimator_
calib = CalibratedClassifierCV(best_estimator, cv=cv, method='sigmoid')
calib.fit(X_train, y_train)
tijdVorigePunt, meldRegels = geeftVoortgangsInformatie("Na fit 8", startTijd, tijdVorigePunt, meldRegels)

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
tijdVorigePunt, meldRegels = geeftVoortgangsInformatie("Na fit 9", startTijd, tijdVorigePunt, meldRegels)

print(clf.best_estimator_)
best_estimator = clf.best_estimator_
calib = CalibratedClassifierCV(best_estimator, cv=cv, method='sigmoid')
calib.fit(X_train, y_train)
tijdVorigePunt, meldRegels = geeftVoortgangsInformatie("Na fit 10", startTijd, tijdVorigePunt, meldRegels)

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

tijdVorigePunt, meldRegels = geeftVoortgangsInformatie("Einde", startTijd, tijdVorigePunt, meldRegels)

print("############################## nog een keer de tijden onder elkaar:")
for meldRegel in meldRegels:
    print(meldRegel)
