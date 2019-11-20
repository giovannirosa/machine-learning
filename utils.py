import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.ensemble import GradientBoostingClassifier

from imblearn.over_sampling import SMOTE

def undersample_data_balance(dataframe):
    a = dataframe['Y'].value_counts()
    min_value = min(a.values)
    balanced_df = pd.concat([dataframe[dataframe['Y'] == 1].sample(n=min_value, random_state=1),
                             dataframe[dataframe['Y'] == 0].sample(n=min_value, random_state=1)],
                            ignore_index=True)
    # balanced_df.describe()
    return balanced_df


def oversample_data_balance(X_train, y_train):
    smt = SMOTE()
    return smt.fit_sample(X_train, y_train)


def feature_selection():
    lasso_clf = LassoCV(cv=10)
    sfm = SelectFromModel(lasso_clf, threshold=0.25)
    sfm.fit(X, y)

    n_feats = sfm.transform(X_train).shape[1]


# threshs = [0.7, 0.8]
path = '/home/lucas/Documentos/trab_ml/machine-learning/scores.txt'

dataframe = pd.read_csv('../credit.csv', header=0)
dataframe.drop('ID', axis=1, inplace=True)

t1 = 0.7
t2 = 0.9
# for t1 in np.arange(0.5, 1, 0.1):
#     for t2 in np.arange(0.5, 1, 0.1):
with open(path, 'a+') as f:
    f.write('\nThreshs: ' + str(t1) + ' -- ' + str(t2) + '\n')

dropped_df = dataframe.dropna(axis='columns', thresh=len(dataframe)*t1).copy()
dropped_df.dropna(axis='rows', thresh=len(dropped_df.columns)*t2, inplace=True)


df1 = dropped_df[dropped_df['Y'] == 1].fillna(dropped_df[dropped_df['Y'] == 1].mean())
df2 = dropped_df[dropped_df['Y'] == 0].fillna(dropped_df[dropped_df['Y'] == 0].mean())

df12 = pd.concat([df1, df2])
# del dataframe

len(dropped_df)
len(dropped_df.columns)

balanced_df = undersample_data_balance(df12)
# balanced_df = dropped_df.copy()
# del dropped_df
len(balanced_df)

X = balanced_df.drop('Y', axis=1)
y = balanced_df['Y']


from sklearn.feature_selection import SelectFromModel
model = SelectFromModel(RandomForestClassifier(n_estimators=50).fit(X, y), prefit=True)
X_new = model.transform(X)
X_new.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.5, random_state=1)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=1)

# X_train, y_train = oversample_data_balance(X_train, y_train)

print(100 * len(X_train)/len(balanced_df))
print(100 * len(X_test)/len(balanced_df))
print(100 * len(X_val)/len(balanced_df))

with open(path, 'a+') as f:
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    dtc_y_pred = dtc.predict(X_test)
    dtc_val_pred = dtc.predict(X_val)
    # print(precision_recall_fscore_support(y_test, dtc_y_pred))
    pred_score = precision_recall_fscore_support(y_val, dtc_val_pred)
    f.write('Decision tree f1: ' + str(pred_score) + '\n')
    print('Decision tree:', pred_score)

    rfc = RandomForestClassifier(n_estimators=50)
    rfc.fit(X_train, y_train)
    rfc_y_pred = rfc.predict(X_test)
    rfc_val_pred = rfc.predict(X_val)
    # print(precision_recall_fscore_support(y_test, rfc_y_pred))
    pred_score = precision_recall_fscore_support(y_val, rfc_val_pred)

    importances = rfc.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1][:10]
    print("Feature ranking:")

    from sklearn.feature_selection import SelectFromModel
    model = SelectFromModel(rfc, prefit=True)

    X_new = model.transform(X_train)

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


    f.write('Random forest f1: ' + str(pred_score) + '\n')
    print('Random forest:', pred_score)

    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    gbc_y_pred = gbc.predict(X_test)
    gbc_val_pred = gbc.predict(X_val)
    # print(precision_recall_fscore_support(y_test, gbc_y_pred))
    pred_score = precision_recall_fscore_support(y_val, gbc_val_pred)
    f.write('GBC f1: ' + str(pred_score) + '\n')
    f.write('GBC roc: ' + str(roc_curve(y_test, gbc_y_pred)) + '\n')
    print('GBC', pred_score)
    print('Roc: ', roc_curve(y_test, gbc_y_pred))


