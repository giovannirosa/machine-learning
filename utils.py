import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


def data_balance(dataframe):
    a = dataframe['Y'].value_counts()
    min_value = min(a.values)
    balanced_df = pd.concat([dataframe[dataframe['Y'] == 1].sample(n=min_value, random_state=1),
                             dataframe[dataframe['Y'] == 0].sample(n=min_value, random_state=1)],
                            ignore_index=True)
    # balanced_df.describe()
    return balanced_df

dataframe = pd.read_csv('../credit.csv', header=0)
dataframe.drop('ID', axis=1, inplace=True)

dropped_df = dataframe.dropna(axis='columns').copy()
del dataframe

balanced_df = data_balance(dropped_df)
del dropped_df


X = balanced_df.drop('Y', axis=1)
y = balanced_df['Y']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=1)

print(100 * len(X_train)/len(balanced_df))
print(100 * len(X_test)/len(balanced_df))
print(100 * len(X_val)/len(balanced_df))


def feature_selection():
    lasso_clf = LassoCV(cv=10)
    sfm = SelectFromModel(lasso_clf, threshold=0.25)
    sfm.fit(X, y)

    n_feats = sfm.transform(X_train).shape[1]


# lr = LogisticRegression()
#
# lr.fit(X_train, y_train)
# y_pred = lr.predict(X_test)
#
#
# precision_recall_fscore_support(y_test, y_pred)


dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_y_pred = dtc.predict(X_test)
dtc_val_pred = dtc.predict(X_val)
print(precision_recall_fscore_support(y_test, dtc_y_pred))
print(precision_recall_fscore_support(y_val, dtc_val_pred))

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_y_pred = rfc.predict(X_test)
rfc_val_pred = rfc.predict(X_val)
print(precision_recall_fscore_support(y_test, rfc_y_pred))
print(precision_recall_fscore_support(y_val, rfc_val_pred))


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_y_pred = gbc.predict(X_test)
gbc_val_pred = gbc.predict(X_val)
print(precision_recall_fscore_support(y_test, gbc_y_pred))
print(precision_recall_fscore_support(y_val, gbc_val_pred))

from sklearn.metrics import roc_curve
roc_curve(y_test, gbc_y_pred)