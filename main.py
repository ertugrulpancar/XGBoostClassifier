import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import scatterplot
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv('/Users/ertugrul/PycharmProjects/DataScienceMLCourse/XGBoostClassifier/20-digitalskysurvey.csv')

df.head()

columns_to_drop = ['objid', 'specobjid', 'rerun', 'camcol', 'field', 'run']

df.drop(columns_to_drop, axis=1, inplace=True)
df.head()

df.info()

df['class'].value_counts()

le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])

df.corr()

sns.scatterplot(data= df, x= 'mjd', y= 'plate', hue= 'class')
plt.show()

X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.33, random_state= 15)


xgb = XGBClassifier(n_estimators= 100)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

params = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [5, 8, 12, 20, 30],
    'colsample_bytree': [0.3, 0.4, 0.5, 0.8, 1],
}

grid = GridSearchCV(estimator= XGBClassifier(), param_grid= params, cv= 5, n_jobs= -1, verbose= 1)

grid.fit(X_train, y_train)
grid.best_params_
grid.best_score_
y_pred = grid.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))