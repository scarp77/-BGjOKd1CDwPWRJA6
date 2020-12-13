import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
​
df=pd.read_csv('ACME-HappinessSurvey2020.csv')
df
Y	X1	X2	X3	X4	X5	X6
0	0	3	3	3	4	2	4
1	0	3	2	3	5	4	3
2	1	5	3	3	3	3	5
3	0	5	4	3	3	3	5
4	0	5	4	3	3	3	5
...	...	...	...	...	...	...	...
121	1	5	2	3	4	4	3
122	1	5	2	3	4	2	5
123	1	5	3	3	4	4	5
124	0	4	3	3	4	4	5
125	0	5	3	2	5	5	5
126 rows × 7 columns

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 126 entries, 0 to 125
Data columns (total 7 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   Y       126 non-null    int64
 1   X1      126 non-null    int64
 2   X2      126 non-null    int64
 3   X3      126 non-null    int64
 4   X4      126 non-null    int64
 5   X5      126 non-null    int64
 6   X6      126 non-null    int64
dtypes: int64(7)
memory usage: 7.0 KB
df.describe()
Y	X1	X2	X3	X4	X5	X6
count	126.000000	126.000000	126.000000	126.000000	126.000000	126.000000	126.000000
mean	0.547619	4.333333	2.531746	3.309524	3.746032	3.650794	4.253968
std	0.499714	0.800000	1.114892	1.023440	0.875776	1.147641	0.809311
min	0.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000
25%	0.000000	4.000000	2.000000	3.000000	3.000000	3.000000	4.000000
50%	1.000000	5.000000	3.000000	3.000000	4.000000	4.000000	4.000000
75%	1.000000	5.000000	3.000000	4.000000	4.000000	4.000000	5.000000
max	1.000000	5.000000	5.000000	5.000000	5.000000	5.000000	5.000000
df.eq(0).sum()
Y     57
X1     0
X2     0
X3     0
X4     0
X5     0
X6     0
dtype: int64
df.corr()
Y	X1	X2	X3	X4	X5	X6
Y	1.000000	0.280160	-0.024274	0.150838	0.064415	0.224522	0.167669
X1	0.280160	1.000000	0.059797	0.283358	0.087541	0.432772	0.411873
X2	-0.024274	0.059797	1.000000	0.184129	0.114838	0.039996	-0.062205
X3	0.150838	0.283358	0.184129	1.000000	0.302618	0.358397	0.203750
X4	0.064415	0.087541	0.114838	0.302618	1.000000	0.293115	0.215888
X5	0.224522	0.432772	0.039996	0.358397	0.293115	1.000000	0.320195
X6	0.167669	0.411873	-0.062205	0.203750	0.215888	0.320195	1.000000
X=df[['X1','X3','X5','X6']]             # Actually, i want 
y=df[['Y']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
rf=RandomForestClassifier(n_estimators=250, random_state=42)
​
rf_model=rf.fit(X_train,y_train)
y_pred=rf_model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
​
C:\Users\ACER\anaconda3\lib\site-packages\ipykernel_launcher.py:1: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  """Entry point for launching an IPython kernel.
0.6052631578947368
from sklearn.model_selection import cross_val_score
cross_val_score(rf_model,X_test,y_test,cv=10).mean()
C:\Users\ACER\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:515: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  estimator.fit(X_train, y_train, **fit_params)
C:\Users\ACER\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:515: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  estimator.fit(X_train, y_train, **fit_params)
C:\Users\ACER\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:515: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  estimator.fit(X_train, y_train, **fit_params)
C:\Users\ACER\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:515: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  estimator.fit(X_train, y_train, **fit_params)
C:\Users\ACER\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:515: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  estimator.fit(X_train, y_train, **fit_params)
C:\Users\ACER\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:515: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  estimator.fit(X_train, y_train, **fit_params)
C:\Users\ACER\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:515: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  estimator.fit(X_train, y_train, **fit_params)
C:\Users\ACER\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:515: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  estimator.fit(X_train, y_train, **fit_params)
C:\Users\ACER\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:515: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  estimator.fit(X_train, y_train, **fit_params)
C:\Users\ACER\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:515: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  estimator.fit(X_train, y_train, **fit_params)
0.8083333333333333
results=[]
results.append(rf_score)
results
results
[0.8083333333333333]
import pickle
filename='ACME-HappinessSurvey2020.sav'
​
ilename
rf.fit(X,y)
pickle.dump(rf,open(filename,'wb'))
C:\Users\ACER\anaconda3\lib\site-packages\ipykernel_launcher.py:1: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  """Entry point for launching an IPython kernel.
ilename
loaded_model=pickle.load(open(filename,'rb'))
loaded_model
loaded_model
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=250,
                       n_jobs=None, oob_score=False, random_state=42, verbose=0,
                       warm_start=False)
X1=5
X3=3
X5=5
X6=4
prediction=loaded_model.predict([[X1,X3,X5,X6]])
prediction
prediction
array([1], dtype=int64)