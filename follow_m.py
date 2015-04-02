
import pandas as pd
from sklearn.cross_validation import LeaveOneOut
import pylab
from sklearn import tree
import numpy as np
# from sklearn.feature_extraction import DictVectorizer
import math


train_df = pd.read_csv('train.csv', parse_dates=[1],index_col='Id')
train_df = train_df[train_df['revenue']<10000000]
train_df = train_df.replace(0, np.nan)
train_df = train_df.dropna(axis=1,thresh=80)
train_df = train_df.replace('IL',0)
train_df = train_df.replace('FC',1)
train_df = train_df.replace('DT',2)
train_df['City Group'] = train_df['City Group'].map({'Big Cities': 0, 'Other': 1})
train_df['Open_Year'] = train_df['Open Date'].map( lambda x : x.year)
# c_year = train_df['City']
# c_year[train_df['Open_Year']<2007] = u'b'
# c_year[train_df['Open_Year']>=2007] = u'g'
# train_df.City = train_df.City.astype('category')
# train_df.Type = train_df.Type.astype('category')
# train_df['City Group'] = train_df['City_group'].astype('category')
# print train_df.columns
# pylab.scatter(train_df['P5'],train_df['revenue'],c=list(c_year))
# pylab.title('Revenue vs P5')
# pylab.show()

X = pd.DataFrame(train_df,columns = ['Open_Year','P2','P28','Type'])
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X.values, train_df['revenue'])
predicty = clf.predict(X.values)
errors = predicty - train_df['revenue']
RMSE = math.sqrt(np.mean(errors ** 2))
print RMSE
# pylab.hist(errors,bins=20)
# pylab.title('errors histogram')
# pylab.xlabel('error')
# pylab.ylabel('count')
# pylab.show()


loo = LeaveOneOut(50)
# features = ['Open_Year', 'Type']
features = [column for column in train_df.columns if column != 'City' and column != 'Open Date']
test_1_predicts = []
errs = []
for train_idx, test_idx in loo:
    train_1 = train_df.iloc[ train_idx ]
    test_1 = train_df.iloc[ test_idx ]
    X = train_1[ features ]
    x_1 = test_1[ features ]
    clf = tree.DecisionTreeRegressor()
    clf = clf.fit(X.values, train_1['revenue'])
    y_1_hat = clf.predict(x_1)
    test_1_predicts.append(y_1_hat)
    
    y_1 = test_1.revenue.iloc[0]
    errs.append(y_1_hat - y_1)

pylab.hist(pd.Series(data=errs), bins=20)
pylab.show()
test_rmse = math.sqrt(np.mean(pd.Series(data=errs) ** 2))
print test_rmse


