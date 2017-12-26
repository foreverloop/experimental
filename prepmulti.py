"""
Analysis performed before attempting multiple regression.

the idea would be to predict a person's lung capacity based on their age and height

but it is best to perform this basic analysis first to get an idea of the kinds of
relationships the data already has.

writing this after running this program, I can already see age and height are positively correlated
which will cause overfitting and problems related to multicolinearity

the simple linear regressions performed at the end indicate, age is a better indicator
of a peron's lung capacity than a person's height
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from scipy.stats import pearsonr

df = pd.read_csv('lungcap.csv')
age = df['Age']
height = df['Height']
lung_capacity = df['LungCap']

plt.scatter(lung_capacity,age,color='purple')
plt.title('Lung Capacity / Age Scatter')
plt.xlabel('Lung Capacity')
plt.ylabel('Age')

#correlation coefficient/p-value for lung capacity / age, unable to fit in legend
print 'Pearson Correlation: ',pearsonr(lung_capacity,age)
plt.show()

plt.scatter(lung_capacity,height,color='green')
plt.xlabel('Lung Capacity')
plt.title('Lung Capacity / Height Scatter')
plt.ylabel('Height')

#correlation coefficient/p-value for lung capacity / height, unable to fit in legend
print "Pearson Correlation: ", pearsonr(lung_capacity,height)
plt.show()

"""
now going to check if age and height are correlated in some manner
if they are, multicolinearity is going to be an issue for the multiple regression
"""

plt.scatter(age,height,color='orange')
plt.xlabel('Age')
plt.ylabel('Height')
plt.title('Age / Height Scatter')
plt.show()

#age and height, our independent variables predicting our dependent lung capacity
#are strongly positively correlated, this will be an issue
print " Correlation / p-value: {0}. ".format(pearsonr(age,height))


#prepare data for simple regression tests before multi tests
train_age = age[:490]
train_height = height[:490]

train_age = train_age.values.reshape([-1,1])
train_height = train_height.values.reshape([-1,1])

train_lcap = lung_capacity[:490]
train_lcap = train_lcap.values.reshape([-1,1])

test_age = age[490:]
test_age = test_age.values.reshape([-1,1])
test_height = height[490:]
test_height = test_height.values.reshape([-1,1])

#simple 1 independent variable regression tests
regr = linear_model.LinearRegression()
regr.fit(train_age,train_lcap)
age_lcap_predict = regr.predict(test_age)
plt.scatter(age[490:],lung_capacity[490:],color='green')
plt.plot(test_age,age_lcap_predict,c='purple')
plt.title('Linear Regression of Age / Lung Capacity')
plt.show()

#now below, coefficients
print "Scikit 1 year in age means a {0} change in lung capacity.".format(regr.coef_)

regr2 = linear_model.LinearRegression()
regr2.fit(train_height,train_lcap)
plt.scatter(height[490:],lung_capacity[490:],color='green')
height_lcap_predict = regr2.predict(test_height)
plt.title('Linear Regression of Height / Lung Capacity')
plt.plot(test_height,height_lcap_predict,c='purple')
plt.show()
print "Scikit 1 inch taller means {0} change in lung capacity.".format(regr2.coef_)

