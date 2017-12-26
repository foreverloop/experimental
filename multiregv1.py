"""
despite knowing of the multicolinearity 
of the two independent variables predicting lung capacity
I am continuing with this as a purely academic exercise 
to show multiple regression at work
"""
import pandas as pd
import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

df = pd.read_csv('lungcap.csv')
age = df['Age']
height = df['Height']
lung_capacity = df['LungCap']

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


#x = np.array(train_age,train_height)
x = np.column_stack((train_age,train_height))
y = np.array(train_lcap)

#x = x.values.reshape([2,2])
#y = y.values.reshape([[-1,1]])
#print np.shape(x),np.shape(y)
#print x
test_age_height = np.column_stack((test_age,test_height))

regr = linear_model.LinearRegression()
regr.fit(x,y)

multi_lcap_predict = regr.predict(test_age_height)
print regr.coef_,regr.intercept_

""" pure nonsense commented out here
multi_age_predict = [regr.coef_[0][0] * year + regr.intercept_ for year in age]
multi_height_predict = [regr.coef_[0][1] * aheight + regr.intercept_ for aheight in height]
plt.scatter(age[490:],lung_capacity[490:],color='green')
print np.shape(test_age),np.shape(multi_age_predict)
plt.plot(test_age,multi_age_predict[:235],c='purple')
plt.plot(test_height,multi_height_predict[:235],c='pink')
plt.plot(multi_height_predict,multi_age_predict)
plt.title('multiple regression lines based on Age/Height')
plt.show()
"""
