#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 12:11:41 2018

@author: williamzimmerman
Not Meeting Target-1
Approaching Target-2
Meeting target-3
Exceeding Target-4
No data-0
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import metrics


schooldirect=pd.read_excel('/Users/williamzimmerman/Downloads/data-science-for-good/2016 School Explorer.xlsx', sheet_name='2016 School Explorer')
SHSATadmit17=pd.read_excel('/Users/williamzimmerman/Downloads/data-science-for-good/2016 School Explorer.xlsx', sheet_name='Shsat admission 2017')
SHSATadmit16=pd.read_excel('/Users/williamzimmerman/Downloads/data-science-for-good/2016 School Explorer.xlsx', sheet_name='Shsat admission 2016')
SHSATadmit15=pd.read_excel('/Users/williamzimmerman/Downloads/data-science-for-good/2016 School Explorer.xlsx', sheet_name='Shsat admission 2015')


Master=pd.read_excel('/Users/williamzimmerman/Downloads/data-science-for-good/2016 School Explorer.xlsx',sheet_name='Master')
Master2=pd.read_excel('/Users/williamzimmerman/Downloads/data-science-for-good/2016 School Explorer.xlsx',sheet_name='Master2')
labels=Master2['Ratio of Offers']



plt.style.use('ggplot')
plt.scatter(Master['Economic Need Index INT'], Master['Ratio of Offers'])
plt.ylabel('% of test takers offered seats')
plt.xlabel('Economic Need Index')
plt.show()

plt.style.use('ggplot')
plt.scatter(Master['Trust Rating'], Master['Ratio of Offers'])
plt.xlabel('Trust Rating')
plt.ylabel('% of test takers offered seats')
plt.show()

plt.style.use('ggplot')
plt.scatter(Master['Collaborative Teachers % INT'], Master['Ratio of Offers'])
plt.xlabel('Collaborative Teachers %')
plt.ylabel('% of test takers offered seats')
plt.show()

plt.style.use('ggplot')
plt.scatter(Master['Rigorous Instruction Rating'], Master['Ratio of Offers'])
plt.xlabel('Rigorous Instruction Rating')
plt.ylabel('% of test takers offered seats')
plt.show()


headers=['Economic Need Index INT', 'Rigorous Instruction Rating', 'Collaborative Teachers % INT', 'Collaborative Teachers Rating', 'Trust % INT', 'Trust Rating', 'Ratio of Offers']

print("Economic Need:")

train_x, test_x, train_y, test_y = train_test_split(Master2[headers[0:1]],Master2[headers[-1]],train_size=.8, random_state=42 )

mul_lr = linear_model.LogisticRegression().fit(train_x, train_y)
print(metrics.accuracy_score(test_y, mul_lr.predict(test_x)))
print(metrics.accuracy_score(train_y, mul_lr.predict(train_x)))

print("Rigorous Instruction Rating:")

train_x, test_x, train_y, test_y = train_test_split(Master2[headers[1:2]],Master2[headers[-1]],train_size=.8, random_state=42 )
mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=2000).fit(train_x, train_y)
print(metrics.accuracy_score(test_y, mul_lr.predict(test_x)))
print(metrics.accuracy_score(train_y, mul_lr.predict(train_x)))

print("collaboration:")
train_x, test_x, train_y, test_y = train_test_split(Master2[headers[3:4]],Master2[headers[-1]],train_size=.8, random_state=42 )
print(test_x.head())
print(train_x.head())
print(test_y.head())
print(train_y.head())
mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=2000).fit(train_x, train_y)
print(metrics.accuracy_score(test_y, mul_lr.predict(test_x)))
print(metrics.accuracy_score(train_y, mul_lr.predict(train_x)))

print("Trust:")
train_x, test_x, train_y, test_y = train_test_split(Master2[headers[5:6]],Master2[headers[-1]],train_size=.8, random_state=42 )
print(test_x.head())
print(train_x.head())
print(test_y.head())
print(train_y.head())
mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=2000).fit(train_x, train_y)
print(metrics.accuracy_score(test_y, mul_lr.predict(test_x)))
print(metrics.accuracy_score(train_y, mul_lr.predict(train_x)))


print("all")
train_x, test_x, train_y, test_y = train_test_split(Master2[headers[0:6]],Master2[headers[-1]],train_size=.8, random_state=42 )


mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=10000).fit(train_x, train_y)
print(metrics.accuracy_score(test_y, mul_lr.predict(test_x)))
print(metrics.accuracy_score(train_y, mul_lr.predict(train_x)))



