#Alfonso Camblor
#Codecademy Machine Learning Fundamentals - Date a Scientist
#Some lines are commented in order to reduce text flood in terminal.

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn import metrics

#We begin by importing the data table.

df = pd.read_csv("profiles.csv")

#I want to explore what values income has in this data.
#print(df.income.value_counts())

#There are more -1 values than properly selected ones. This could mean people doesn't like to publish personal salary. We need to eliminate -1 values.
#This will generate some bias in our predictions. Some groups could be over-represented. Predictions may not be accurate.

#Now I want to explore education column.
#print(df.education.value_counts())

#I see most of the people have some college degree and there are a lot of people with high-level studies.
#Due to the -1 number at income, we aren't able to make accurate predictions, but we will try anyways.

#We need to eliminate -1 values:

print('All rows: ', len(df['income']))
df_income = df.drop(df[df.income == -1].index)
df_income = df_income.reset_index(drop=True)
print('Rows with income: ', len(df_income['income']))

#After the -1 removal, we drop from 59946 rows to 11504. More or less, we now have 1/6 of the total columns. This might introduce bias.

#Age is a needed field, so it would be filled. Anyways, we should look for some unusual values.
#print(df.age.value_counts())
#As we see, there are people with 110 and 109 years... This won't help us making predictions, so I'm going to eliminate that rows.
df_incomeage = df_income.drop(df_income[df_income.age > 69].index)
df_incomeage = df_income.reset_index(drop=True)

#Now we are ready to plot age with income. Let's see what happens.
#plt.scatter(df_incomeage.age, df_incomeage.income)
#plt.xlabel("Age")
#plt.ylabel("Income")
#plt.show()


#I'm going to map education values by:
#---------------------------------------------------
#0: no education
#1: until bachelor
#2: bachelor
#3: master
#4: phd or above
#---------------------------------------------------
education_mapping = {
    'graduated from college/university': 2,
    'graduated from masters program': 3,
    'working on college/university': 1,
    'working on masters program': 2,
    'graduated from two-year college': 1,
    'graduated from high school': 1,
    'graduated from ph.d program': 4,
    'graduated from law school': 3,
    'working on two-year college': 1,
    'dropped out of college/university': 1,
    'working on ph.d program': 3,
    'college/university': 2,
    'graduated from space camp': 1,
    'dropped out of space camp': 0,
    'graduated from med school': 3,
    'working on space camp': 0,
    'working on law school': 2,
    'two-year college': 1,
    'working on med school': 2,
    'dropped out of two-year college': 1,
    'dropped out of masters program': 2,
    'masters program': 3,
    'dropped out of ph.d program': 3,
    'dropped out of high school': 0,
    'high school': 1,
    'working on high school': 0,
    'space camp': 0,
    'ph.d program': 4,
    'law school': 3,
    'dropped out of law school': 2,
    'dropped out of med school': 2,
    'med school': 3,
}
df_income['e_level'] = df_income.education.map(education_mapping)
df_income['e_level'] = df_income['e_level'].replace(np.nan, -1, regex=True)

#Now I have a new column, e_level with the new classification labels for education column.
#There are some values that are -1. We can't work with those values, so I'm going to delete that rows.
df_income = df_income.drop(df_income[df_income.e_level == -1].index)
df_income = df_income.reset_index(drop=True)
#Now we have gone from 11504 to 10782 rows. More or less, we continue at 1/6 of the total data.

#Plot the data.
#plt.scatter(df_income.income, df_income.e_level)
#plt.xlabel("Income")
#plt.ylabel("Education")
#plt.show()

#We now see income and education aren't that related.

#_________________________________________________________________________________________________________

# Normalize income data in order to make predictions

df_income['e_level'] = df_income['e_level'].replace(np.nan, 0, regex=True)
df_income['income'] = df_income['income'].replace(np.nan, 0, regex=True)

df_regressor = df_income[['income', 'e_level']]

x = df_regressor.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

df_regressor = pd.DataFrame(x_scaled, columns=df_regressor.columns)



#First we separatee into train and test set.

X = df_income[['income']]
Y = df_income[['e_level']]
x_train, x_test, y_train, y_test = train_test_split(X, Y)


#Performing regression (Education from Income) (Multiple Linear Regression)
print('--- Predicting education level knowing income ---')

mlr = LinearRegression()
mlr.fit(x_train, y_train)
#print(mlr.score(x_test, y_test))
#print(mlr.coef_)
#print(mlr.intercept_)

#Now we're trying to predict it with another method:
#Performing regression (Education from Income) (KNeighbors Regressor)
knr = KNeighborsRegressor( weights='distance')

knr.fit(x_train, y_train)

#print('KNr Score: ', knr.score(x_test, y_test))
#As we expected, we can't predict this values very well.


#Now we are going to perform regression into (Income - Age)

#First we separatee into train and test set.
X = df_incomeage[['age']]
Y = df_incomeage[['income']]
x_train, x_test, y_train, y_test = train_test_split(X, Y)

#Performing regression (Income from Age) (Multiple Linear Regression)
mlr = LinearRegression()
mlr.fit(x_train, y_train)
#print(mlr.score(x_test, y_test))

#Performing regression (Income from Age) (KNeighbors Regressor)
knr = KNeighborsRegressor( weights='distance')

knr.fit(x_train, y_train)

#print('KNr Score: ', knr.score(x_test, y_test))
#As we expected, we can't neither predict this values.


#_________________________________________________________________________________________________________


# Now I'm trying to classify into jobs depending on salary and education.
# First of all, we are going to map jobs and plot the data.

# Jobs will be map with the following schema:
# 0: unknown values
# 1: student
# 2: unemployed, 
# 3: retired
#+4: Own labels for each job
job_mapping = {
    'other': 0,
    'student': 1,
    'science / tech / engineering': 4,
    'computer / hardware / software': 4,
    'artistic / musical / writer': 5,
    'sales / marketing / biz dev': 6,
    'medicine / health': 7,
    'education / academia': 8,
    'executive / management': 9,
    'banking / financial / real estate': 10,
    'entertainment / media': 11,
    'law / legal services': 12,
    'hospitality / travel': 13,
    'construction / craftsmanship': 14,
    'clerical / administrative': 15,
    'political / government': 16,
    'rather not say': 0,
    'transportation': 17,
    'unemployed': 2,
    'retired': 3,
    'military': 18,
}
df_income['job_labels'] = df_income.job.map(job_mapping)
df_income['job_labels'] = df_income['job_labels'].replace(np.nan, 0, regex=True)

#Now we separate into datapoints and labels in order to see if our classifier works.

datapoints = df_income[['e_level', 'income']]
labels = df_income['job_labels']

#plt.scatter(datapoints['income'], df_incomeage.income)
#plt.xlabel("Income")
#plt.ylabel("Job")
#plt.show()

#plt.scatter(datapoints['e_level'], df_incomeage.income)
#plt.xlabel("e_level")
#plt.ylabel("Job")
#plt.show()

x_train, x_test, y_train, y_test = train_test_split(datapoints, labels, random_state = 20)

# Performing regression (Job from Income and Education) (KNeighbors Classifier)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

y_predicted = knn.predict(x_test)
score = knn.score(x_test, y_test)

print('KNN score: ', score)
#print('KNN metrics: ', metrics.classification_report(y_test, y_predicted))

# Performing regression (Job from Income and Education) (Support Vector Machines)
classifier = SVC(kernel='rbf', gamma=0.1)
classifier.fit(x_train, y_train)

y_predicted_svc = classifier.predict(x_test)
score_svc = classifier.score(x_test, y_test)

print('SVC score: ', score_svc)
#print('SVC metrics: ', metrics.classification_report(y_test, y_predicted_svc))
