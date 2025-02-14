# Note: This data is about who person purchased a home with his age,salary,state and in this purchased is dependant variable & all other are independent variables

# IMPORTNING THE LIBRARY

import numpy as np 	#Array		

import matplotlib.pyplot as plt		

import pandas as pd		

#--------------------------------------------

# import the dataset & divided my dataset into independe & dependent

dataset = pd.read_csv(r"D:\Data Analytics\Dec 17\17th - ML Intro, Data Preprocessing\17th - ML Intro, Data Preprocessing\3. Data preprocessing (ML Practical)\Data.csv")

X = dataset.iloc[:, :-1].values	# Independent variables(state,age,salary)

y = dataset.iloc[:,3].values # Dependent variable(purchased)

#--------------------------------------------

from sklearn.impute import SimpleImputer # SPYDER 4 


imputer = SimpleImputer() 


imputer = imputer.fit(X[:,1:3]) 

X[:, 1:3] = imputer.transform(X[:,1:3]) # Removing null value from salary & age column


# HOW TO ENCODE CATEGORICAL DATA & CREATE A DUMMY VARIABLE(string datatype of state & purchased columns converted into 0's & 1's)

from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()

labelencoder_X.fit_transform(X[:,0]) 

X[:,0] = labelencoder_X.fit_transform(X[:,0]) 

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)

#-----------------------------------------------------------------------

# SPLITING THE DATASET IN TRAINING SET & TESTING SET

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size= 0.2, random_state=0) 

# if you remove random_stat then your model not behave as accurate 
# 0-not purchased & 1-purchased
# 20% is test data(i.e.2) & 80% is train data(i.e.8)

#-----------------------------------------------------------------------

# FEATURE SCALING

from sklearn.preprocessing import Normalizer

sc_X = Normalizer() 

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

#-----------------------------------------------------------------------














