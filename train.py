# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 19:56:52 2021
@author: Abdul Basit
"""

import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

def clean_data(data):
    

    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    def remove_zero_fares(row):
        if row.Fare == 0:
            row.Fare = np.NaN
        return row
    # Apply the function
    x_df = x_df.apply(remove_zero_fares, axis=1)
    x_df['Fare'] = x_df.Fare.fillna(x_df.Fare.median())
    # If person has cabin, there is expected to be some value. Else, the feature has a NaN value
    x_df['Has_Cabin'] = ~x_df.Cabin.isnull()
    # Handling null values in Embarked - impute using mode
    x_df['Embarked']=x_df.Embarked.fillna('S')
    # Removing age null values
    x_df['Age'] = x_df.Age.fillna(x_df.Age.median())
    #Binning Age and fare -> Categorical features
    categorical_age = pd.cut(x_df['Age'], 5)
    categorical_fare = pd.qcut(x_df['Fare'], 4)
    # Mapping Age
    x_df.loc[ x_df['Age'] <= 16, 'Age'] = 0
    x_df.loc[(x_df['Age'] > 16) & (x_df['Age'] <= 32), 'Age'] = 1
    x_df.loc[(x_df['Age'] > 32) & (x_df['Age'] <= 48), 'Age'] = 2
    x_df.loc[(x_df['Age'] > 48) & (x_df['Age'] <= 64), 'Age'] = 3
    x_df.loc[ x_df['Age'] > 64, 'Age'] = 4 ;
    x_df['Age'] = x_df['Age'].astype(int)

    # Mapping Fare
    x_df.loc[ x_df['Fare'] <= 7.925, 'Fare'] = 0
    x_df.loc[(x_df['Fare'] > 7.925) & (x_df['Fare'] <= 14.5), 'Fare'] = 1
    x_df.loc[(x_df['Fare'] > 14.5) & (x_df['Fare'] <= 31.275), 'Fare'] = 2
    x_df.loc[ x_df['Fare'] > 31.275, 'Fare'] = 3
    x_df['Fare'] = x_df['Fare'].astype(int)
    x_df.Cabin.fillna('Unknown',inplace=True)
    # Extract first letter
    x_df['Cabin'] = x_df['Cabin'].map(lambda x : x[0])
    #New Feature
    family_size = x_df['SibSp'] + x_df['Parch'] + 1
    # Creation of four groups
    x_df['Fam_type'] = pd.cut(family_size, [0,1,4,7,11], labels=['Solo', 'Small', 'Big', 'Very big'])
    # Title Feature from Name
    def get_title(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search.group(1)
        return ""
    # Create a new feature Title, containing the titles of passenger names
    x_df['Title'] = x_df['Name'].apply(get_title)
    # Substitute rare female titles
    x_df['Title'].replace(['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Countess', 'Dona'], 'Miss', inplace=True)
    # Substitute rare male titles
    x_df['Title'].replace(['Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer'], 'Mr', inplace=True)
    features_to_drop = ['Name', 'SibSp', 'Parch', 'Ticket']
    x_df.drop(features_to_drop, axis = 1, inplace = True)
    categorical_features = ['Pclass', 'Sex', 'Fam_type', 'Title', 'Embarked', 'Cabin']
    x_df = pd.get_dummies(x_df, columns = categorical_features, drop_first=True)    
    x_df = x_df.drop(columns=['Sex_male', 'Has_Cabin'])
    x_df['Survived'] = x_df['Survived'].astype(int)
    
    
    y_df=x_df.pop('Survived')
    
    return x_df, y_df
    
    
data = "https://raw.githubusercontent.com/Basit040/Capstone-Azure-Machine-Learning/main/titanic.csv"
#creating data in Tabular format via TabularDatasetFactory
ds = TabularDatasetFactory.from_delimited_files(data)
run = Run.get_context()

# clean data function
# Extracting x and y from clean data function
x, y = clean_data(ds)

# Split data into train and test sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()
    
    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/hyperDrive_{}_{}'.format(args.C,args.max_iter))

if __name__ == '__main__':
    main()

