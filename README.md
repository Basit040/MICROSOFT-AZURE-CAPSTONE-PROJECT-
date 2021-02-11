# Capstone Project- Azure Machine Learning (Predicts Titanic Survivals)
This is the third and last project of Udacity Azure Machine Learning Nanodegree. We use all the knowledge and approaches gained from this nanodegree program and solve the problem of our choice. Dataset should be external.

In this project, we create two models 
1)	Automated ML

2)	Hyperdrive whose parameters are tuned

Then we compare the performance of both models and deploy one of them as HTTP REST endpoint. Once the model is deployed successfully we then test the endpoint
Azure ML Studio graphical interface is not used in this project, we use python SDK notebooks to complete the project. 

## DATASET
### OVERVIEW:
Dataset used in this project has been taken from Kaggle.com. Dataset is titanic machine learning dataset. 

The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15,1912 during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this dataset problem, we build a model that answer the question: “What sort of people were more likely to survive?” using passenger data (i.e. name, age, gender, socio-economic class etc.)

Dataset consists of data 1309 passengers with 11 features.    

### TASK:
Use machine learning to create a model that predicts which passengers survived the Titanic shipwreck. This is a binary classification problem with target variable “Survival” which is set 1 or 0.

### Data Dictionary:
| Variables/ Features | Definition | Key |
|---------------------|------------|-----|
| survival	| Survival	| 0= No, 1=Yes |
| Pclass	| Ticket Class	| 1= 1st, 2=2nd, 3=3rd | 
| Sex	| Sex | |	
| Age	 | Age in years | |	
| Sibsp	| No. of siblings/ spouses abroad the Titanic | |
| Parch	| No. of parents/ children abroad the Titanic |	|
| ticket	| Ticket Number | |	
| Fare | Passenger fare | |	
| Cabin |	Cabin number | |	
| embarked |	Port of Embarkation |	C= Cherbourg, Q= Qweenstown, S= Southampton |

### Variable Notes:
Pclass: a proxy for socioeconomic status (SES)
1st = Upper

2nd = Middle

3rd = Lower

### Age: Age is fractional if less than 1, if the age is estimated, is it in the form of xx.5
Sibsp: The dataset defines family relations in this way

Sibling: brother, sister, stepbrother, stepsister

Spouse: husband, wife (mistresses and fiances were ignored)

Parch: The dataset defines family relation in this way..

Parent= mother, father

Child= daughter, son, stepdaughter, stepson

Some children travelled only with a nanny, therefore parch=0 for them

ACCESS
The data has been loaded in the repository so that it can be used in notebooks via the link below”
“dataset link”











## Overview
This project is part of the Udacity Azure ML Nanodegree. In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model. This model is then compared to an Azure AutoML run. 

Azure Python SDK is the open-source Azure libraries for Python simplify provisioning, managing, and using Azure resources from Python application code. It is composed solely of over 180 individual Python libraries that relate to specific Azure services.

AutoML, is the process of automating the time consuming, iterative tasks of machine learning model development. It allows data scientists, analysts, and developers to build ML models with high scale, efficiency, and productivity all while sustaining model quality. Traditional machine learning model development is resource-intensive, requiring significant domain knowledge and time to produce and compare dozens of models. With automated machine learning, you'll accelerate the time it takes to get production-ready ML models with great ease and efficiency.

## Summary
### Problem
We have given a dataset of Bank Marketing of a Portuguese banking institution. Primary task is classification and we have to predict if the client will subscribe a term deposit. Dataset consists of 20 features and 32,950 rows.
We will have to clean the dataset according to requirement before using any model and moreover dataset has class imbalance and no null values.
The link for the dataset is https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv

### Solution
We have used two different approaches to solve the given problem. Firstly we used logistic regression on Python SDK with or choice of parameters and secondly we used Azure AutoML. After using Python SDK we got accuracy of 0.91296 and after using AutoML we got .9171 with VotingEnsemble algorithm. 

## Scikit-learn Pipeline
We wrote train.py script and hyperdrive coding to do data collection, data cleaning, data splitting, model training, hyperparameter sampling model testing, early stopping policy and saving the model. Following are the steps involved in scikit-learn pipeline:

a)	Dataset of bankmarketing is extracted from the link provided and we used TabularDatasetFactory. We did some exploration and understand the meaning of each features.

b)	After loading the dataset we started data cleaning by dropping rows with empty values and one hot encoding for categorical columns.

c)	After cleaning the dataset, we split the data into training and testing dataset. For this experiment we split our data in to 75% for training and 25% for testing.

d)	After splitting the data, further we went for selection of best algorithm for classification problem which is LogisticRegression because we are trying to predict if a client will subscribe to a term deposit product. 

e)	After the creation of the model, we calculate it's Accuracy.

f)	To improve the accuracy of the model we optimized hyperparameters using Hyperdrive. There are two hyperparameters for this model, C and max_iter. C is the inverse regularization strength and max_iter is the maximum iteration to converge for the Logistic Regression.

g)	We used RandomParameterSampling to try different sets of hyperparameter in order to maximize Accuracy. Benefit of RandomSampling is it choose hyperparamters randmoly thus reducing the computational time and complexity. Other options we had were Grid sampling and Bayesian sampling both of them had their own importance like Grid sampling confines the search within a given set of paramters while Bayesian sampling select the next set of hyperparameters based on how the previous hyperparams performed

h)	The parameter search space used for C is [0.01,0.02,0.03,0.04,0.05] and for max_iter is [70,150,220,300,400]

i)	 For this experiment the configuratin used is; evaluation_interval=1, slack_factor=0.01, and delay_evaluation=3

j)	Early stopping policy used here is BanditPolicy. This policy is based on slack factor/slack amount and evaluation interval. Its benefit is it terminates any runs early where the primary metric (accuracy) is not within the selected slack factor with respect to the best performing training run. This helps to improves computational efficiency.

k)	We run this Pipeline multiple times and we register our model for future use. In this case the best model was generated using this hyperparameters (C = '0.02', max_iter = '300') and give us an Accuracy of 0.9129

l)	The trained model is then saved, this is important if we want to deploy our model or use it in some other experiments.


<img src = "2.png"  />

<img src = "c and max_iter.png" />

## AutoML
25 models were generated from AutoML and few of them gave better accuracy than Logistic Regression that we used in Hyperdrive. One model that gave us high accuracy is “Voting Ensemble” i.e. 0.91712 accuracy. 

A voting ensemble works by combining the predictions from multiple models. It can be used for classification or regression. In the case of regression, this involves calculating the average of the predictions from the models. In the case of classification, the predictions for each label are summed and the label with the majority vote is predicted.

<img src = "automl best.png"  />
<img src = "automl metrics.png"  />
<img src = "bestmodel.png"  />
