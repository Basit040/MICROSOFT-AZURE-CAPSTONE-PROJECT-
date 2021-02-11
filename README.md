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

### ACCESS
The data has been loaded in the repository so that it can be used in notebooks via the link below
“https://raw.githubusercontent.com/Basit040/Capstone-Azure-Machine-Learning/main/titanic.csv”

## CLEANING OF DATASET USING train.py script:
Following approaches has been used to clean the data before building any machine learning model:
1)	Handle null values with the median for fare column
2)	Add a feature “Has_Cabin” if someone has a cabin and fill with 0 or 1.
3)	Handle null values with the mode for embarked column
4)	Handle null values with the median for age column
5)	Binning age and fare features
6)	Then mapping the bin values to age and fare columns
7)	Create a new feature “Fam_type” by adding “Sibsp” and “Parch” 
8)	Extract title features
9)	Drop few not required features like Name, sibsp, parch and ticket
10)	Encoding categorical features to pclass, sex, fam_type, title, embarked and cabin

## AUTOMATED ML
In order to configure the Automated ML run we use the following settings or parameters:
| Configuration |	Details |	Value |
|---------------|---------|-------|
| Task |	Type of task to run the experiment | Tasks can be of “Classification”, “Regression” etc |	
| label_column_name	| Label/ target column name | "Survived" |	
| n_cross_validations	| Number of cross validations | 5 |	
| experiment_timeout_minutes	| How long the experiment will run in minutes. This can be used as exit criteria | 25 |	
| max_concurrent_iterations |	Maximum number of iterations would be executed in parallel | 4 |	
| primary_metrics |	Metric we want to optimize | accurcay |	
| training_data |	Dataset store in datastore | data |	
| Compute_target	| To define the compute cluster we will use | trainCluster |	

Best model generated using AutoML is StandardScalerWrapper,Light GBM and give accuracy of 0.7878
AutoML generated about 86 models. AutoML results includes the best model and also delivers information why this choice of model was made in this case of problem by learning what features are directly impacting the model and why. 

 Below are some screenshots of AutoML:
 
 <img src = "Screenshots/1.png"  />
 
 <img src = "Screenshots/2.png"  />
 
 <img src = "Screenshots/3.png"  />
 
 <img src = "Screenshots/4.png"  />
 
 <img src = "Screenshots/5.png"  />
 
 <img src = "Screenshots/6.png"  />
 
 <img src = "Screenshots/7.png"  />
 
 <img src = "Screenshots/8.png"  />
 
 <img src = "Screenshots/9.png"  />
 
 <img src = "Screenshots/10.png"  />
 
 
### Improvement:
AutoML run can be improved in the future by adding more data, by giving more time to run and use deep learning which can give us good result.

## HYPERPARAMETER TUNING
In hyperdrive configuration, it consist of compute target created in azure and a python script. 

After cleaning the dataset, we split the data into training and testing dataset. For this experiment we split our data in to 75% for training and 25% for testing.

After splitting the data, further we went for selection of best algorithm for classification problem which is LogisticRegression because we are trying to predict for survival.

After the creation of the model, we calculate it's Accuracy.

To improve the accuracy of the model we optimized hyperparameters using Hyperdrive. There are two hyperparameters for this model, C and max_iter. C is the inverse regularization strength and max_iter is the maximum iteration to converge for the Logistic Regression.

We used RandomParameterSampling to try different sets of hyperparameter in order to maximize Accuracy. Benefit of RandomSampling is it choose hyperparamters randmoly thus reducing the computational time and complexity. Other options we had were Grid sampling and Bayesian sampling both of them had their own importance like Grid sampling confines the search within a given set of paramters while Bayesian sampling select the next set of hyperparameters based on how the previous hyperparams performed.

The parameter search space used for C is [1,2,3,4,5] and for max_iter is [5,10,15,20,25]

For this experiment the configuratin used is; evaluation_interval=1, slack_factor=0.01, and delay_evaluation=3

Early stopping policy used here is BanditPolicy. This policy is based on slack factor/slack amount and evaluation interval. Its benefit is it terminates any runs early where the primary metric (accuracy) is not within the selected slack factor with respect to the best performing training run. This helps to improves computational efficiency.

We run this Pipeline multiple times and we register our model for future use. In this case the best model was generated using this hyperparameters (C = '3', max_iter = '15') and give us an Accuracy of 0.8695

<img src = "Screenshots/11.png"  />

<img src = "Screenshots/12.png"  />

<img src = "Screenshots/13.png"  />

<img src = "Screenshots/14.png"  />

<img src = "Screenshots/15.png"  />

<img src = "Screenshots/16.png"  />

<img src = "Screenshots/17.png"  />

<img src = "Screenshots/18.png"  />

<img src = "Screenshots/19.png"  />

<img src = "Screenshots/20.png"  />

## MODEL DEPLOYMENT
We deployed the automl model. It  is deployed using Azure Containce Instance (ACI Service) with 2 cpu_cores and 1 gb of memory_gb. Ofcourse this deployment wont be possible without scoring file 'scoringScript.py'  alongside 'envFile.yml'. Scoring file was important for creating Inference Configration for ACI.

<img src = "Screenshots/21.png"  />

<img src = "Screenshots/22.png"  />

<img src = "Screenshots/23.png"  />

<img src = "Screenshots/24.png"  />

<img src = "Screenshots/25.png"  />




## SCREEN RECORDING



