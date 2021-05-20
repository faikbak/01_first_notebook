# **Springboard Decision Tree Specialty Coffee Case Study - Tier 3**




# The Scenario

Imagine you've just finished the Springboard Data Science Career Track course, and have been hired by a rising popular specialty coffee company - RR Diner Coffee - as a data scientist. Congratulations!

RR Diner Coffee sells two types of thing:
- specialty coffee beans, in bulk (by the kilogram only) 
- coffee equipment and merchandise (grinders, brewing equipment, mugs, books, t-shirts).

RR Diner Coffee has three stores, two in Europe and one in the USA. The flagshap store is in the USA, and everything is quality assessed there, before being shipped out. Customers further away from the USA flagship store have higher shipping charges. 

You've been taken on at RR Diner Coffee because the company are turning towards using data science and machine learning to systematically make decisions about which coffee farmers they should strike deals with. 

RR Diner Coffee typically buys coffee from farmers, processes it on site, brings it back to the USA, roasts it, packages it, markets it, and ships it (only in bulk, and after quality assurance) to customers internationally. These customers all own coffee shops in major cities like New York, Paris, London, Hong Kong, Tokyo, and Berlin. 

Now, RR Diner Coffee has a decision about whether to strike a deal with a legendary coffee farm (known as the **Hidden Farm**) in rural China: there are rumours their coffee tastes of lychee and dark chocolate, while also being as sweet as apple juice. 

It's a risky decision, as the deal will be expensive, and the coffee might not be bought by customers. The stakes are high: times are tough, stocks are low, farmers are reverting to old deals with the larger enterprises and the publicity of selling *Hidden Farm* coffee could save the RR Diner Coffee business. 

Your first job, then, is ***to build a decision tree to predict how many units of the Hidden Farm Chinese coffee will be purchased by RR Diner Coffee's most loyal customers.*** 

To this end, you and your team have conducted a survey of 710 of the most loyal RR Diner Coffee customers, collecting data on the customers':
- age
- gender 
- salary 
- whether they have bought at least one RR Diner Coffee product online
- their distance from the flagship store in the USA (standardized to a number between 0 and 11) 
- how much they spent on RR Diner Coffee products on the week of the survey 
- how much they spent on RR Diner Coffee products in the month preeding the survey
- the number of RR Diner coffee bean shipments each customer has ordered over the preceding year. 

You also asked each customer participating in the survey whether they would buy the Hidden Farm coffee, and some (but not all) of the customers gave responses to that question. 

You sit back and think: if more than 70% of the interviewed customers are likely to buy the Hidden Farm coffee, you will strike the deal with the local Hidden Farm farmers and sell the coffee. Otherwise, you won't strike the deal and the Hidden Farm coffee will remain in legends only. There's some doubt in your mind about whether 70% is a reasonable threshold, but it'll do for the moment. 

To solve the problem, then, you will build a decision tree to implement a classification solution. 


-------------------------------
As ever, this notebook is **tiered**, meaning you can elect that tier that is right for your confidence and skill level. There are 3 tiers, with tier 1 being the easiest and tier 3 being the hardest. This is ***tier 3***, so it will be challenging. 

**1. Sourcing and loading** 
- Import packages
- Load data
- Explore the data

 
**2. Cleaning, transforming and visualizing**
- Cleaning the data
- Train/test split
  
  
**3. Modelling** 
- Model 1: Entropy model - no max_depth
- Model 2: Gini impurity model - no max_depth
- Model 3: Entropy model - max depth 3
- Model 4: Gini impurity model - max depth 3


**4. Evaluating and concluding** 
- How many customers will buy Hidden Farm coffee?
- Decision

**5. Random Forest** 
- Import necessary modules
- Model
- Revise conclusion
    

# 0. Overview

This notebook uses decision trees to determine whether the factors of salary, gender, age, how much money the customer spent last week and during the preceding month on RR Diner Coffee products, how many kilogram coffee bags the customer bought over the last year, whether they have bought at least one RR Diner Coffee product online, and their distance from the flagship store in the USA, could predict whether customers would purchase the Hidden Farm coffee if a deal with its farmers were struck. 

# 1. Sourcing and loading
## 1a. Import Packages


```python
import pandas as pd
import numpy as np
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO  
from IPython.display import Image  
import pydotplus
```

## 1b. Load data 


```python
# Read in the data to a variable called coffeeData
data=pd.read_csv('RRDinerCoffeeData.csv')
```

## 1c. Explore the data

As we've seen, exploration entails doing things like checking out the **initial appearance** of the data with head(), the **dimensions** of our data with .shape, the **data types** of the variables with .info(), the **number of non-null values**, how much **memory** is being used to store the data, and finally the major summary statistcs capturing **central tendancy, dispersion and the null-excluding shape of the dataset's distribution**. 

How much of this can you do yourself by this point in the course? Have a real go. 


```python
# Call head() on your data 
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>num_coffeeBags_per_year</th>
      <th>spent_week</th>
      <th>spent_month</th>
      <th>SlrAY</th>
      <th>Distance</th>
      <th>Online</th>
      <th>Decision</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>36</td>
      <td>Female</td>
      <td>0</td>
      <td>24</td>
      <td>73</td>
      <td>42789</td>
      <td>0.003168</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>24</td>
      <td>Male</td>
      <td>0</td>
      <td>44</td>
      <td>164</td>
      <td>74035</td>
      <td>0.520906</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>24</td>
      <td>Male</td>
      <td>0</td>
      <td>39</td>
      <td>119</td>
      <td>30563</td>
      <td>0.916005</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20</td>
      <td>Male</td>
      <td>0</td>
      <td>30</td>
      <td>107</td>
      <td>13166</td>
      <td>0.932098</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24</td>
      <td>Female</td>
      <td>0</td>
      <td>20</td>
      <td>36</td>
      <td>14244</td>
      <td>0.965881</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Call .shape on your data
data.shape
```




    (702, 9)




```python
# Call info() on your data
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 702 entries, 0 to 701
    Data columns (total 9 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   Age                      702 non-null    int64  
     1   Gender                   702 non-null    object 
     2   num_coffeeBags_per_year  702 non-null    int64  
     3   spent_week               702 non-null    int64  
     4   spent_month              702 non-null    int64  
     5   SlrAY                    702 non-null    int64  
     6   Distance                 702 non-null    float64
     7   Online                   702 non-null    int64  
     8   Decision                 474 non-null    float64
    dtypes: float64(2), int64(6), object(1)
    memory usage: 49.5+ KB
    


```python
# Call describe() on your data to get the relevant summary statistics for your data 
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>num_coffeeBags_per_year</th>
      <th>spent_week</th>
      <th>spent_month</th>
      <th>SlrAY</th>
      <th>Distance</th>
      <th>Online</th>
      <th>Decision</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>702.000000</td>
      <td>702.000000</td>
      <td>702.000000</td>
      <td>702.000000</td>
      <td>702.000000</td>
      <td>702.000000</td>
      <td>702.000000</td>
      <td>474.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>34.243590</td>
      <td>2.710826</td>
      <td>32.853276</td>
      <td>107.923077</td>
      <td>43819.843305</td>
      <td>4.559186</td>
      <td>0.531339</td>
      <td>0.639241</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13.927945</td>
      <td>1.593629</td>
      <td>15.731878</td>
      <td>55.348485</td>
      <td>26192.626943</td>
      <td>3.116275</td>
      <td>0.499373</td>
      <td>0.480728</td>
    </tr>
    <tr>
      <th>min</th>
      <td>16.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1617.000000</td>
      <td>0.003168</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>23.000000</td>
      <td>1.000000</td>
      <td>24.250000</td>
      <td>62.000000</td>
      <td>22812.250000</td>
      <td>1.877812</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>28.000000</td>
      <td>3.000000</td>
      <td>36.000000</td>
      <td>113.500000</td>
      <td>41975.000000</td>
      <td>4.196167</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>46.000000</td>
      <td>4.000000</td>
      <td>43.000000</td>
      <td>150.750000</td>
      <td>60223.000000</td>
      <td>6.712022</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>90.000000</td>
      <td>5.000000</td>
      <td>62.000000</td>
      <td>210.000000</td>
      <td>182058.000000</td>
      <td>10.986203</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



# 2. Cleaning, transforming and visualizing
## 2a. Cleaning the data

Some datasets don't require any cleaning, but almost all do. This one does. We need to replace '1.0' and '0.0' in the 'Decision' column by 'YES' and 'NO' respectively, clean up the values of the 'gender' column, and change the column names to words which maximize meaning and clarity. 

First, let's change the name of `spent_week`, `spent_month`, and `SlrAY` to `spent_last_week` and `spent_last_month` and `salary` respectively.


```python
# Check out the names of our data's columns 
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>num_coffeeBags_per_year</th>
      <th>spent_week</th>
      <th>spent_month</th>
      <th>SlrAY</th>
      <th>Distance</th>
      <th>Online</th>
      <th>Decision</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>36</td>
      <td>Female</td>
      <td>0</td>
      <td>24</td>
      <td>73</td>
      <td>42789</td>
      <td>0.003168</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>24</td>
      <td>Male</td>
      <td>0</td>
      <td>44</td>
      <td>164</td>
      <td>74035</td>
      <td>0.520906</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>24</td>
      <td>Male</td>
      <td>0</td>
      <td>39</td>
      <td>119</td>
      <td>30563</td>
      <td>0.916005</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20</td>
      <td>Male</td>
      <td>0</td>
      <td>30</td>
      <td>107</td>
      <td>13166</td>
      <td>0.932098</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24</td>
      <td>Female</td>
      <td>0</td>
      <td>20</td>
      <td>36</td>
      <td>14244</td>
      <td>0.965881</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Make the relevant name changes to spent_week and spent_per_week.
#data.rename(columns={'spent week':'Spent_last_week','spent_month':'spent_last_month','SlrAY':'salary'},inplace=True)
data.rename(columns={'spent_week':'Spent_per_week','spent_month':'Spent_last_month','SlrAY':'Salary',},inplace=True)
```


```python
# Check out the column names
data.columns
```




    Index(['Age', 'Gender', 'num_coffeeBags_per_year', 'Spent_per_week',
           'Spent_last_month', 'Salary', 'Distance', 'Online', 'Decision'],
          dtype='object')




```python
# Let's have a closer look at the gender column. Its values need cleaning.
data.Gender.sample(10)
```




    287      Male
    422    Female
    149      Male
    439    Female
    254      Male
    109      Male
    288    Female
    166    Female
    400    Female
    510      Male
    Name: Gender, dtype: object




```python
# See the gender column's unique values 
data.Gender.unique()
```




    array(['Female', 'Male', 'female', 'F', 'f ', 'FEMALE', 'MALE', 'male',
           'M'], dtype=object)



We can see a bunch of inconsistency here.

Use replace() to make the values of the `gender` column just `Female` and `Male`.


```python
# Replace all alternate values for the Female entry with 'Female'
data=data.replace({'Gender' : {'female':'Female', 'F':'Female', 'f ':'Female', 'FEMALE':'Female'}})
data['Gender']=data['Gender'].replace(['female', 'F', 'f ', 'FEMALE',], 'Female')
```


```python
# Check out the unique values for the 'gender' column
data.Gender.unique()
```




    array(['Female', 'Male'], dtype=object)




```python
# Replace all alternate values with "Male"
data['Gender']=data['Gender'].replace(['Male', 'MALE', 'male', 'M'], 'Male')
```


```python
# Let's check the unique values of the column "gender"
data.Gender.unique()
```




    array(['Female', 'Male'], dtype=object)




```python
# Check out the unique values of the column 'Decision'
data.Decision.unique()
```




    array(['YES', nan, 'NO'], dtype=object)



We now want to replace `1.0` and `0.0` in the `Decision` column by `YES` and `NO` respectively.


```python
# Replace 1.0 and 0.0 by 'Yes' and 'No'
data['Decision'] = data['Decision'].replace([1.], ['YES'])
data['Decision'] = data['Decision'].replace([0.], ['NO'])
```


```python
# Check that our replacing those values with 'YES' and 'NO' worked, with unique()
data['Decision'].unique()
```




    array(['YES', nan, 'NO'], dtype=object)



## 2b. Train/test split
To execute the train/test split properly, we need to do five things: 
1. Drop all rows with a null value in the `Decision` column, and save the result as NOPrediction: a dataset that will contain all known values for the decision 
2. Visualize the data using scatter and boxplots of several variables in the y-axis and the decision on the x-axis
3. Get the subset of coffeeData with null values in the `Decision` column, and save that subset as Prediction
4. Divide the NOPrediction subset into X and y, and then further divide those subsets into train and test subsets for X and y respectively
5. Create dummy variables to deal with categorical inputs

### 1. Drop all null values within the `Decision` column, and save the result as NoPrediction


```python
# NoPrediction will contain all known values for the decision
# Call dropna() on coffeeData, and store the result in a variable NOPrediction 
# Call describe() on the Decision column of NoPrediction after calling dropna() on coffeeData
NoPrediction=data.dropna()
NoPrediction.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>num_coffeeBags_per_year</th>
      <th>Spent_per_week</th>
      <th>Spent_last_month</th>
      <th>Salary</th>
      <th>Distance</th>
      <th>Online</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>474.000000</td>
      <td>474.000000</td>
      <td>474.000000</td>
      <td>474.000000</td>
      <td>474.000000</td>
      <td>474.000000</td>
      <td>474.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>35.417722</td>
      <td>2.590717</td>
      <td>32.592827</td>
      <td>106.727848</td>
      <td>44731.892405</td>
      <td>5.102898</td>
      <td>0.512658</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13.604135</td>
      <td>1.585211</td>
      <td>15.758109</td>
      <td>56.101085</td>
      <td>25567.674430</td>
      <td>3.354891</td>
      <td>0.500368</td>
    </tr>
    <tr>
      <th>min</th>
      <td>16.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3150.000000</td>
      <td>0.003168</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25.000000</td>
      <td>1.000000</td>
      <td>24.000000</td>
      <td>60.000000</td>
      <td>27284.250000</td>
      <td>2.077965</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>29.000000</td>
      <td>3.000000</td>
      <td>35.000000</td>
      <td>113.500000</td>
      <td>42791.500000</td>
      <td>4.832168</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>49.000000</td>
      <td>4.000000</td>
      <td>43.000000</td>
      <td>150.000000</td>
      <td>61621.750000</td>
      <td>7.869086</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>90.000000</td>
      <td>5.000000</td>
      <td>62.000000</td>
      <td>210.000000</td>
      <td>172658.000000</td>
      <td>10.986203</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### 2. Visualize the data using scatter and boxplots of several variables in the y-axis and the decision on the x-axis


```python
# Exploring our new NOPrediction dataset
# Make a boxplot on NOPrediction where the x axis is Decision, and the y axis is spent_last_week
sns.boxplot(x=data.Decision,y=data.Spent_per_week)
```




    <AxesSubplot:xlabel='Decision', ylabel='Spent_per_week'>




    
![png](output_34_1.png)
    


Can you admissibly conclude anything from this boxplot? Write your answer here:




```python
# Make a scatterplot on NOPrediction, where x is distance, y is spent_last_month and hue is Decision 
sns.scatterplot(x='Distance' ,y='spent_last_month',hue='Decision',data=NOPrediction)
```




    <AxesSubplot:xlabel='Distance', ylabel='spent_last_month'>




    
![png](output_36_1.png)
    


Can you admissibly conclude anything from this scatterplot? Remember: we are trying to build a tree to classify unseen examples. Write your answer here:

### 3. Get the subset of coffeeData with null values in the Decision column, and save that subset as Prediction


```python
# Get just those rows whose value for the Decision column is null  
Prediction=data.Decision.isna()
```


```python
# Call describe() on Prediction
Prediction.describe()
```




    count       702
    unique        2
    top       False
    freq        474
    Name: Decision, dtype: object



### 4. Divide the NOPrediction subset into X and y


```python
# Check the names of the columns of NOPrediction
NoPrediction.columns
```




    Index(['Age', 'Gender', 'num_coffeeBags_per_year', 'Spent_per_week',
           'Spent_last_month', 'Salary', 'Distance', 'Online', 'Decision'],
          dtype='object')




```python
# Let's do our feature selection.
# Make a variable called 'features', and a list containing the strings of every column except "Decision"
a=NoPrediction.loc[:, ~NoPrediction.columns.isin(['Decision'])]
features=a.columns

# Make an explanatory variable called X, and assign it: NoPrediction[features]
X=NoPrediction[features]

# Make a dependent variable called y, and assign it: NoPrediction.Decision
y=NoPrediction.Decision
```

### 5. Create dummy variables to deal with categorical inputs
One-hot encoding replaces each unique value of a given column with a new column, and puts a 1 in the new column for a given row just if its initial value for the original column matches the new column. Check out [this resource](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f) if you haven't seen one-hot-encoding before. 

**Note**: We will do this before we do our train/test split as to do it after could mean that some categories only end up in the train or test split of our data by chance and this would then lead to different shapes of data for our `X_train` and `X_test` which could/would cause downstream issues when fitting or predicting using a trained model.


```python
# One-hot encode all features in X.
import sklearn
from sklearn.preprocessing import OneHotEncoder
enc=sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')
enc.fit(X)
```




    OneHotEncoder(handle_unknown='ignore')



### 6. Further divide those subsets into train and test subsets for X and y respectively: X_train, X_test, y_train, y_test


```python
# Call train_test_split on X, y. Make the test_size = 0.25, and random_state = 246
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size = 0.25, random_state = 246)
```

# 3. Modelling
It's useful to look at the scikit-learn documentation on decision trees https://scikit-learn.org/stable/modules/tree.html before launching into applying them. If you haven't seen them before, take a look at that link, in particular the section `1.10.5.` 

## Model 1: Entropy model - no max_depth

We'll give you a little more guidance here, as the Python is hard to deduce, and scikitlearn takes some getting used to.

Theoretically, let's remind ourselves of what's going on with a decision tree implementing an entropy model.

Ross Quinlan's **ID3 Algorithm** was one of the first, and one of the most basic, to use entropy as a metric.

**Entropy** is a measure of how uncertain we are about which category the data-points fall into at a given point in the tree. The **Information gain** of a specific feature with a threshold (such as 'spent_last_month <= 138.0') is the difference in entropy that exists before and after splitting on that feature; i.e., the information we gain about the categories of the data-points by splitting on that feature and that threshold. 

Naturally, we want to minimize entropy and maximize information gain. Quinlan's ID3 algorithm is designed to output a tree such that the features at each node, starting from the root, and going all the way down to the leaves, have maximial information gain. We want a tree whose leaves have elements that are *homogeneous*, that is, all of the same category. 

The first model will be the hardest. Persevere and you'll reap the rewards: you can use almost exactly the same code for the other models. 


```python
entr_model.fit(X_train,y_train)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-442-06e82a27b140> in <module>
    ----> 1 entr_model.fit(X_train,y_train)
    

    C:\Users\toshiba\anaconda3\lib\site-packages\sklearn\tree\_classes.py in fit(self, X, y, sample_weight, check_input, X_idx_sorted)
        888         """
        889 
    --> 890         super().fit(
        891             X, y,
        892             sample_weight=sample_weight,
    

    C:\Users\toshiba\anaconda3\lib\site-packages\sklearn\tree\_classes.py in fit(self, X, y, sample_weight, check_input, X_idx_sorted)
        154             check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
        155             check_y_params = dict(ensure_2d=False, dtype=None)
    --> 156             X, y = self._validate_data(X, y,
        157                                        validate_separately=(check_X_params,
        158                                                             check_y_params))
    

    C:\Users\toshiba\anaconda3\lib\site-packages\sklearn\base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
        427                 # :(
        428                 check_X_params, check_y_params = validate_separately
    --> 429                 X = check_array(X, **check_X_params)
        430                 y = check_array(y, **check_y_params)
        431             else:
    

    C:\Users\toshiba\anaconda3\lib\site-packages\sklearn\utils\validation.py in inner_f(*args, **kwargs)
         70                           FutureWarning)
         71         kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
    ---> 72         return f(**kwargs)
         73     return inner_f
         74 
    

    C:\Users\toshiba\anaconda3\lib\site-packages\sklearn\utils\validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator)
        596                     array = array.astype(dtype, casting="unsafe", copy=False)
        597                 else:
    --> 598                     array = np.asarray(array, order=order, dtype=dtype)
        599             except ComplexWarning:
        600                 raise ValueError("Complex data not supported\n"
    

    C:\Users\toshiba\anaconda3\lib\site-packages\numpy\core\_asarray.py in asarray(a, dtype, order)
         81 
         82     """
    ---> 83     return array(a, dtype, copy=False, order=order)
         84 
         85 
    

    C:\Users\toshiba\anaconda3\lib\site-packages\pandas\core\generic.py in __array__(self, dtype)
       1779 
       1780     def __array__(self, dtype=None) -> np.ndarray:
    -> 1781         return np.asarray(self._values, dtype=dtype)
       1782 
       1783     def __array_wrap__(self, result, context=None):
    

    C:\Users\toshiba\anaconda3\lib\site-packages\numpy\core\_asarray.py in asarray(a, dtype, order)
         81 
         82     """
    ---> 83     return array(a, dtype, copy=False, order=order)
         84 
         85 
    

    ValueError: could not convert string to float: 'Female'



```python
# Declare a variable called entr_model and use tree.DecisionTreeClassifier. 
entr_model=tree.DecisionTreeClassifier(criterion='entropy',random_state = 100)

# Call fit() on entr_model
entr_model.fit(X_train,y_train)

# Call predict() on entr_model with X_test passed to it, and assign the result to a variable y_pred 
y_pred=entr_model.predict(X_test)

# Call Series on our y_pred variable with the following: pd.Series(y_pred)
y_pred=pd.Series(y_pred)

# Check out entr_model
entr_model
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-443-81b0ebd04ae2> in <module>
          3 
          4 # Call fit() on entr_model
    ----> 5 entr_model.fit(X_train,y_train)
          6 
          7 # Call predict() on entr_model with X_test passed to it, and assign the result to a variable y_pred
    

    C:\Users\toshiba\anaconda3\lib\site-packages\sklearn\tree\_classes.py in fit(self, X, y, sample_weight, check_input, X_idx_sorted)
        888         """
        889 
    --> 890         super().fit(
        891             X, y,
        892             sample_weight=sample_weight,
    

    C:\Users\toshiba\anaconda3\lib\site-packages\sklearn\tree\_classes.py in fit(self, X, y, sample_weight, check_input, X_idx_sorted)
        154             check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
        155             check_y_params = dict(ensure_2d=False, dtype=None)
    --> 156             X, y = self._validate_data(X, y,
        157                                        validate_separately=(check_X_params,
        158                                                             check_y_params))
    

    C:\Users\toshiba\anaconda3\lib\site-packages\sklearn\base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
        427                 # :(
        428                 check_X_params, check_y_params = validate_separately
    --> 429                 X = check_array(X, **check_X_params)
        430                 y = check_array(y, **check_y_params)
        431             else:
    

    C:\Users\toshiba\anaconda3\lib\site-packages\sklearn\utils\validation.py in inner_f(*args, **kwargs)
         70                           FutureWarning)
         71         kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
    ---> 72         return f(**kwargs)
         73     return inner_f
         74 
    

    C:\Users\toshiba\anaconda3\lib\site-packages\sklearn\utils\validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator)
        596                     array = array.astype(dtype, casting="unsafe", copy=False)
        597                 else:
    --> 598                     array = np.asarray(array, order=order, dtype=dtype)
        599             except ComplexWarning:
        600                 raise ValueError("Complex data not supported\n"
    

    C:\Users\toshiba\anaconda3\lib\site-packages\numpy\core\_asarray.py in asarray(a, dtype, order)
         81 
         82     """
    ---> 83     return array(a, dtype, copy=False, order=order)
         84 
         85 
    

    C:\Users\toshiba\anaconda3\lib\site-packages\pandas\core\generic.py in __array__(self, dtype)
       1779 
       1780     def __array__(self, dtype=None) -> np.ndarray:
    -> 1781         return np.asarray(self._values, dtype=dtype)
       1782 
       1783     def __array_wrap__(self, result, context=None):
    

    C:\Users\toshiba\anaconda3\lib\site-packages\numpy\core\_asarray.py in asarray(a, dtype, order)
         81 
         82     """
    ---> 83     return array(a, dtype, copy=False, order=order)
         84 
         85 
    

    ValueError: could not convert string to float: 'Female'



```python
# Now we want to visualize the tree
_ _ _

# We can do so with export_graphviz
_ _ _ 

# Alternatively for class_names use entr_model.classes_
_ _ _
```


      File "<ipython-input-444-ea9078cf8b36>", line 2
        _ _ _
          ^
    SyntaxError: invalid syntax
    


## Model 1: Entropy model - no max_depth: Interpretation and evaluation


```python
# Run this block for model evaluation metrics 
print("Model Entropy - no max depth")
print("Accuracy:", metrics.accuracy_score(y_test,y_pred))
print("Balanced accuracy:", metrics.balanced_accuracy_score(y_test,y_pred))
print('Precision score for "Yes"' , metrics.precision_score(y_test,y_pred, pos_label = "YES"))
print('Precision score for "No"' , metrics.precision_score(y_test,y_pred, pos_label = "NO"))
print('Recall score for "Yes"' , metrics.recall_score(y_test,y_pred, pos_label = "YES"))
print('Recall score for "No"' , metrics.recall_score(y_test,y_pred, pos_label = "NO"))
```

What can you infer from these results? Write your conclusions here:

## Model 2: Gini impurity model - no max_depth

Gini impurity, like entropy, is a measure of how well a given feature (and threshold) splits the data into categories.

Their equations are similar, but Gini impurity doesn't require logorathmic functions, which can be computationally expensive. 


```python
# Make a variable called gini_model, and assign it exactly what you assigned entr_model with above, but with the
# criterion changed to 'gini'
_ _ _ 

# Call fit() on the gini_model as you did with the entr_model
_ _ _ 

# Call predict() on the gini_model as you did with the entr_model 
_ _ _ 

# Turn y_pred into a series, as before
_ _ _ 

# Check out gini_model
_ _ _ 
```


```python
# As before, but make the model name gini_model
_ _ _ 
_ _ _ 

# Alternatively for class_names use gini_model.classes_
_ _ _ 
```


```python
# Run this block for model evaluation
print("Model Gini impurity model")
print("Accuracy:", metrics.accuracy_score(y_test,y_pred))
print("Balanced accuracy:", metrics.balanced_accuracy_score(y_test,y_pred))
print('Precision score' , metrics.precision_score(y_test,y_pred, pos_label = "YES"))
print('Recall score' , metrics.recall_score(y_test,y_pred, pos_label = "NO"))
```

How do the results here compare to the previous model? Write your judgements here: 

## Model 3: Entropy model - max depth 3
We're going to try to limit the depth of our decision tree, using entropy first.  

As you know, we need to strike a balance with tree depth. 

Insufficiently deep, and we're not giving the tree the opportunity to spot the right patterns in the training data.

Excessively deep, and we're probably going to make a tree that overfits to the training data, at the cost of very high error on the (hitherto unseen) test data. 

Sophisticated data scientists use methods like random search with cross-validation to systematically find a good depth for their tree. We'll start with picking 3, and see how that goes. 


```python
# Made a model as before, but call it entr_model2, and make the max_depth parameter equal to 3. 
# Execute the fitting, predicting, and Series operations as before
_ _ _
```


```python
# As before, we need to visualize the tree to grasp its nature
_ _ _ 

# Alternatively for class_names use entr_model2.classes_
_ _ _
```


```python
# Run this block for model evaluation 
print("Model Entropy model max depth 3")
print("Accuracy:", metrics.accuracy_score(y_test,y_pred))
print("Balanced accuracy:", metrics.balanced_accuracy_score(y_test,y_pred))
print('Precision score for "Yes"' , metrics.precision_score(y_test,y_pred, pos_label = "YES"))
print('Recall score for "No"' , metrics.recall_score(y_test,y_pred, pos_label = "NO"))
```

So our accuracy decreased, but is this certainly an inferior tree to the max depth original tree we did with Model 1? Write your conclusions here: 

## Model 4: Gini impurity  model - max depth 3
We're now going to try the same with the Gini impurity model. 


```python
# As before, make a variable, but call it gini_model2, and ensure the max_depth parameter is set to 3
_ _ _ = _ _ _._ _ _(_ _ _ ='_ _ _ ', _ _ _ = 1234, _ _ _  = _ _ _ )

# Do the fit, predict, and series transformations as before. 
_ _ _
```


```python
dot_data = StringIO()
_ _ _ 


# Alternatively for class_names use gini_model2.classes_
_ _ _
```


```python
print("Gini impurity  model - max depth 3")
print("Accuracy:", metrics.accuracy_score(y_test,y_pred))
print("Balanced accuracy:", metrics.balanced_accuracy_score(y_test,y_pred))
print('Precision score' , metrics.precision_score(y_test,y_pred, pos_label = "YES"))
print('Recall score' , metrics.recall_score(y_test,y_pred, pos_label = "NO"))
```

Now this is an elegant tree. Its accuracy might not be the highest, but it's still the best model we've produced so far. Why is that? Write your answer here: 

# 4. Evaluating and concluding
## 4a. How many customers will buy Hidden Farm coffee? 
Let's first ascertain how many loyal customers claimed, in the survey, that they will purchase the Hidden Farm coffee. 


```python
# Call value_counts() on the 'Decision' column of the original coffeeData
_ _ _ 
```

Let's now determine the number of people that, according to the model, will be willing to buy the Hidden Farm coffee. 
1. First we subset the Prediction dataset into `new_X` considering all the variables except `Decision` 
2. Use that dataset to predict a new variable called `potential_buyers`


```python
# Feature selection
# Make a variable called feature_cols, and assign it a list containing all the column names except 'Decision'
_ _ _ 

# Make a variable called new_X, and assign it the subset of Prediction, containing just the feature_cols 
_ _ _ 
```


```python
# Call get_dummies() on the Pandas object pd, with new_X plugged in, to one-hot encode all features in the training set
_ _ _ 

# Make a variable called potential_buyers, and assign it the result of calling predict() on a model of your choice; 
# don't forget to pass new_X to predict()
_ _ _ 
```


```python
# Let's get the numbers of YES's and NO's in the potential buyers 
# Call unique() on np, and pass potential_buyers and return_counts=True 
_ _ _ 
```

The total number of potential buyers is 303 + 183 = 486


```python
# Print the total number of surveyed people 
_ _ _ 
```


```python
# Let's calculate the proportion of buyers
_ _ _ 
```


```python
# Print the percentage of people who want to buy the Hidden Farm coffee, by our model 
_ _ _ 
```

## 4b. Decision
Remember how you thought at the start: if more than 70% of the interviewed customers are likely to buy the Hidden Farm coffee, you will strike the deal with the local Hidden Farm farmers and sell the coffee. Otherwise, you won't strike the deal and the Hidden Farm coffee will remain in legends only. Well now's crunch time. Are you going to go ahead with that idea? If so, you won't be striking the deal with the Chinese farmers. 

They're called `decision trees`, aren't they? So where's the decision? What should you do? (Cue existential cat emoji). 

Ultimately, though, we can't write an algorithm to actually *make the business decision* for us. This is because such decisions depend on our values, what risks we are willing to take, the stakes of our decisions, and how important it us for us to *know* that we will succeed. What are you going to do with the models you've made? Are you going to risk everything, strike the deal with the *Hidden Farm* farmers, and sell the coffee? 

The philosopher of language Jason Stanley once wrote that the number of doubts our evidence has to rule out in order for us to know a given proposition depends on our stakes: the higher our stakes, the more doubts our evidence has to rule out, and therefore the harder it is for us to know things. We can end up paralyzed in predicaments; sometimes, we can act to better our situation only if we already know certain things, which we can only if our stakes were lower and we'd *already* bettered our situation. 

Data science and machine learning can't solve such problems. But what it can do is help us make great use of our data to help *inform* our decisions.

## 5. Random Forest
You might have noticed an important fact about decision trees. Each time we run a given decision tree algorithm to make a prediction (such as whether customers will buy the Hidden Farm coffee) we will actually get a slightly different result. This might seem weird, but it has a simple explanation: machine learning algorithms are by definition ***stochastic***, in that their output is at least partly determined by randomness. 

To account for this variability and ensure that we get the most accurate prediction, we might want to actually make lots of decision trees, and get a value that captures the centre or average of the outputs of those trees. Luckily, there's a method for this, known as the ***Random Forest***. 

Essentially, Random Forest involves making lots of trees with similar properties, and then performing summary statistics on the outputs of those trees to reach that central value. Random forests are hugely powerful classifers, and they can improve predictive accuracy and control over-fitting. 

Why not try to inform your decision with random forest? You'll need to make use of the RandomForestClassifier function within the sklearn.ensemble module, found [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). 

### 5a. Import necessary modules


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
```

### 5b. Model
You'll use your X_train and y_train variables just as before.

You'll then need to make a variable (call it firstRFModel) to store your new Random Forest model. You'll assign this variable the result of calling RandomForestClassifier().

Then, just as before, you'll call fit() on that firstRFModel variable, and plug in X_train and y_train.

Finally, you should make a variable called y_pred, and assign it the result of calling the predict() method on your new firstRFModel, with the X_test data passed to it. 


```python
# Plug in appropriate max_depth and random_state parameters 
_ _ _ 

# Model and fit
_ _ _ 



```

### 5c. Revise conclusion

Has your conclusion changed? Or is the result of executing random forest the same as your best model reached by a single decision tree? 
