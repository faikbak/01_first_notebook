# Springboard Regression Case Study - The Red Wine Dataset - Tier 3

Welcome to the Springboard Regression case study! Please note: this is ***Tier 3*** of the case study.

This case study was designed for you to **use Python to apply the knowledge you've acquired in reading *The Art of Statistics* (hereinafter *AoS*) by Professor Spiegelhalter**. Specifically, the case study will get you doing regression analysis; a method discussed in Chapter 5 on p.121. It might be useful to have the book open at that page when doing the case study to remind you of what it is we're up to (but bear in mind that other statistical concepts, such as training and testing, will be applied, so you might have to glance at other chapters too).  

The aim is to ***use exploratory data analysis (EDA) and regression to predict alcohol levels in wine with a model that's as accurate as possible***. 

We'll try a *univariate* analysis (one involving a single explanatory variable) as well as a *multivariate* one (involving multiple explanatory variables), and we'll iterate together towards a decent model by the end of the notebook. The main thing is for you to see how regression analysis looks in Python and jupyter, and to get some practice implementing this analysis.

Throughout this case study, **questions** will be asked in the markdown cells. Try to **answer these yourself in a simple text file** when they come up. Most of the time, the answers will become clear as you progress through the notebook. Some of the answers may require a little research with Google and other basic resources available to every data scientist. 

For this notebook, we're going to use the red wine dataset, wineQualityReds.csv. Make sure it's downloaded and sitting in your working directory. This is a very common dataset for practicing regression analysis and is actually freely available on Kaggle, [here](https://www.kaggle.com/piyushgoyal443/red-wine-dataset).

You're pretty familiar with the data science pipeline at this point. This project will have the following structure: 
**1. Sourcing and loading** 
- Import relevant libraries
- Load the data 
- Exploring the data
- Choosing a dependent variable
 
**2. Cleaning, transforming, and visualizing**
- Visualizing correlations
  
  
**3. Modeling** 
- Train/Test split
- Making a Linear regression model: your first model
- Making a Linear regression model: your second model: Ordinary Least Squares (OLS) 
- Making a Linear regression model: your third model: multiple linear regression
- Making a Linear regression model: your fourth model: avoiding redundancy

**4. Evaluating and concluding** 
- Reflection 
- Which model was best?
- Other regression algorithms

### 1. Sourcing and loading

#### 1a. Import relevant libraries 


```python
# Import relevant libraries and packages.
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns # For all our visualization needs.
import statsmodels.api as sm # What does this do? Find out and type here.
from statsmodels.graphics.api import abline_plot # What does this do? Find out and type here.
from sklearn.metrics import mean_squared_error, r2_score # What does this do? Find out and type here.
from sklearn.model_selection import train_test_split #  What does this do? Find out and type here.
from sklearn import linear_model, preprocessing # What does this do? Find out and type here.
import warnings # For handling error messages.
# Don't worry about the following two instructions: they just suppress warnings that could occur later. 
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
```

#### 1b. Load the data


```python
# Load the data. 
wine=pd.read_csv('wineQualityReds.csv')
```

#### 1c. Exploring the data


```python
# Check out its appearance. 
wine.head()
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
      <th>Unnamed: 0</th>
      <th>fixed.acidity</th>
      <th>volatile.acidity</th>
      <th>citric.acid</th>
      <th>residual.sugar</th>
      <th>chlorides</th>
      <th>free.sulfur.dioxide</th>
      <th>total.sulfur.dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Another very useful method to call on a recently imported dataset is .info(). Call it here to get a good
# overview of the data
wine.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1599 entries, 0 to 1598
    Data columns (total 13 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   Unnamed: 0            1599 non-null   int64  
     1   fixed.acidity         1599 non-null   float64
     2   volatile.acidity      1599 non-null   float64
     3   citric.acid           1599 non-null   float64
     4   residual.sugar        1599 non-null   float64
     5   chlorides             1599 non-null   float64
     6   free.sulfur.dioxide   1599 non-null   float64
     7   total.sulfur.dioxide  1599 non-null   float64
     8   density               1599 non-null   float64
     9   pH                    1599 non-null   float64
     10  sulphates             1599 non-null   float64
     11  alcohol               1599 non-null   float64
     12  quality               1599 non-null   int64  
    dtypes: float64(11), int64(2)
    memory usage: 162.5 KB
    

What can you infer about the nature of these variables, as output by the info() method?

Which variables might be suitable for regression analysis, and why? For those variables that aren't suitable for regression analysis, is there another type of statistical modeling for which they are suitable?


```python
# We should also look more closely at the dimensions of the dataset. 
np.shape(wine)
```




    (1599, 13)



#### 1d. Choosing a dependent variable

We now need to pick a dependent variable for our regression analysis: a variable whose values we will predict. 

'Quality' seems to be as good a candidate as any. Let's check it out. One of the quickest and most informative ways to understand a variable is to make a histogram of it. This gives us an idea of both the center and spread of its values. 


```python
# Making a histogram of the quality variable.
plt.hist(wine['quality'])
```




    (array([ 10.,   0.,  53.,   0., 681.,   0., 638.,   0., 199.,  18.]),
     array([3. , 3.5, 4. , 4.5, 5. , 5.5, 6. , 6.5, 7. , 7.5, 8. ]),
     <BarContainer object of 10 artists>)




    
![png](output_14_1.png)
    


We can see so much about the quality variable just from this simple visualization. Answer yourself: what value do most wines have for quality? What is the minimum quality value below, and the maximum quality value? What is the range? Remind yourself of these summary statistical concepts by looking at p.49 of the *AoS*.

But can you think of a problem with making this variable the dependent variable of regression analysis? Remember the example in *AoS* on p.122 of predicting the heights of children from the heights of parents? Take a moment here to think about potential problems before reading on. 

The issue is this: quality is a *discrete* variable, in that its values are integers (whole numbers) rather than floating point numbers. Thus, quality is not a *continuous* variable. But this means that it's actually not the best target for regression analysis. 

Before we dismiss the quality variable, however, let's verify that it is indeed a discrete variable with some further exploration. 


```python
# Get a basic statistical summary of the variable 
wine['quality'].describe()

# What do you notice from this summary? 
#all values are discret
```




    count    1599.000000
    mean        5.636023
    std         0.807569
    min         3.000000
    25%         5.000000
    50%         6.000000
    75%         6.000000
    max         8.000000
    Name: quality, dtype: float64




```python
# Get a list of the values of the quality variable, and the number of occurrences of each. 
wine['quality'].value_counts()   
```




    5    681
    6    638
    7    199
    4     53
    8     18
    3     10
    Name: quality, dtype: int64



The outputs of the describe() and value_counts() methods are consistent with our histogram, and since there are just as many values as there are rows in the dataset, we can infer that there are no NAs for the quality variable. 

But scroll up again to when we called info() on our wine dataset. We could have seen there, already, that the quality variable had int64 as its type. As a result, we had sufficient information, already, to know that the quality variable was not appropriate for regression analysis. Did you figure this out yourself? If so, kudos to you!

The quality variable would, however, conduce to proper classification analysis. This is because, while the values for the quality variable are numeric, those numeric discrete values represent *categories*; and the prediction of category-placement is most often best done by classification algorithms. You saw the decision tree output by running a classification algorithm on the Titanic dataset on p.168 of Chapter 6 of *AoS*. For now, we'll continue with our regression analysis, and continue our search for a suitable dependent variable. 

Now, since the rest of the variables of our wine dataset are continuous, we could — in theory — pick any of them. But that does not mean that are all equally sutiable choices. What counts as a suitable dependent variable for regression analysis is determined not just by *intrinsic* features of the dataset (such as data types, number of NAs etc) but by *extrinsic* features, such as, simply, which variables are the most interesting or useful to predict, given our aims and values in the context we're in. Almost always, we can only determine which variables are sensible choices for dependent variables with some **domain knowledge**. 

Not all of you might be wine buffs, but one very important and interesting quality in wine is [acidity](https://waterhouse.ucdavis.edu/whats-in-wine/fixed-acidity). As the Waterhouse Lab at the University of California explains, 'acids impart the sourness or tartness that is a fundamental feature in wine taste.  Wines lacking in acid are "flat." Chemically the acids influence titrable acidity which affects taste and pH which affects  color, stability to oxidation, and consequantly the overall lifespan of a wine.'

If we cannot predict quality, then it seems like **fixed acidity** might be a great option for a dependent variable. Let's go for that.

So if we're going for fixed acidity as our dependent variable, what we now want to get is an idea of *which variables are related interestingly to that dependent variable*. 

We can call the .corr() method on our wine data to look at all the correlations between our variables. As the [documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html) shows, the default correlation coefficient is the Pearson correlation coefficient (p.58 and p.396 of the *AoS*); but other coefficients can be plugged in as parameters. Remember, the Pearson correlation coefficient shows us how close to a straight line the data-points fall, and is a number between -1 and 1. 


```python
# Call the .corr() method on the wine dataset 
wine.corr()
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
      <th>Unnamed: 0</th>
      <th>fixed.acidity</th>
      <th>volatile.acidity</th>
      <th>citric.acid</th>
      <th>residual.sugar</th>
      <th>chlorides</th>
      <th>free.sulfur.dioxide</th>
      <th>total.sulfur.dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Unnamed: 0</th>
      <td>1.000000</td>
      <td>-0.268484</td>
      <td>-0.008815</td>
      <td>-0.153551</td>
      <td>-0.031261</td>
      <td>-0.119869</td>
      <td>0.090480</td>
      <td>-0.117850</td>
      <td>-0.368372</td>
      <td>0.136005</td>
      <td>-0.125307</td>
      <td>0.245123</td>
      <td>0.066453</td>
    </tr>
    <tr>
      <th>fixed.acidity</th>
      <td>-0.268484</td>
      <td>1.000000</td>
      <td>-0.256131</td>
      <td>0.671703</td>
      <td>0.114777</td>
      <td>0.093705</td>
      <td>-0.153794</td>
      <td>-0.113181</td>
      <td>0.668047</td>
      <td>-0.682978</td>
      <td>0.183006</td>
      <td>-0.061668</td>
      <td>0.124052</td>
    </tr>
    <tr>
      <th>volatile.acidity</th>
      <td>-0.008815</td>
      <td>-0.256131</td>
      <td>1.000000</td>
      <td>-0.552496</td>
      <td>0.001918</td>
      <td>0.061298</td>
      <td>-0.010504</td>
      <td>0.076470</td>
      <td>0.022026</td>
      <td>0.234937</td>
      <td>-0.260987</td>
      <td>-0.202288</td>
      <td>-0.390558</td>
    </tr>
    <tr>
      <th>citric.acid</th>
      <td>-0.153551</td>
      <td>0.671703</td>
      <td>-0.552496</td>
      <td>1.000000</td>
      <td>0.143577</td>
      <td>0.203823</td>
      <td>-0.060978</td>
      <td>0.035533</td>
      <td>0.364947</td>
      <td>-0.541904</td>
      <td>0.312770</td>
      <td>0.109903</td>
      <td>0.226373</td>
    </tr>
    <tr>
      <th>residual.sugar</th>
      <td>-0.031261</td>
      <td>0.114777</td>
      <td>0.001918</td>
      <td>0.143577</td>
      <td>1.000000</td>
      <td>0.055610</td>
      <td>0.187049</td>
      <td>0.203028</td>
      <td>0.355283</td>
      <td>-0.085652</td>
      <td>0.005527</td>
      <td>0.042075</td>
      <td>0.013732</td>
    </tr>
    <tr>
      <th>chlorides</th>
      <td>-0.119869</td>
      <td>0.093705</td>
      <td>0.061298</td>
      <td>0.203823</td>
      <td>0.055610</td>
      <td>1.000000</td>
      <td>0.005562</td>
      <td>0.047400</td>
      <td>0.200632</td>
      <td>-0.265026</td>
      <td>0.371260</td>
      <td>-0.221141</td>
      <td>-0.128907</td>
    </tr>
    <tr>
      <th>free.sulfur.dioxide</th>
      <td>0.090480</td>
      <td>-0.153794</td>
      <td>-0.010504</td>
      <td>-0.060978</td>
      <td>0.187049</td>
      <td>0.005562</td>
      <td>1.000000</td>
      <td>0.667666</td>
      <td>-0.021946</td>
      <td>0.070377</td>
      <td>0.051658</td>
      <td>-0.069408</td>
      <td>-0.050656</td>
    </tr>
    <tr>
      <th>total.sulfur.dioxide</th>
      <td>-0.117850</td>
      <td>-0.113181</td>
      <td>0.076470</td>
      <td>0.035533</td>
      <td>0.203028</td>
      <td>0.047400</td>
      <td>0.667666</td>
      <td>1.000000</td>
      <td>0.071269</td>
      <td>-0.066495</td>
      <td>0.042947</td>
      <td>-0.205654</td>
      <td>-0.185100</td>
    </tr>
    <tr>
      <th>density</th>
      <td>-0.368372</td>
      <td>0.668047</td>
      <td>0.022026</td>
      <td>0.364947</td>
      <td>0.355283</td>
      <td>0.200632</td>
      <td>-0.021946</td>
      <td>0.071269</td>
      <td>1.000000</td>
      <td>-0.341699</td>
      <td>0.148506</td>
      <td>-0.496180</td>
      <td>-0.174919</td>
    </tr>
    <tr>
      <th>pH</th>
      <td>0.136005</td>
      <td>-0.682978</td>
      <td>0.234937</td>
      <td>-0.541904</td>
      <td>-0.085652</td>
      <td>-0.265026</td>
      <td>0.070377</td>
      <td>-0.066495</td>
      <td>-0.341699</td>
      <td>1.000000</td>
      <td>-0.196648</td>
      <td>0.205633</td>
      <td>-0.057731</td>
    </tr>
    <tr>
      <th>sulphates</th>
      <td>-0.125307</td>
      <td>0.183006</td>
      <td>-0.260987</td>
      <td>0.312770</td>
      <td>0.005527</td>
      <td>0.371260</td>
      <td>0.051658</td>
      <td>0.042947</td>
      <td>0.148506</td>
      <td>-0.196648</td>
      <td>1.000000</td>
      <td>0.093595</td>
      <td>0.251397</td>
    </tr>
    <tr>
      <th>alcohol</th>
      <td>0.245123</td>
      <td>-0.061668</td>
      <td>-0.202288</td>
      <td>0.109903</td>
      <td>0.042075</td>
      <td>-0.221141</td>
      <td>-0.069408</td>
      <td>-0.205654</td>
      <td>-0.496180</td>
      <td>0.205633</td>
      <td>0.093595</td>
      <td>1.000000</td>
      <td>0.476166</td>
    </tr>
    <tr>
      <th>quality</th>
      <td>0.066453</td>
      <td>0.124052</td>
      <td>-0.390558</td>
      <td>0.226373</td>
      <td>0.013732</td>
      <td>-0.128907</td>
      <td>-0.050656</td>
      <td>-0.185100</td>
      <td>-0.174919</td>
      <td>-0.057731</td>
      <td>0.251397</td>
      <td>0.476166</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Ok - you might be thinking, but wouldn't it be nice if we visualized these relationships? It's hard to get a picture of the correlations between the variables without anything visual. 

Very true, and this brings us to the next section.

### 2. Cleaning, Transforming, and Visualizing 

#### 2a. Visualizing correlations 
The heading of this stage of the data science pipeline ('Cleaning, Transforming, and Visualizing') doesn't imply that we have to do all of those operations in *that order*. Sometimes (and this is a case in point) our data is already relatively clean, and the priority is to do some visualization. Normally, however, our data is less sterile, and we have to do some cleaning and transforming first prior to visualizing. 

Now that we've chosen **fixed acidity** as our dependent variable for regression analysis, we can begin by plotting the pairwise relationships in the dataset, to check out how our variables relate to one another.


```python
# Make a pairplot of the wine data
sns.pairplot(wine)
```




    <seaborn.axisgrid.PairGrid at 0x2684e3f77f0>




    
![png](output_25_1.png)
    


If you've never executed your own Seaborn pairplot before, just take a moment to look at the output. They certainly output a lot of information at once. What can you infer from it? What can you *not* justifiably infer from it?

... All done? 

Here's a couple things you might have noticed: 
- a given cell value represents the correlation that exists between two variables 
- on the diagonal, you can see a bunch of histograms. This is because pairplotting the variables with themselves would be pointless, so the pairplot() method instead makes histograms to show the distributions of those variables' values. This allows us to quickly see the shape of each variable's values.  
- the plots for the quality variable form horizontal bands, due to the fact that it's a discrete variable. We were certainly right in not pursuing a regression analysis of this variable.
- Notice that some of the nice plots invite a line of best fit, such as alcohol vs density. Others, such as citric acid vs alcohol, are more inscrutable.

So we now have called the .corr() method, and the .pairplot() Seaborn method, on our wine data. Both have flaws. Happily, we can get the best of both worlds with a heatmap. 


```python
# Make a heatmap of the data 
sns.heatmap(wine.corr())

```




    <AxesSubplot:>




    
![png](output_28_1.png)
    


Take a moment to think about the following questions:
- How does color relate to extent of correlation?
- How might we use the plot to show us interesting relationships worth investigating? 
- More precisely, what does the heatmap show us about the fixed acidity variable's relationship to the density variable? 

There is a relatively strong correlation between the density and fixed acidity variables respectively. In the next code block, call the scatterplot() method on our sns object. Make the x-axis parameter 'density', the y-axis parameter 'fixed.acidity', and the third parameter specify our wine dataset.  


```python
# Plot density against fixed.acidity
plt.scatter(wine['density'],wine['fixed.acidity'])
```




    <matplotlib.collections.PathCollection at 0x26855181df0>




    
![png](output_30_1.png)
    


We can see a positive correlation, and quite a steep one. There are some outliers, but as a whole, there is a steep looking line that looks like it ought to be drawn. 


```python
# Call the regplot method on your sns object, with parameters: x = 'density', y = 'fixed.acidity'
sns.regplot(data=wine,x=wine['density'],y=wine['fixed.acidity'])
```




    <AxesSubplot:xlabel='density', ylabel='fixed.acidity'>




    
![png](output_32_1.png)
    


The line of best fit matches the overall shape of the data, but it's clear that there are some points that deviate from the line, rather than all clustering close. 

Let's see if we can predict fixed acidity based on density using linear regression. 

### 3. Modeling 

#### 3a. Train/Test Split
While this dataset is super clean, and hence doesn't require much for analysis, we still need to split our dataset into a test set and a training set.

You'll recall from p.158 of *AoS* that such a split is important good practice when evaluating statistical models. On p.158, Professor Spiegelhalter was evaluating a classification tree, but the same applies when we're doing regression. Normally, we train with 75% of the data and test on the remaining 25%. 

To be sure, for our first model, we're only going to focus on two variables: fixed acidity as our dependent variable, and density as our sole independent predictor variable. 

We'll be using [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) here. Don't worry if not all of the syntax makes sense; just follow the rationale for what we're doing. 


```python
# Subsetting our data into our dependent and independent variables.
dependent=wine[['fixed.acidity']]
independent=wine[['density']]
# Split the data. This line uses the sklearn function train_test_split().
# The test_size parameter means we can train with 75% of the data, and test on 25%. 
X_train,X_test,Y_train,Y_test = train_test_split(dependent,independent,test_size=.25,random_state=42)
```


```python
# We now want to check the shape of the X train, y_train, X_test and y_test to make sure the proportions are right. 
X_train.shape, Y_train.shape, X_test.shape, Y_test.shape

```




    ((1199, 1), (1199, 1), (400, 1), (400, 1))



#### 3b. Making a Linear Regression model: our first model
Sklearn has a [LinearRegression()](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) function built into the linear_model module. We'll be using that to make our regression model. 


```python
# Create the model: make a variable called rModel, and use it linear_model.LinearRegression appropriately
rModel1 = linear_model.LinearRegression()
```


```python
# Evaluate the model  
rModel1.score(X_train,Y_train)
```


    ---------------------------------------------------------------------------

    NotFittedError                            Traceback (most recent call last)

    <ipython-input-153-1ac30703b788> in <module>
          1 # Evaluate the model
    ----> 2 rModel1.score(X_train,Y_train)
    

    C:\Users\toshiba\anaconda3\lib\site-packages\sklearn\base.py in score(self, X, y, sample_weight)
        549 
        550         from .metrics import r2_score
    --> 551         y_pred = self.predict(X)
        552         return r2_score(y, y_pred, sample_weight=sample_weight)
        553 
    

    C:\Users\toshiba\anaconda3\lib\site-packages\sklearn\linear_model\_base.py in predict(self, X)
        234             Returns predicted values.
        235         """
    --> 236         return self._decision_function(X)
        237 
        238     _preprocess_data = staticmethod(_preprocess_data)
    

    C:\Users\toshiba\anaconda3\lib\site-packages\sklearn\linear_model\_base.py in _decision_function(self, X)
        214 
        215     def _decision_function(self, X):
    --> 216         check_is_fitted(self)
        217 
        218         X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
    

    C:\Users\toshiba\anaconda3\lib\site-packages\sklearn\utils\validation.py in inner_f(*args, **kwargs)
         70                           FutureWarning)
         71         kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
    ---> 72         return f(**kwargs)
         73     return inner_f
         74 
    

    C:\Users\toshiba\anaconda3\lib\site-packages\sklearn\utils\validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
       1017 
       1018     if not attrs:
    -> 1019         raise NotFittedError(msg % {'name': type(estimator).__name__})
       1020 
       1021 
    

    NotFittedError: This LinearRegression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.


The above score is called R-Squared coefficient, or the "coefficient of determination". It's basically a measure of how successfully our model predicts the variations in the data away from the mean: 1 would mean a perfect model that explains 100% of the variation. At the moment, our model explains only about 45% of the variation from the mean. There's more work to do!


```python
# Use the model to make predictions about our test data
pred1= rModel1.predict(X_test)
```


    ---------------------------------------------------------------------------

    NotFittedError                            Traceback (most recent call last)

    <ipython-input-154-b29e541ee071> in <module>
          1 # Use the model to make predictions about our test data
    ----> 2 pred1= rModel1.predict(X_test)
    

    C:\Users\toshiba\anaconda3\lib\site-packages\sklearn\linear_model\_base.py in predict(self, X)
        234             Returns predicted values.
        235         """
    --> 236         return self._decision_function(X)
        237 
        238     _preprocess_data = staticmethod(_preprocess_data)
    

    C:\Users\toshiba\anaconda3\lib\site-packages\sklearn\linear_model\_base.py in _decision_function(self, X)
        214 
        215     def _decision_function(self, X):
    --> 216         check_is_fitted(self)
        217 
        218         X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
    

    C:\Users\toshiba\anaconda3\lib\site-packages\sklearn\utils\validation.py in inner_f(*args, **kwargs)
         70                           FutureWarning)
         71         kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
    ---> 72         return f(**kwargs)
         73     return inner_f
         74 
    

    C:\Users\toshiba\anaconda3\lib\site-packages\sklearn\utils\validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
       1017 
       1018     if not attrs:
    -> 1019         raise NotFittedError(msg % {'name': type(estimator).__name__})
       1020 
       1021 
    

    NotFittedError: This LinearRegression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.



```python
# Let's plot the predictions against the actual result. Use scatter()
plt.scatter(Y_test,pred1)
```




    <matplotlib.collections.PathCollection at 0x26857acd310>




    
![png](output_44_1.png)
    


The above scatterplot represents how well the predictions match the actual results. 

Along the x-axis, we have the actual fixed acidity, and along the y-axis we have the predicted value for the fixed acidity.

There is a visible positive correlation, as the model has not been totally unsuccesful, but it's clear that it is not maximally accurate: wines with an actual fixed acidity of just over 10 have been predicted as having acidity levels from about 6.3 to 13.

Let's build a similar model using a different package, to see if we get a better result that way.

#### 3c. Making a Linear Regression model: our second model: Ordinary Least Squares (OLS)


```python
# Create the test and train sets. Here, we do things slightly differently.  
# We make the explanatory variable X as before.
X=wine["density"] 

# But here, reassign X the value of adding a constant to it. This is required for Ordinary Least Squares Regression.
# Further explanation of this can be found here: 
# https://www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.OLS.html
X=sm.add_constant(X)
```


```python
# The rest of the preparation is as before.
dependent=wine[['fixed.acidity']]
independent=wine[['density']]

X_train,X_test,Y_train,Y_test=train_test_split(dependent,independent,test_size=.25,random_state=42)
```


```python
# Create the model
rModel_2 = sm.OLS(Y_train,X_train)

# Fit the model with fit() 
rModel2=rModel_2.fit()
```


```python
# Evaluate the model with .summary()
rModel2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>density</td>     <th>  R-squared (uncentered):</th>      <td>   0.960</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared (uncentered):</th> <td>   0.960</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>          <td>2.857e+04</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 11 May 2021</td> <th>  Prob (F-statistic):</th>           <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>12:43:11</td>     <th>  Log-Likelihood:    </th>          <td>  228.56</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1199</td>      <th>  AIC:               </th>          <td>  -455.1</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1198</td>      <th>  BIC:               </th>          <td>  -450.0</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>              <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>              <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>fixed.acidity</th> <td>    0.1150</td> <td>    0.001</td> <td>  169.015</td> <td> 0.000</td> <td>    0.114</td> <td>    0.116</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>168.596</td> <th>  Durbin-Watson:     </th> <td>   1.959</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 256.528</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.971</td>  <th>  Prob(JB):          </th> <td>1.97e-56</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.167</td>  <th>  Cond. No.          </th> <td>    1.00</td>
</tr>
</table><br/><br/>Notes:<br/>[1] R² is computed without centering (uncentered) since the model does not contain a constant.<br/>[2] Standard Errors assume that the covariance matrix of the errors is correctly specified.



One of the great things about Statsmodels (sm) is that you get so much information from the summary() method. 

There are lots of values here, whose meanings you can explore at your leisure, but here's one of the most important: the R-squared score is 0.455, the same as what it was with the previous model. This makes perfect sense, right? It's the same value as the score from sklearn, because they've both used the same algorithm on the same data.

Here's a useful link you can check out if you have the time: https://www.theanalysisfactor.com/assessing-the-fit-of-regression-models/


```python
# Let's use our new model to make predictions of the dependent variable y. Use predict(), and plug in X_test as the parameter
pred2=rModel2.predict(X_test)
```


```python
# Plot the predictions
# Build a scatterplot
plt.scatter(Y_test,pred1)

# Add a line for perfect correlation. Can you see what this line is doing? Use plot()
plt.plot([x for x in range(9, 15)], [x for x in range(9, 15)], color = 'red')
```




    [<matplotlib.lines.Line2D at 0x26857a88400>]




    
![png](output_54_1.png)
    


The red line shows a theoretically perfect correlation between our actual and predicted values - the line that would exist if every prediction was completely correct. It's clear that while our points have a generally similar direction, they don't match the red line at all; we still have more work to do. 

To get a better predictive model, we should use more than one variable.

#### 3d. Making a Linear Regression model: our third model: multiple linear regression
Remember, as Professor Spiegelhalter explains on p.132 of *AoS*, including more than one explanatory variable into a linear regression analysis is known as ***multiple linear regression***. 


```python
# Create test and train datasets
# This is again very similar, but now we include more columns in the predictors
# Include all columns from data in the explanatory variables X except fixed.acidity and quality (which was an integer)
X=wine.drop(wine[['fixed.acidity','quality']],axis=1)
```


```python
# We can use almost identical code to create the third model, because it is the same algorithm, just different inputs
rModel_3 = sm.OLS(Y_train,X_train)

# Fit the model with fit() 
r_fit3=rModel_3.fit()
```


```python
# Evaluate the model
r_fit3.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>density</td>     <th>  R-squared (uncentered):</th>      <td>   0.960</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared (uncentered):</th> <td>   0.960</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>          <td>2.857e+04</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 11 May 2021</td> <th>  Prob (F-statistic):</th>           <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>12:44:58</td>     <th>  Log-Likelihood:    </th>          <td>  228.56</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1199</td>      <th>  AIC:               </th>          <td>  -455.1</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1198</td>      <th>  BIC:               </th>          <td>  -450.0</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>              <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>              <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>fixed.acidity</th> <td>    0.1150</td> <td>    0.001</td> <td>  169.015</td> <td> 0.000</td> <td>    0.114</td> <td>    0.116</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>168.596</td> <th>  Durbin-Watson:     </th> <td>   1.959</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 256.528</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.971</td>  <th>  Prob(JB):          </th> <td>1.97e-56</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.167</td>  <th>  Cond. No.          </th> <td>    1.00</td>
</tr>
</table><br/><br/>Notes:<br/>[1] R² is computed without centering (uncentered) since the model does not contain a constant.<br/>[2] Standard Errors assume that the covariance matrix of the errors is correctly specified.



The R-Squared score shows a big improvement - our first model predicted only around 45% of the variation, but now we are predicting 87%!


```python
# Use our new model to make predictions
pred3=r_fit3.predict(X_test)

```


```python
# Plot the predictions
# Build a scatterplot
plt.scatter(pred3,Y_test)
# Add a line for perfect correlation
plt.plot([x for x in range (9, 15)], [x for x in range (9, 15)], color='red')

# Label it nicely
plt.xlabel("nice labels")
plt.ylabel("predictions")
```




    Text(0, 0.5, 'predictions')




    
![png](output_62_1.png)
    


We've now got a much closer match between our data and our predictions, and we can see that the shape of the data points is much more similar to the red line. 

We can check another metric as well - the RMSE (Root Mean Squared Error). The MSE is defined by Professor Spiegelhalter on p.393 of *AoS*, and the RMSE is just the square root of that value. This is a measure of the accuracy of a regression model. Very simply put, it's formed by finding the average difference between predictions and actual values. Check out p. 163 of *AoS* for a reminder of how this works. 


```python
# Define a function to check the RMSE. Remember the def keyword needed to make functions? 
def rmse(predicted,test):
 
    MSE = np.square(np.subtract(test,predicted)).mean() 
 
    RMSE = math.sqrt(MSE)
#or jusst return
    return np.sqrt(((predicted-test)**2).mean())
```


```python
# Get predictions from rModel3
pred3=r_fit3.predict(X_test)

# Put the predictions & actual values into a dataframe
matches = pd.DataFrame(X_test)
matches.rename(columns = {'fixed.acidity':'actual'}, inplace=True)
matches["predicted"] = pred3
#mathes.head()
rmse(matches['actual'], matches['predicted'])
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-167-82b8ebe6f165> in <module>
          1 # Get predictions from rModel3
    ----> 2 pred3=r_fit3.predict(X_test)
          3 
          4 # Put the predictions & actual values into a dataframe
          5 matches = pd.DataFrame(X_test)
    

    C:\Users\toshiba\anaconda3\lib\site-packages\statsmodels\base\model.py in predict(self, exog, transform, *args, **kwargs)
       1097             exog = np.atleast_2d(exog)  # needed in count model shape[1]
       1098 
    -> 1099         predict_results = self.model.predict(self.params, exog, *args,
       1100                                              **kwargs)
       1101 
    

    C:\Users\toshiba\anaconda3\lib\site-packages\statsmodels\regression\linear_model.py in predict(self, params, exog)
        378             exog = self.exog
        379 
    --> 380         return np.dot(exog, params)
        381 
        382     def get_distribution(self, params, scale, exog=None, dist_class=None):
    

    <__array_function__ internals> in dot(*args, **kwargs)
    

    ValueError: shapes (400,2) and (1,) not aligned: 2 (dim 1) != 1 (dim 0)


The RMSE tells us how far, on average, our predictions were mistaken. An RMSE of 0 would mean we were making perfect predictions. 0.6 signifies that we are, on average, about 0.6 of a unit of fixed acidity away from the correct answer. That's not bad at all.

#### 3e. Making a Linear Regression model: our fourth model: avoiding redundancy 

We can also see from our early heat map that volatile.acidity and citric.acid are both correlated with pH. We can make a model that ignores those two variables and just uses pH, in an attempt to remove redundancy from our model.


```python
wine.info()
```


```python
# Create test and train datasets
# Include the remaining six columns as predictors
X = wine[['residual.sugar', 'chlorides', 'total.sulfur.dioxide', 'density', 'pH', 'sulphates']]
# Create constants for X, so the model knows its bounds
X=sm.add_constant(X)

# Split the data
X_train,X_test,Y_train,Y_test=train_test_split(dependent,independent,test_size=.25)
```


```python
# Create the fifth model
rmodel5=sm.OLS(Y_train,X_train)
# Fit the model
model5_fit=rmodel5.fit()
# Evaluate the model
model5_fit.summary()

```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>density</td>     <th>  R-squared (uncentered):</th>      <td>   0.958</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared (uncentered):</th> <td>   0.958</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>          <td>2.729e+04</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 11 May 2021</td> <th>  Prob (F-statistic):</th>           <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>12:46:30</td>     <th>  Log-Likelihood:    </th>          <td>  202.28</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1199</td>      <th>  AIC:               </th>          <td>  -402.6</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1198</td>      <th>  BIC:               </th>          <td>  -397.5</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>              <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>              <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>fixed.acidity</th> <td>    0.1146</td> <td>    0.001</td> <td>  165.195</td> <td> 0.000</td> <td>    0.113</td> <td>    0.116</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>183.469</td> <th>  Durbin-Watson:     </th> <td>   2.012</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 294.153</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-1.013</td>  <th>  Prob(JB):          </th> <td>1.33e-64</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.336</td>  <th>  Cond. No.          </th> <td>    1.00</td>
</tr>
</table><br/><br/>Notes:<br/>[1] R² is computed without centering (uncentered) since the model does not contain a constant.<br/>[2] Standard Errors assume that the covariance matrix of the errors is correctly specified.



The R-squared score has reduced, showing us that actually, the removed columns were important.

### Conclusions & next steps

Congratulations on getting through this implementation of regression and good data science practice in Python! 

Take a moment to reflect on which model was the best, before reading on.

.
.
.

Here's one conclusion that seems right. While our most predictively powerful model was rModel3, this model had explanatory variables that were correlated with one another, which made some redundancy. Our most elegant and economical model was rModel4 - it used just a few predictors to get a good result. 

All of our models in this notebook have used the OLS algorithm - Ordinary Least Squares. There are many other regression algorithms, and if you have time, it would be good to investigate them. You can find some examples [here](https://www.statsmodels.org/dev/examples/index.html#regression). Be sure to make a note of what you find, and chat through it with your mentor at your next call.

