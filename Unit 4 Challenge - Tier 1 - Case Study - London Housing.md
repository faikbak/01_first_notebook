# Springboard Data Science Career Track Unit 4 Challenge - Tier 1 Complete


## Objectives
Hey! Great job getting through those challenging DataCamp courses. You're learning a lot in a short span of time - so let's apply your knowledge in the real world. 

In this notebook, we're going to apply the skills you've been learning, bridging the gap between the controlled environment of DataCamp and the *slightly* messier work that data scientists do with actual datasets!

Here’s the mystery we’re going to solve: ***which boroughs of London have seen the greatest increase in housing prices, on average, over the last two decades?***


A borough is just a fancy word for district. You may be familiar with the five boroughs of New York… [well, there are 32 boroughs within Greater London](https://en.wikipedia.org/wiki/London_boroughs). Some of them are more desirable areas to live in, and the data will reflect that with a greater rise in housing prices.

This is the Tier One notebook, which is the easiest tier. More of the code has been filled in already for you, so you have to less research on how to complete the lines. We ask that you only complete this tier if you've given Tiers Two and Three your best effort and been thwarted. We also ask, once you complete this tier, to go back to the highest difficulty level and give it another go. We really want you to internalize the skills you're learning.


This challenge will make use of only what you learned in the following DataCamp courses: 
- Prework courses (Introduction to Python for Data Science, Intermediate Python for Data Science)
- Data Types for Data Science
- Python Data Science Toolbox (Part One) 
- pandas Foundations
- Manipulating DataFrames with pandas
- Merging DataFrames with pandas

Of the tools, techniques and concepts in the above DataCamp courses, this challenge should require the application of the following: 
- **pandas**
    - **data ingestion and inspection** (pandas Foundations, Module One) 
    - **exploratory data analysis** (pandas Foundations, Module Two)
    - **tidying and cleaning** (Manipulating DataFrames with pandas, Module Three) 
    - **transforming DataFrames** (Manipulating DataFrames with pandas, Module One)
    - **subsetting DataFrames with lists** (Manipulating DataFrames with pandas, Module One) 
    - **filtering DataFrames** (Manipulating DataFrames with pandas, Module One) 
    - **grouping data** (Manipulating DataFrames with pandas, Module Four) 
    - **melting data** (Manipulating DataFrames with pandas, Module Three) 
    - **advanced indexing** (Manipulating DataFrames with pandas, Module Four) 
- **matplotlib** (Intermediate Python for Data Science, Module One)
- **fundamental data types** (Data Types for Data Science, Module One) 
- **dictionaries** (Intermediate Python for Data Science, Module Two)
- **handling dates and times** (Data Types for Data Science, Module Four)
- **function definition** (Python Data Science Toolbox - Part One, Module One)
- **default arguments, variable length, and scope** (Python Data Science Toolbox - Part One, Module Two) 
- **lambda functions and error handling** (Python Data Science Toolbox - Part One, Module Four) 

## The Data Science Pipeline
Data Science is magical. In this case study, you'll get to apply some complex machine learning algorithms. But as  [David Spiegelhalter](https://www.youtube.com/watch?v=oUs1uvsz0Ok) reminds us, there is no substitute for simply **taking a really, really good look at the data.** Sometimes, this is all we need to answer our question.

Data Science projects generally adhere to the four stages of Data Science Pipeline:
1. Sourcing and loading 
2. Cleaning, transforming, and visualizing 
3. Modeling 
4. Evaluating and concluding 

### 1. Sourcing and Loading 

Any Data Science project kicks off by importing  ***pandas***. The documentation of this wonderful library can be found [here](https://pandas.pydata.org/). As you've seen, pandas is conveniently connected to the [Numpy](http://www.numpy.org/) and [Matplotlib](https://matplotlib.org/) libraries. 

***Hint:*** This part of the data science pipeline will test those skills you acquired in the pandas Foundations course, Module One. 

#### 1.1. Importing Libraries


```python
# Let's import the pandas, numpy libraries as pd, and np respectively. 
import pandas as pd
import numpy as np
from datetime import date
# Load the pyplot collection of functions from matplotlib, as plt 
import matplotlib.pyplot as plt
```

#### 1.2.  Loading the data


Your the data comes from the [London Datastore](https://data.london.gov.uk/): a free, open-source data-sharing portal with a massive range of London-oriented datasets.


```python
# First, make a variable called url_LondonHousePrices, and assign it the following link, enclosed in quotation-marks as a string:
# https://data.london.gov.uk/download/uk-house-price-index/70ac0766-8902-4eb5-aab5-01951aaed773/UK%20House%20price%20index.xls

url_LondonHousePrices= "https://data.london.gov.uk/download/uk-house-price-index/70ac0766-8902-4eb5-aab5-01951aaed773/UK%20House%20price%20index.xls"

# The dataset we're interested in contains the Average prices of the houses, and is actually on a particular sheet of the Excel file. 
# As a result, we need to specify the sheet name in the read_excel() method.
# Put this data into a variable called properties.  
properties = pd.read_excel(url_LondonHousePrices, sheet_name='Average price', index_col= None)
```

### 2. Cleaning, transforming, and visualizing 
This second stage is arguably the most important part of any Data Science project. The first thing to do is take a proper look at the data. Cleaning forms the majority of this stage, and can be done both before or after Transformation.

The end goal of data cleaning is to have tidy data. When data is tidy: 

1. Each variable has a column.
2. Each observation forms a row.

Keep the end goal in mind as you move through this process, every step will take you closer. 



***Hint:*** This part of the data science pipeline should test those skills you acquired in: 
- Intermediate Python for data science, all modules.
- pandas Foundations, all modules. 
- Manipulating DataFrames with pandas, all modules.
- Data Types for Data Science, Module Four.
- Python Data Science Toolbox - Part One, all modules

#### 2.1. Exploring the data


```python
# First off, let's use .shape feature of pandas DataFrames to look at the number of rows and columns. 
properties.shape
```




    (313, 49)




```python
# Using the .head() method, let's check out the state of our dataset.  
properties.head()
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
      <th>City of London</th>
      <th>Barking &amp; Dagenham</th>
      <th>Barnet</th>
      <th>Bexley</th>
      <th>Brent</th>
      <th>Bromley</th>
      <th>Camden</th>
      <th>Croydon</th>
      <th>Ealing</th>
      <th>...</th>
      <th>NORTH WEST</th>
      <th>YORKS &amp; THE HUMBER</th>
      <th>EAST MIDLANDS</th>
      <th>WEST MIDLANDS</th>
      <th>EAST OF ENGLAND</th>
      <th>LONDON</th>
      <th>SOUTH EAST</th>
      <th>SOUTH WEST</th>
      <th>Unnamed: 47</th>
      <th>England</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaT</td>
      <td>E09000001</td>
      <td>E09000002</td>
      <td>E09000003</td>
      <td>E09000004</td>
      <td>E09000005</td>
      <td>E09000006</td>
      <td>E09000007</td>
      <td>E09000008</td>
      <td>E09000009</td>
      <td>...</td>
      <td>E12000002</td>
      <td>E12000003</td>
      <td>E12000004</td>
      <td>E12000005</td>
      <td>E12000006</td>
      <td>E12000007</td>
      <td>E12000008</td>
      <td>E12000009</td>
      <td>NaN</td>
      <td>E92000001</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1995-01-01</td>
      <td>91449</td>
      <td>50460.2</td>
      <td>93284.5</td>
      <td>64958.1</td>
      <td>71306.6</td>
      <td>81671.5</td>
      <td>120933</td>
      <td>69158.2</td>
      <td>79885.9</td>
      <td>...</td>
      <td>43958.5</td>
      <td>44803.4</td>
      <td>45544.5</td>
      <td>48527.5</td>
      <td>56701.6</td>
      <td>74435.8</td>
      <td>64018.9</td>
      <td>54705.2</td>
      <td>NaN</td>
      <td>53202.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1995-02-01</td>
      <td>82202.8</td>
      <td>51085.8</td>
      <td>93190.2</td>
      <td>64787.9</td>
      <td>72022.3</td>
      <td>81657.6</td>
      <td>119509</td>
      <td>68951.1</td>
      <td>80897.1</td>
      <td>...</td>
      <td>43925.4</td>
      <td>44528.8</td>
      <td>46051.6</td>
      <td>49341.3</td>
      <td>56593.6</td>
      <td>72777.9</td>
      <td>63715</td>
      <td>54356.1</td>
      <td>NaN</td>
      <td>53096.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1995-03-01</td>
      <td>79120.7</td>
      <td>51269</td>
      <td>92247.5</td>
      <td>64367.5</td>
      <td>72015.8</td>
      <td>81449.3</td>
      <td>120282</td>
      <td>68712.4</td>
      <td>81379.9</td>
      <td>...</td>
      <td>44434.9</td>
      <td>45200.5</td>
      <td>45383.8</td>
      <td>49442.2</td>
      <td>56171.2</td>
      <td>73896.8</td>
      <td>64113.6</td>
      <td>53583.1</td>
      <td>NaN</td>
      <td>53201.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1995-04-01</td>
      <td>77101.2</td>
      <td>53133.5</td>
      <td>90762.9</td>
      <td>64277.7</td>
      <td>72965.6</td>
      <td>81124.4</td>
      <td>120098</td>
      <td>68610</td>
      <td>82188.9</td>
      <td>...</td>
      <td>44267.8</td>
      <td>45614.3</td>
      <td>46124.2</td>
      <td>49455.9</td>
      <td>56567.9</td>
      <td>74455.3</td>
      <td>64623.2</td>
      <td>54786</td>
      <td>NaN</td>
      <td>53590.9</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 49 columns</p>
</div>



Oh no! What are you supposed to do with this?

You've got the data, but it doesn't look tidy. At this stage, you'd struggle to perform analysis on it. It is normal for your initial data set to be formatted in a way that is not conducive to analysis. A big part of your job is fixing that. 

Best practice is for pandas DataFrames to contain the **observations of interest** as rows, and the features of those observations as columns. You want **tidy** DataFrames: whose rows are observations and whose columns are variables.


Notice here that the column headings are the *particular* boroughs, which is your observation of interest. The first column contains datetime objects that capture a particular month and year, which is a variable. Most of the other cell-values are the average proprety values of the borough corresponding to that time stamp. 

Clearly, you need to roll up your sleeves and do some  **cleaning**. 

####  2.2. Cleaning the data (Part 1)
Data cleaning has a bad rep, but remember what your momma told you: cleanliness is next to godliness. Data cleaning can be really satisfying and fun. In the dark ages of programming, data cleaning was a tedious and difficult ordeal. Nowadays, new and improved tools have simplified the process. Getting good at data cleaning opens up a world of possibilities for data scientists and programmers. 
 
The first operation you want to do on the dataset is called **transposition**. You *transpose* a table when you flip the columns into rows, and *vice versa*. 

If you transpose this DataFrame then the borough names will become the row indices, and the date time objects will become the column headers. Since your end goal is tidy data, where each row will represent a borough and each column will contain data about that borough at a certain point in time, transposing the table bring us closer to where you want to be.

Python makes transposition simple.

Each pandas DataFrame already has the *.T* attribute which is the transposed version of that DataFrame.

Assign the transposed version of the original to a new variable. Let’s call it *properties_T*. 

Boom! You’ve got a transposed table to play with.


```python
# Do this here
properties_T = properties.T
```


```python
# Let's check the head of our new Transposed DataFrame. 
properties_T.head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>303</th>
      <th>304</th>
      <th>305</th>
      <th>306</th>
      <th>307</th>
      <th>308</th>
      <th>309</th>
      <th>310</th>
      <th>311</th>
      <th>312</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Unnamed: 0</th>
      <td>NaT</td>
      <td>1995-01-01 00:00:00</td>
      <td>1995-02-01 00:00:00</td>
      <td>1995-03-01 00:00:00</td>
      <td>1995-04-01 00:00:00</td>
      <td>1995-05-01 00:00:00</td>
      <td>1995-06-01 00:00:00</td>
      <td>1995-07-01 00:00:00</td>
      <td>1995-08-01 00:00:00</td>
      <td>1995-09-01 00:00:00</td>
      <td>...</td>
      <td>2020-03-01 00:00:00</td>
      <td>2020-04-01 00:00:00</td>
      <td>2020-05-01 00:00:00</td>
      <td>2020-06-01 00:00:00</td>
      <td>2020-07-01 00:00:00</td>
      <td>2020-08-01 00:00:00</td>
      <td>2020-09-01 00:00:00</td>
      <td>2020-10-01 00:00:00</td>
      <td>2020-11-01 00:00:00</td>
      <td>2020-12-01 00:00:00</td>
    </tr>
    <tr>
      <th>City of London</th>
      <td>E09000001</td>
      <td>91449</td>
      <td>82202.8</td>
      <td>79120.7</td>
      <td>77101.2</td>
      <td>84409.1</td>
      <td>94900.5</td>
      <td>110128</td>
      <td>112329</td>
      <td>104473</td>
      <td>...</td>
      <td>836241</td>
      <td>920444</td>
      <td>918209</td>
      <td>882872</td>
      <td>786627</td>
      <td>798227</td>
      <td>774654</td>
      <td>813356</td>
      <td>788715</td>
      <td>784006</td>
    </tr>
    <tr>
      <th>Barking &amp; Dagenham</th>
      <td>E09000002</td>
      <td>50460.2</td>
      <td>51085.8</td>
      <td>51269</td>
      <td>53133.5</td>
      <td>53042.2</td>
      <td>53700.3</td>
      <td>52113.1</td>
      <td>52232.2</td>
      <td>51471.6</td>
      <td>...</td>
      <td>301413</td>
      <td>293603</td>
      <td>293816</td>
      <td>300526</td>
      <td>304556</td>
      <td>304336</td>
      <td>302165</td>
      <td>305197</td>
      <td>306987</td>
      <td>317947</td>
    </tr>
    <tr>
      <th>Barnet</th>
      <td>E09000003</td>
      <td>93284.5</td>
      <td>93190.2</td>
      <td>92247.5</td>
      <td>90762.9</td>
      <td>90258</td>
      <td>90107.2</td>
      <td>91441.2</td>
      <td>92361.3</td>
      <td>93273.1</td>
      <td>...</td>
      <td>522115</td>
      <td>526689</td>
      <td>526033</td>
      <td>518175</td>
      <td>523280</td>
      <td>527952</td>
      <td>536177</td>
      <td>534081</td>
      <td>535055</td>
      <td>534719</td>
    </tr>
    <tr>
      <th>Bexley</th>
      <td>E09000004</td>
      <td>64958.1</td>
      <td>64787.9</td>
      <td>64367.5</td>
      <td>64277.7</td>
      <td>63997.1</td>
      <td>64252.3</td>
      <td>63722.7</td>
      <td>64432.6</td>
      <td>64509.5</td>
      <td>...</td>
      <td>339182</td>
      <td>341553</td>
      <td>339353</td>
      <td>340893</td>
      <td>344091</td>
      <td>346997</td>
      <td>345829</td>
      <td>346656</td>
      <td>350585</td>
      <td>354466</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 313 columns</p>
</div>



You've made some progress! But with new progress comes new issues. For one, the row indices of our DataFrame contain the names of the boroughs. You should never have a piece of information we want to analyze as an index, this information should be within the DataFrame itself. The indices should just be a unique ID, almost always a number.

Those names are perhaps the most important piece of information! Put them where you can work with them.


```python
# To confirm what our row indices are, let's call the .index variable on our properties_T DataFrame. 
properties_T.index
```




    Index(['Unnamed: 0', 'City of London', 'Barking & Dagenham', 'Barnet',
           'Bexley', 'Brent', 'Bromley', 'Camden', 'Croydon', 'Ealing', 'Enfield',
           'Greenwich', 'Hackney', 'Hammersmith & Fulham', 'Haringey', 'Harrow',
           'Havering', 'Hillingdon', 'Hounslow', 'Islington',
           'Kensington & Chelsea', 'Kingston upon Thames', 'Lambeth', 'Lewisham',
           'Merton', 'Newham', 'Redbridge', 'Richmond upon Thames', 'Southwark',
           'Sutton', 'Tower Hamlets', 'Waltham Forest', 'Wandsworth',
           'Westminster', 'Unnamed: 34', 'Inner London', 'Outer London',
           'Unnamed: 37', 'NORTH EAST', 'NORTH WEST', 'YORKS & THE HUMBER',
           'EAST MIDLANDS', 'WEST MIDLANDS', 'EAST OF ENGLAND', 'LONDON',
           'SOUTH EAST', 'SOUTH WEST', 'Unnamed: 47', 'England'],
          dtype='object')




```python
# Our suspicion was correct. 
# Call the .reset_index() method on properties_T to reset the indices, and the reassign the result to properties_T: 
properties_T = properties_T.reset_index()
```


```python
# Now let's check out our DataFrames indices: 
properties_T.index
```




    RangeIndex(start=0, stop=49, step=1)



Progress! 

The indicies are now a numerical RangeIndex, exactly what you want. 

**Note**: if you call the reset_index() line more than once, you'll get an error because a whole extra level of row indices will have been inserted! If you do this, don't worry. Just hit Kernel > Restart, then run all the cells up to here to get back to where you were. 


```python
# Call the head() function again on properties_T to check out the new row indices: 
properties_T.head()
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
      <th>index</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>303</th>
      <th>304</th>
      <th>305</th>
      <th>306</th>
      <th>307</th>
      <th>308</th>
      <th>309</th>
      <th>310</th>
      <th>311</th>
      <th>312</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Unnamed: 0</td>
      <td>NaT</td>
      <td>1995-01-01 00:00:00</td>
      <td>1995-02-01 00:00:00</td>
      <td>1995-03-01 00:00:00</td>
      <td>1995-04-01 00:00:00</td>
      <td>1995-05-01 00:00:00</td>
      <td>1995-06-01 00:00:00</td>
      <td>1995-07-01 00:00:00</td>
      <td>1995-08-01 00:00:00</td>
      <td>...</td>
      <td>2020-03-01 00:00:00</td>
      <td>2020-04-01 00:00:00</td>
      <td>2020-05-01 00:00:00</td>
      <td>2020-06-01 00:00:00</td>
      <td>2020-07-01 00:00:00</td>
      <td>2020-08-01 00:00:00</td>
      <td>2020-09-01 00:00:00</td>
      <td>2020-10-01 00:00:00</td>
      <td>2020-11-01 00:00:00</td>
      <td>2020-12-01 00:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>City of London</td>
      <td>E09000001</td>
      <td>91449</td>
      <td>82202.8</td>
      <td>79120.7</td>
      <td>77101.2</td>
      <td>84409.1</td>
      <td>94900.5</td>
      <td>110128</td>
      <td>112329</td>
      <td>...</td>
      <td>836241</td>
      <td>920444</td>
      <td>918209</td>
      <td>882872</td>
      <td>786627</td>
      <td>798227</td>
      <td>774654</td>
      <td>813356</td>
      <td>788715</td>
      <td>784006</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>50460.2</td>
      <td>51085.8</td>
      <td>51269</td>
      <td>53133.5</td>
      <td>53042.2</td>
      <td>53700.3</td>
      <td>52113.1</td>
      <td>52232.2</td>
      <td>...</td>
      <td>301413</td>
      <td>293603</td>
      <td>293816</td>
      <td>300526</td>
      <td>304556</td>
      <td>304336</td>
      <td>302165</td>
      <td>305197</td>
      <td>306987</td>
      <td>317947</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Barnet</td>
      <td>E09000003</td>
      <td>93284.5</td>
      <td>93190.2</td>
      <td>92247.5</td>
      <td>90762.9</td>
      <td>90258</td>
      <td>90107.2</td>
      <td>91441.2</td>
      <td>92361.3</td>
      <td>...</td>
      <td>522115</td>
      <td>526689</td>
      <td>526033</td>
      <td>518175</td>
      <td>523280</td>
      <td>527952</td>
      <td>536177</td>
      <td>534081</td>
      <td>535055</td>
      <td>534719</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bexley</td>
      <td>E09000004</td>
      <td>64958.1</td>
      <td>64787.9</td>
      <td>64367.5</td>
      <td>64277.7</td>
      <td>63997.1</td>
      <td>64252.3</td>
      <td>63722.7</td>
      <td>64432.6</td>
      <td>...</td>
      <td>339182</td>
      <td>341553</td>
      <td>339353</td>
      <td>340893</td>
      <td>344091</td>
      <td>346997</td>
      <td>345829</td>
      <td>346656</td>
      <td>350585</td>
      <td>354466</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 314 columns</p>
</div>



You're getting somewhere, but your column headings are mainly just integers. The first one is the string 'index' and the rest are integers ranging from 0 to 296, inclusive.

For the ultimate aim of having a *tidy* DataFrame, you'll turn the datetimes found along the first row (at index 0) into the column headings.  The resulting DataFrame will have boroughs as rows, the columns as dates (each representing a particular month), and the cell-values as the average property value sold in that borough for that month. 


```python
# To confirm that our DataFrame's columns are mainly just integers, call the .columns feature on our DataFrame:
properties_T.dtypes 
```




    index    object
    0        object
    1        object
    2        object
    3        object
              ...  
    308      object
    309      object
    310      object
    311      object
    312      object
    Length: 314, dtype: object



To confirm that the first row contains the proper values for column headings, use the  ***iloc[] method*** on our *properties_T* DataFrame. Use index 0. You'll recall from DataCamp that if you use single square brackets, you'll return a series. If you use double square brackets, a DataFrame is returned.


```python
# Call the iloc[] method with double square brackets on the properties_T DataFrame, to see the row at index 0. 
properties_T.iloc[[0]]
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
      <th>index</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>303</th>
      <th>304</th>
      <th>305</th>
      <th>306</th>
      <th>307</th>
      <th>308</th>
      <th>309</th>
      <th>310</th>
      <th>311</th>
      <th>312</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Unnamed: 0</td>
      <td>NaT</td>
      <td>1995-01-01 00:00:00</td>
      <td>1995-02-01 00:00:00</td>
      <td>1995-03-01 00:00:00</td>
      <td>1995-04-01 00:00:00</td>
      <td>1995-05-01 00:00:00</td>
      <td>1995-06-01 00:00:00</td>
      <td>1995-07-01 00:00:00</td>
      <td>1995-08-01 00:00:00</td>
      <td>...</td>
      <td>2020-03-01 00:00:00</td>
      <td>2020-04-01 00:00:00</td>
      <td>2020-05-01 00:00:00</td>
      <td>2020-06-01 00:00:00</td>
      <td>2020-07-01 00:00:00</td>
      <td>2020-08-01 00:00:00</td>
      <td>2020-09-01 00:00:00</td>
      <td>2020-10-01 00:00:00</td>
      <td>2020-11-01 00:00:00</td>
      <td>2020-12-01 00:00:00</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 314 columns</p>
</div>



**Notice that these values are all the months from January 1995 to August 2019, inclusive**. You can reassign the columns of your DataFrame the values within this row at index 0 by making use of the *.columns* feature.


```python
# Try this now. 
properties_T.columns = properties_T.iloc[0]
```


```python
# Check out our DataFrame again: 
properties_T.head()
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
      <th>NaN</th>
      <th>1995-01-01 00:00:00</th>
      <th>1995-02-01 00:00:00</th>
      <th>1995-03-01 00:00:00</th>
      <th>1995-04-01 00:00:00</th>
      <th>1995-05-01 00:00:00</th>
      <th>1995-06-01 00:00:00</th>
      <th>1995-07-01 00:00:00</th>
      <th>1995-08-01 00:00:00</th>
      <th>...</th>
      <th>2020-03-01 00:00:00</th>
      <th>2020-04-01 00:00:00</th>
      <th>2020-05-01 00:00:00</th>
      <th>2020-06-01 00:00:00</th>
      <th>2020-07-01 00:00:00</th>
      <th>2020-08-01 00:00:00</th>
      <th>2020-09-01 00:00:00</th>
      <th>2020-10-01 00:00:00</th>
      <th>2020-11-01 00:00:00</th>
      <th>2020-12-01 00:00:00</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Unnamed: 0</td>
      <td>NaT</td>
      <td>1995-01-01 00:00:00</td>
      <td>1995-02-01 00:00:00</td>
      <td>1995-03-01 00:00:00</td>
      <td>1995-04-01 00:00:00</td>
      <td>1995-05-01 00:00:00</td>
      <td>1995-06-01 00:00:00</td>
      <td>1995-07-01 00:00:00</td>
      <td>1995-08-01 00:00:00</td>
      <td>...</td>
      <td>2020-03-01 00:00:00</td>
      <td>2020-04-01 00:00:00</td>
      <td>2020-05-01 00:00:00</td>
      <td>2020-06-01 00:00:00</td>
      <td>2020-07-01 00:00:00</td>
      <td>2020-08-01 00:00:00</td>
      <td>2020-09-01 00:00:00</td>
      <td>2020-10-01 00:00:00</td>
      <td>2020-11-01 00:00:00</td>
      <td>2020-12-01 00:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>City of London</td>
      <td>E09000001</td>
      <td>91449</td>
      <td>82202.8</td>
      <td>79120.7</td>
      <td>77101.2</td>
      <td>84409.1</td>
      <td>94900.5</td>
      <td>110128</td>
      <td>112329</td>
      <td>...</td>
      <td>836241</td>
      <td>920444</td>
      <td>918209</td>
      <td>882872</td>
      <td>786627</td>
      <td>798227</td>
      <td>774654</td>
      <td>813356</td>
      <td>788715</td>
      <td>784006</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>50460.2</td>
      <td>51085.8</td>
      <td>51269</td>
      <td>53133.5</td>
      <td>53042.2</td>
      <td>53700.3</td>
      <td>52113.1</td>
      <td>52232.2</td>
      <td>...</td>
      <td>301413</td>
      <td>293603</td>
      <td>293816</td>
      <td>300526</td>
      <td>304556</td>
      <td>304336</td>
      <td>302165</td>
      <td>305197</td>
      <td>306987</td>
      <td>317947</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Barnet</td>
      <td>E09000003</td>
      <td>93284.5</td>
      <td>93190.2</td>
      <td>92247.5</td>
      <td>90762.9</td>
      <td>90258</td>
      <td>90107.2</td>
      <td>91441.2</td>
      <td>92361.3</td>
      <td>...</td>
      <td>522115</td>
      <td>526689</td>
      <td>526033</td>
      <td>518175</td>
      <td>523280</td>
      <td>527952</td>
      <td>536177</td>
      <td>534081</td>
      <td>535055</td>
      <td>534719</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bexley</td>
      <td>E09000004</td>
      <td>64958.1</td>
      <td>64787.9</td>
      <td>64367.5</td>
      <td>64277.7</td>
      <td>63997.1</td>
      <td>64252.3</td>
      <td>63722.7</td>
      <td>64432.6</td>
      <td>...</td>
      <td>339182</td>
      <td>341553</td>
      <td>339353</td>
      <td>340893</td>
      <td>344091</td>
      <td>346997</td>
      <td>345829</td>
      <td>346656</td>
      <td>350585</td>
      <td>354466</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 314 columns</p>
</div>



You need to drop the row at index 0! 

A good way to do this is reassign *properties_T* with the return value of calling the DataFrame ***drop()*** method, passing 0 as the index.


```python
# Have a go at this now. 
properties_T = properties_T.drop(0)
```


```python
# Now check out our DataFrame again to see how it looks. 
properties_T.head()
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
      <th>NaN</th>
      <th>1995-01-01 00:00:00</th>
      <th>1995-02-01 00:00:00</th>
      <th>1995-03-01 00:00:00</th>
      <th>1995-04-01 00:00:00</th>
      <th>1995-05-01 00:00:00</th>
      <th>1995-06-01 00:00:00</th>
      <th>1995-07-01 00:00:00</th>
      <th>1995-08-01 00:00:00</th>
      <th>...</th>
      <th>2020-03-01 00:00:00</th>
      <th>2020-04-01 00:00:00</th>
      <th>2020-05-01 00:00:00</th>
      <th>2020-06-01 00:00:00</th>
      <th>2020-07-01 00:00:00</th>
      <th>2020-08-01 00:00:00</th>
      <th>2020-09-01 00:00:00</th>
      <th>2020-10-01 00:00:00</th>
      <th>2020-11-01 00:00:00</th>
      <th>2020-12-01 00:00:00</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>City of London</td>
      <td>E09000001</td>
      <td>91449</td>
      <td>82202.8</td>
      <td>79120.7</td>
      <td>77101.2</td>
      <td>84409.1</td>
      <td>94900.5</td>
      <td>110128</td>
      <td>112329</td>
      <td>...</td>
      <td>836241</td>
      <td>920444</td>
      <td>918209</td>
      <td>882872</td>
      <td>786627</td>
      <td>798227</td>
      <td>774654</td>
      <td>813356</td>
      <td>788715</td>
      <td>784006</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>50460.2</td>
      <td>51085.8</td>
      <td>51269</td>
      <td>53133.5</td>
      <td>53042.2</td>
      <td>53700.3</td>
      <td>52113.1</td>
      <td>52232.2</td>
      <td>...</td>
      <td>301413</td>
      <td>293603</td>
      <td>293816</td>
      <td>300526</td>
      <td>304556</td>
      <td>304336</td>
      <td>302165</td>
      <td>305197</td>
      <td>306987</td>
      <td>317947</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Barnet</td>
      <td>E09000003</td>
      <td>93284.5</td>
      <td>93190.2</td>
      <td>92247.5</td>
      <td>90762.9</td>
      <td>90258</td>
      <td>90107.2</td>
      <td>91441.2</td>
      <td>92361.3</td>
      <td>...</td>
      <td>522115</td>
      <td>526689</td>
      <td>526033</td>
      <td>518175</td>
      <td>523280</td>
      <td>527952</td>
      <td>536177</td>
      <td>534081</td>
      <td>535055</td>
      <td>534719</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bexley</td>
      <td>E09000004</td>
      <td>64958.1</td>
      <td>64787.9</td>
      <td>64367.5</td>
      <td>64277.7</td>
      <td>63997.1</td>
      <td>64252.3</td>
      <td>63722.7</td>
      <td>64432.6</td>
      <td>...</td>
      <td>339182</td>
      <td>341553</td>
      <td>339353</td>
      <td>340893</td>
      <td>344091</td>
      <td>346997</td>
      <td>345829</td>
      <td>346656</td>
      <td>350585</td>
      <td>354466</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Brent</td>
      <td>E09000005</td>
      <td>71306.6</td>
      <td>72022.3</td>
      <td>72015.8</td>
      <td>72965.6</td>
      <td>73704</td>
      <td>74310.5</td>
      <td>74127</td>
      <td>73547</td>
      <td>...</td>
      <td>466250</td>
      <td>470601</td>
      <td>482808</td>
      <td>484160</td>
      <td>482303</td>
      <td>493930</td>
      <td>519117</td>
      <td>523644</td>
      <td>517569</td>
      <td>496316</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 314 columns</p>
</div>



You're slowly but surely getting there! Exciting, right? 

**Each column now represents a month and year, and each cell-value represents the average price of houses sold in borough of the corresponding row**. 

You have total control over your data! 

#### 2.3. Cleaning the data (Part 2)
You can see from the *.head()* list call that you need to rename some columns. 

'Unnamed: 0' should be something like 'London Borough' and 'NaN' should  be changed. 

Recall, that pandas DataFrames have a ***.rename()*** method. One of the keyworded arguments to this method is *columns*. You can assign it a dictionary whose keys are the current column names you want to change, and whose values are the desired new names.

**Note**: you can change the 'Unnamed: 0' name of the first column just by including that string as a key in our dictionary, but 'NaN' stands for Not A Number,  and is denoted by *pd.NaT*. Do not use quotes when you include this value. NaN means Not A Number, and NaT means Not A Time - both of these values represent undefined or unrepresenable values like 0/0. They are functionally Null values. Don't worry, we'll help you with this.

Call the **rename()** method on *properties_T* and set the *columns* keyword equal to the following dictionary: 
{'Unnamed: 0':'London_Borough', pd.NaT: 'ID'} 
, then reassign that value to properties_T to update the DataFrame.


```python
# Try this here. 
properties_T = properties_T.rename(columns = {'Unnamed: 0':'London_Borough', pd.NaT: 'ID'})
```


```python
# Let's check out the DataFrame again to admire our good work. 
properties_T.head()
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
      <th>London_Borough</th>
      <th>ID</th>
      <th>1995-01-01 00:00:00</th>
      <th>1995-02-01 00:00:00</th>
      <th>1995-03-01 00:00:00</th>
      <th>1995-04-01 00:00:00</th>
      <th>1995-05-01 00:00:00</th>
      <th>1995-06-01 00:00:00</th>
      <th>1995-07-01 00:00:00</th>
      <th>1995-08-01 00:00:00</th>
      <th>...</th>
      <th>2020-03-01 00:00:00</th>
      <th>2020-04-01 00:00:00</th>
      <th>2020-05-01 00:00:00</th>
      <th>2020-06-01 00:00:00</th>
      <th>2020-07-01 00:00:00</th>
      <th>2020-08-01 00:00:00</th>
      <th>2020-09-01 00:00:00</th>
      <th>2020-10-01 00:00:00</th>
      <th>2020-11-01 00:00:00</th>
      <th>2020-12-01 00:00:00</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>City of London</td>
      <td>E09000001</td>
      <td>91449</td>
      <td>82202.8</td>
      <td>79120.7</td>
      <td>77101.2</td>
      <td>84409.1</td>
      <td>94900.5</td>
      <td>110128</td>
      <td>112329</td>
      <td>...</td>
      <td>836241</td>
      <td>920444</td>
      <td>918209</td>
      <td>882872</td>
      <td>786627</td>
      <td>798227</td>
      <td>774654</td>
      <td>813356</td>
      <td>788715</td>
      <td>784006</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>50460.2</td>
      <td>51085.8</td>
      <td>51269</td>
      <td>53133.5</td>
      <td>53042.2</td>
      <td>53700.3</td>
      <td>52113.1</td>
      <td>52232.2</td>
      <td>...</td>
      <td>301413</td>
      <td>293603</td>
      <td>293816</td>
      <td>300526</td>
      <td>304556</td>
      <td>304336</td>
      <td>302165</td>
      <td>305197</td>
      <td>306987</td>
      <td>317947</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Barnet</td>
      <td>E09000003</td>
      <td>93284.5</td>
      <td>93190.2</td>
      <td>92247.5</td>
      <td>90762.9</td>
      <td>90258</td>
      <td>90107.2</td>
      <td>91441.2</td>
      <td>92361.3</td>
      <td>...</td>
      <td>522115</td>
      <td>526689</td>
      <td>526033</td>
      <td>518175</td>
      <td>523280</td>
      <td>527952</td>
      <td>536177</td>
      <td>534081</td>
      <td>535055</td>
      <td>534719</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bexley</td>
      <td>E09000004</td>
      <td>64958.1</td>
      <td>64787.9</td>
      <td>64367.5</td>
      <td>64277.7</td>
      <td>63997.1</td>
      <td>64252.3</td>
      <td>63722.7</td>
      <td>64432.6</td>
      <td>...</td>
      <td>339182</td>
      <td>341553</td>
      <td>339353</td>
      <td>340893</td>
      <td>344091</td>
      <td>346997</td>
      <td>345829</td>
      <td>346656</td>
      <td>350585</td>
      <td>354466</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Brent</td>
      <td>E09000005</td>
      <td>71306.6</td>
      <td>72022.3</td>
      <td>72015.8</td>
      <td>72965.6</td>
      <td>73704</td>
      <td>74310.5</td>
      <td>74127</td>
      <td>73547</td>
      <td>...</td>
      <td>466250</td>
      <td>470601</td>
      <td>482808</td>
      <td>484160</td>
      <td>482303</td>
      <td>493930</td>
      <td>519117</td>
      <td>523644</td>
      <td>517569</td>
      <td>496316</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 314 columns</p>
</div>



You're making great leaps forward. But your DataFrame still has lots of columns. Find out exactly how many by calling ***.columns*** on our DataFrame.  


```python
# Try this here. 
properties_T.columns
```




    Index([   'London_Borough',                'ID', 1995-01-01 00:00:00,
           1995-02-01 00:00:00, 1995-03-01 00:00:00, 1995-04-01 00:00:00,
           1995-05-01 00:00:00, 1995-06-01 00:00:00, 1995-07-01 00:00:00,
           1995-08-01 00:00:00,
           ...
           2020-03-01 00:00:00, 2020-04-01 00:00:00, 2020-05-01 00:00:00,
           2020-06-01 00:00:00, 2020-07-01 00:00:00, 2020-08-01 00:00:00,
           2020-09-01 00:00:00, 2020-10-01 00:00:00, 2020-11-01 00:00:00,
           2020-12-01 00:00:00],
          dtype='object', name=0, length=314)



#### 2.4. Transforming the data
Your data would be tidier if you had fewer columns. 

Wouldn't a ***single*** column for time be better than nearly 300? This single column will contain all of the datetimes in your current column headings. 

**Remember** the two most important properties of tidy data are:
1. **Each column is a variable.**

2. **Each row is an observation.**

One of the miraculous things about pandas is ***melt()***, which enables you to melt those values along the column headings of your current DataFrame into a single column.  

Make a new DataFrame called clean_properties, and assign it the return value of ***pd.melt()*** with the parameters: *properties_T* and *id_vars = ['Borough', 'ID']*. 

The result will be a DataFrame with rows representing the average house price within a given month and a given borough. Exactly what you want. 


```python
# Try this here: 
clean_properties = pd.melt(properties_T, id_vars= ['London_Borough', 'ID'])
```


```python
clean_properties.head()
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
      <th>London_Borough</th>
      <th>ID</th>
      <th>0</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>City of London</td>
      <td>E09000001</td>
      <td>1995-01-01</td>
      <td>91449</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1995-01-01</td>
      <td>50460.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barnet</td>
      <td>E09000003</td>
      <td>1995-01-01</td>
      <td>93284.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bexley</td>
      <td>E09000004</td>
      <td>1995-01-01</td>
      <td>64958.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brent</td>
      <td>E09000005</td>
      <td>1995-01-01</td>
      <td>71306.6</td>
    </tr>
  </tbody>
</table>
</div>



Awesome. This is looking good. 

Rename the '0' column 'Month', and the 'value' column 'Average_price'. 

Use the ***rename()*** method, and reassign *clean_properties* with the result. 


```python
# Re-name the column names
clean_properties = clean_properties.rename(columns = {0: 'Month', 'value': 'Average_price'})
```


```python
# Check out the DataFrame: 
clean_properties.head()
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
      <th>London_Borough</th>
      <th>ID</th>
      <th>Month</th>
      <th>Average_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>City of London</td>
      <td>E09000001</td>
      <td>1995-01-01</td>
      <td>91449</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1995-01-01</td>
      <td>50460.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barnet</td>
      <td>E09000003</td>
      <td>1995-01-01</td>
      <td>93284.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bexley</td>
      <td>E09000004</td>
      <td>1995-01-01</td>
      <td>64958.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brent</td>
      <td>E09000005</td>
      <td>1995-01-01</td>
      <td>71306.6</td>
    </tr>
  </tbody>
</table>
</div>



You need to check out the data types of your clean_properties DataFrame, just in case you need to do any type conversions. 


```python
# Let's use the .dtypes attribute to check the data types of our clean_properties DataFrame:
clean_properties.dtypes 
```




    London_Borough            object
    ID                        object
    Month             datetime64[ns]
    Average_price             object
    dtype: object



Change the Average_price column to a numeric type, specifically, a float.

Call the ***to_numeric()*** method on *pd*, pass the 'Average_price' column into its brackets, and reassign the result to the *clean_properties* 'Average_price' column.


```python
# Try this here
clean_properties['Average_price'] = pd.to_numeric(clean_properties['Average_price'],downcast='float')
```


```python
# Check out the new data types:
clean_properties.dtypes
```




    London_Borough            object
    ID                        object
    Month             datetime64[ns]
    Average_price            float32
    dtype: object




```python
# To see if there are any missing values, we should call the count() method on our DataFrame:
clean_properties.count()
```




    London_Borough    14976
    ID                14040
    Month             14976
    Average_price     14040
    dtype: int64



#### 2.5. Cleaning the data (Part 3) 
Houston, we have a problem!

There are fewer data points in some of the columns. Why might this be? Let's investigate.

Since there are only 32 London boroughs, check out the unique values of the 'London_Borough' column to see if they're all there.

Just call the ***unique()*** method on the London_Borough column. 


```python
# Do this here. 
clean_properties['London_Borough'].unique()
```




    array(['City of London', 'Barking & Dagenham', 'Barnet', 'Bexley',
           'Brent', 'Bromley', 'Camden', 'Croydon', 'Ealing', 'Enfield',
           'Greenwich', 'Hackney', 'Hammersmith & Fulham', 'Haringey',
           'Harrow', 'Havering', 'Hillingdon', 'Hounslow', 'Islington',
           'Kensington & Chelsea', 'Kingston upon Thames', 'Lambeth',
           'Lewisham', 'Merton', 'Newham', 'Redbridge',
           'Richmond upon Thames', 'Southwark', 'Sutton', 'Tower Hamlets',
           'Waltham Forest', 'Wandsworth', 'Westminster', 'Unnamed: 34',
           'Inner London', 'Outer London', 'Unnamed: 37', 'NORTH EAST',
           'NORTH WEST', 'YORKS & THE HUMBER', 'EAST MIDLANDS',
           'WEST MIDLANDS', 'EAST OF ENGLAND', 'LONDON', 'SOUTH EAST',
           'SOUTH WEST', 'Unnamed: 47', 'England'], dtype=object)



Aha! Some of these strings are not London boroughs. You're basically Sherlock Holmes, getting ever closer solving the mystery! 

The strings that don't belong:
- 'Unnamed: 34'
- 'Unnamed: 37'
- 'NORTH EAST'
- 'NORTH WEST'
- 'YORKS & THE HUMBER' 
- 'EAST MIDLANDS'
- 'WEST MIDLANDS'
- 'EAST OF ENGLAND'
- 'LONDON' 
- 'SOUTH EAST' 
- 'SOUTH WEST'
- 'Unnamed: 47' 
- 'England'

See what information is contained in rows where London_Boroughs is 'Unnamed’ and, if there’s nothing valuable, you can drop them.  To investigate, subset the clean_properties DataFrame on this condition.


```python
# Subset clean_properties on the condition: df['London_Borough'] == 'Unnamed: 34' to see what information these rows contain. 
clean_properties[clean_properties['London_Borough'] ==  'Unnamed: 34'].head()
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
      <th>London_Borough</th>
      <th>ID</th>
      <th>Month</th>
      <th>Average_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33</th>
      <td>Unnamed: 34</td>
      <td>NaN</td>
      <td>1995-01-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>81</th>
      <td>Unnamed: 34</td>
      <td>NaN</td>
      <td>1995-02-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>129</th>
      <td>Unnamed: 34</td>
      <td>NaN</td>
      <td>1995-03-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>177</th>
      <td>Unnamed: 34</td>
      <td>NaN</td>
      <td>1995-04-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>225</th>
      <td>Unnamed: 34</td>
      <td>NaN</td>
      <td>1995-05-01</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Let's do the same for 'Unnamed: 37':
clean_properties[clean_properties['London_Borough'] == 'Unnamed: 37'].head()
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
      <th>London_Borough</th>
      <th>ID</th>
      <th>Month</th>
      <th>Average_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>36</th>
      <td>Unnamed: 37</td>
      <td>NaN</td>
      <td>1995-01-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>84</th>
      <td>Unnamed: 37</td>
      <td>NaN</td>
      <td>1995-02-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>132</th>
      <td>Unnamed: 37</td>
      <td>NaN</td>
      <td>1995-03-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>180</th>
      <td>Unnamed: 37</td>
      <td>NaN</td>
      <td>1995-04-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>228</th>
      <td>Unnamed: 37</td>
      <td>NaN</td>
      <td>1995-05-01</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



These rows don't contain any valuable information. Delete them.



```python
# Let's look at how many rows have NAs as their value for ID. 
# To this end, subset clean_properties on the condition: clean_properties['ID'].isna().
# Notice that this line doesn't actually reassign a new value to clean_properties. 
clean_properties[clean_properties['ID'].isna()]
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
      <th>London_Borough</th>
      <th>ID</th>
      <th>Month</th>
      <th>Average_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33</th>
      <td>Unnamed: 34</td>
      <td>NaN</td>
      <td>1995-01-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Unnamed: 37</td>
      <td>NaN</td>
      <td>1995-01-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Unnamed: 47</td>
      <td>NaN</td>
      <td>1995-01-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>81</th>
      <td>Unnamed: 34</td>
      <td>NaN</td>
      <td>1995-02-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>84</th>
      <td>Unnamed: 37</td>
      <td>NaN</td>
      <td>1995-02-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>14916</th>
      <td>Unnamed: 37</td>
      <td>NaN</td>
      <td>2020-11-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14926</th>
      <td>Unnamed: 47</td>
      <td>NaN</td>
      <td>2020-11-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14961</th>
      <td>Unnamed: 34</td>
      <td>NaN</td>
      <td>2020-12-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14964</th>
      <td>Unnamed: 37</td>
      <td>NaN</td>
      <td>2020-12-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14974</th>
      <td>Unnamed: 47</td>
      <td>NaN</td>
      <td>2020-12-01</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>936 rows × 4 columns</p>
</div>



You always have a ***choice*** about how to deal with Null (NaN) values. We show you two methods today:
1. filtering on ***notna()***
2. reassigning on ***dropna()***

Try ***notna()*** first.  ***notna()*** will return a series of booleans, where the value will be true if there's a not a null and false if there is a null.

Make a new variable called *NaNFreeDF1* and assign it the result of filtering *clean_properties* on the condition: *clean_properties['Average_price'].notna()*


```python
# Try your hand at method (1) here: 
NaNFreeDF1 = clean_properties['Average_price'].notna()
NaNFreeDF1.head(48)
```




    0      True
    1      True
    2      True
    3      True
    4      True
    5      True
    6      True
    7      True
    8      True
    9      True
    10     True
    11     True
    12     True
    13     True
    14     True
    15     True
    16     True
    17     True
    18     True
    19     True
    20     True
    21     True
    22     True
    23     True
    24     True
    25     True
    26     True
    27     True
    28     True
    29     True
    30     True
    31     True
    32     True
    33    False
    34     True
    35     True
    36    False
    37     True
    38     True
    39     True
    40     True
    41     True
    42     True
    43     True
    44     True
    45     True
    46    False
    47     True
    Name: Average_price, dtype: bool




```python
# If we do a count on our new DataFrame, we'll see how many rows we have that have complete information: 
NaNFreeDF1.count()
```




    14976



Looks good! 

For completeness, now use ***dropna()***. ***dropna()*** will drop all null values. 

Make a new variable called *NaNFreeDF2*, and assign it the result of calling ***dropna()*** on *clean_properties*. 


```python
# filtering the data with NaN values
NaNFreeDF2=clean_properties.dropna()

```


```python
# Let's do a count on this DataFrame object: 
NaNFreeDF2.count()
```




    London_Borough    14040
    ID                14040
    Month             14040
    Average_price     14040
    dtype: int64




```python
NaNFreeDF2['London_Borough'].unique()
```




    array(['City of London', 'Barking & Dagenham', 'Barnet', 'Bexley',
           'Brent', 'Bromley', 'Camden', 'Croydon', 'Ealing', 'Enfield',
           'Greenwich', 'Hackney', 'Hammersmith & Fulham', 'Haringey',
           'Harrow', 'Havering', 'Hillingdon', 'Hounslow', 'Islington',
           'Kensington & Chelsea', 'Kingston upon Thames', 'Lambeth',
           'Lewisham', 'Merton', 'Newham', 'Redbridge',
           'Richmond upon Thames', 'Southwark', 'Sutton', 'Tower Hamlets',
           'Waltham Forest', 'Wandsworth', 'Westminster', 'Inner London',
           'Outer London', 'NORTH EAST', 'NORTH WEST', 'YORKS & THE HUMBER',
           'EAST MIDLANDS', 'WEST MIDLANDS', 'EAST OF ENGLAND', 'LONDON',
           'SOUTH EAST', 'SOUTH WEST', 'England'], dtype=object)



Both these methods did the job! Thus, you can pick either resultant DataFrame.


```python
# Using the .shape attribute, compare the dimenions of clean_properties, NaNFreeDF1, and NaNFreeDF2: 
print(clean_properties.shape)
print(NaNFreeDF1.shape)
print(NaNFreeDF2.shape)
```

    (14976, 4)
    (14976,)
    (14040, 4)
    

Our suggestions is to pick NaNFreeDF2.

Drop the rest of the invalid 'London Borough' values.

An elegant way to do this is to make a list of all those invalid values, then use the isin() method, combined with the negation operator ~, to remove those values. Call this list nonBoroughs.


```python
# A list of non-boroughs. 
nonBoroughs = ['Inner London', 'Outer London', 
               'NORTH EAST', 'NORTH WEST', 'YORKS & THE HUMBER', 
               'EAST MIDLANDS', 'WEST MIDLANDS',
              'EAST OF ENGLAND', 'LONDON', 'SOUTH EAST', 
              'SOUTH WEST', 'England']
```

Filter *NanFreeDF2* first on the condition that the rows' values for *London_Borough* is *in* the *nonBoroughs* list. 


```python
# Do this here. 
NaNFreeDF2['London_Borough'].isin(nonBoroughs)
```




    0        False
    1        False
    2        False
    3        False
    4        False
             ...  
    14970     True
    14971     True
    14972     True
    14973     True
    14975     True
    Name: London_Borough, Length: 14040, dtype: bool



You can now just put the negation operator *~* before the filter statement to get just those rows whose values for *London_Borough* is **not** in the *nonBoroughs* list:


```python
NaNFreeDF2[~NaNFreeDF2.London_Borough.isin(nonBoroughs)]
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
      <th>London_Borough</th>
      <th>ID</th>
      <th>Month</th>
      <th>Average_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>City of London</td>
      <td>E09000001</td>
      <td>1995-01-01</td>
      <td>91448.984375</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1995-01-01</td>
      <td>50460.226562</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barnet</td>
      <td>E09000003</td>
      <td>1995-01-01</td>
      <td>93284.515625</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bexley</td>
      <td>E09000004</td>
      <td>1995-01-01</td>
      <td>64958.089844</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brent</td>
      <td>E09000005</td>
      <td>1995-01-01</td>
      <td>71306.570312</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>14956</th>
      <td>Sutton</td>
      <td>E09000029</td>
      <td>2020-12-01</td>
      <td>398639.312500</td>
    </tr>
    <tr>
      <th>14957</th>
      <td>Tower Hamlets</td>
      <td>E09000030</td>
      <td>2020-12-01</td>
      <td>471445.031250</td>
    </tr>
    <tr>
      <th>14958</th>
      <td>Waltham Forest</td>
      <td>E09000031</td>
      <td>2020-12-01</td>
      <td>485676.343750</td>
    </tr>
    <tr>
      <th>14959</th>
      <td>Wandsworth</td>
      <td>E09000032</td>
      <td>2020-12-01</td>
      <td>618530.687500</td>
    </tr>
    <tr>
      <th>14960</th>
      <td>Westminster</td>
      <td>E09000033</td>
      <td>2020-12-01</td>
      <td>930070.312500</td>
    </tr>
  </tbody>
</table>
<p>10296 rows × 4 columns</p>
</div>



Then execute the reassignment: 




```python
NaNFreeDF2 = NaNFreeDF2[~NaNFreeDF2.London_Borough.isin(nonBoroughs)]
```


```python
NaNFreeDF2.head()
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
      <th>London_Borough</th>
      <th>ID</th>
      <th>Month</th>
      <th>Average_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>City of London</td>
      <td>E09000001</td>
      <td>1995-01-01</td>
      <td>91448.984375</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1995-01-01</td>
      <td>50460.226562</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barnet</td>
      <td>E09000003</td>
      <td>1995-01-01</td>
      <td>93284.515625</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bexley</td>
      <td>E09000004</td>
      <td>1995-01-01</td>
      <td>64958.089844</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brent</td>
      <td>E09000005</td>
      <td>1995-01-01</td>
      <td>71306.570312</td>
    </tr>
  </tbody>
</table>
</div>



Make a new variable called 'df', which is what data scientists typically name their final, analysis-ready DataFrame. 


```python
# Do that here. 
df = NaNFreeDF2
```


```python
df.head()
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
      <th>London_Borough</th>
      <th>ID</th>
      <th>Month</th>
      <th>Average_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>City of London</td>
      <td>E09000001</td>
      <td>1995-01-01</td>
      <td>91448.984375</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1995-01-01</td>
      <td>50460.226562</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barnet</td>
      <td>E09000003</td>
      <td>1995-01-01</td>
      <td>93284.515625</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bexley</td>
      <td>E09000004</td>
      <td>1995-01-01</td>
      <td>64958.089844</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brent</td>
      <td>E09000005</td>
      <td>1995-01-01</td>
      <td>71306.570312</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dtypes
```




    London_Borough            object
    ID                        object
    Month             datetime64[ns]
    Average_price            float32
    dtype: object



#### 2.6. Visualizing the data
Time to get a visual idea of the price shift occurring in the London boroughs. 

Restrict your observations to Camden for now. 

How have housing prices changed since 1995?


```python
# First of all, make a variable called camden_prices, and assign it the result of filtering df on the following condition:
# df['London_Borough'] == 'Camden'
camden_prices = df[df['London_Borough'] == 'Camden']
df['London_Borough'] = df['London_Borough'].astype(str)
# Make a variable called ax. Assign it the result of calling the plot() method, and plugging in the following values as parameters:
# kind ='line', x = 'Month', y='Average_price'
ax = camden_prices.plot(kind ='line', x= 'Month', y='Average_price')

# Finally, call the set_ylabel() method on ax, and set that label to the string: 'Price'. 
ax.set_ylabel('Price')

```




    Text(0, 0.5, 'Price')




    
![png](output_79_1.png)
    


To limit the amount of temporal data-points you have, it would be useful to extract the year from every value in our *Month* column. 300 is more datapoints than you need.

To this end, you'll apply a ***lambda function***. The logic works as follows. You'll:
1. look through the `Month` column
2. extract the year from each individual value in that column 
3. store that corresponding year as separate column


```python
# Try this yourself. 
#df["Month"].year
df['Year'] = df['Month'].apply(lambda t: t.year)

# Call the tail() method on df
df['Year'].tail()
```




    14956    2020
    14957    2020
    14958    2020
    14959    2020
    14960    2020
    Name: Year, dtype: int64



To calculate the mean house price for each year, you first need to **group by** the London_Borough and Year columns.

Make a new variable called *dfg*, and assign it the result of calling the ***groupby()*** method on *df*. Plug in the parameters: by=['Borough', 'Year']. To get the ***mean()*** of the result you'll chain that onto the end. 

We've helped you with this line, it's a little tricky. 


```python
# Using the function 'groupby' will help you to calculate the mean for each year and for each Borough. 
## As you can see, the variables Borough and Year are now indices
dfg = df.groupby(by=['London_Borough','Year']).mean()
dfg.sample(10)
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
      <th></th>
      <th>Average_price</th>
    </tr>
    <tr>
      <th>London_Borough</th>
      <th>Year</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Newham</th>
      <th>2012</th>
      <td>212824.703125</td>
    </tr>
    <tr>
      <th>Brent</th>
      <th>2007</th>
      <td>286995.343750</td>
    </tr>
    <tr>
      <th>Bexley</th>
      <th>2017</th>
      <td>335694.468750</td>
    </tr>
    <tr>
      <th>Kensington &amp; Chelsea</th>
      <th>2003</th>
      <td>460490.625000</td>
    </tr>
    <tr>
      <th>Hillingdon</th>
      <th>2017</th>
      <td>413586.562500</td>
    </tr>
    <tr>
      <th>Hounslow</th>
      <th>1999</th>
      <td>112073.664062</td>
    </tr>
    <tr>
      <th>Harrow</th>
      <th>2019</th>
      <td>449347.125000</td>
    </tr>
    <tr>
      <th>Kensington &amp; Chelsea</th>
      <th>2004</th>
      <td>512186.656250</td>
    </tr>
    <tr>
      <th>Brent</th>
      <th>1995</th>
      <td>73029.843750</td>
    </tr>
    <tr>
      <th>Hounslow</th>
      <th>2013</th>
      <td>290577.281250</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Let's reset the index for our new DataFrame dfg, and call the head() method on it. 
dfg.reset_index()
dfg.head()

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
      <th></th>
      <th>Average_price</th>
    </tr>
    <tr>
      <th>London_Borough</th>
      <th>Year</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Barking &amp; Dagenham</th>
      <th>1995</th>
      <td>51817.968750</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>51718.191406</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>55974.261719</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>60285.820312</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>65320.933594</td>
    </tr>
  </tbody>
</table>
</div>



### 3. Modelling
Now comes the really exciting stuff. 

You want to create a function that will calculate a ratio of house prices, that compares the price of a house in 2018 to the price in 1998. 

Call this function create_price_ratio. 

You want this function to:

1. Take a filter of dfg, specifically where this filter constrains the London_Borough, as an argument. For example, one admissible argument should be: **dfg[dfg['London_Borough']=='Camden']**. 

2. Get the Average Price for that borough for 1998 and, seperately, for 2018. 

3. Calculate the ratio of the Average Price for 1998 divided by the Average Price for 2018. 

4. Return that ratio. 

Once you've written this function, you'll use it to iterate through all the unique London_Boroughs and work out the ratio capturing the difference of house prices between 1998 and 2018.

***Hint***: This section should test the skills you acquired in:
- Python Data Science Toolbox (Part 1), all modules



```python
# Here's where you should write your function:
def create_price_ratio(d):
    y1998 = d['Average_price'][d['Year']==1998]
    y2018 = d['Average_price'][d['Year']==2018]
    y1998= y1998.astype(float)
    y2018= y2018.astype(float)
    ratio = y1998.div(y2018.values,axis=0)
    return ratio

```


```python
#  Test out the function by calling it with the following argument:
create_price_ratio(df[df['London_Borough'] == 'Barking & Dagenham'])
```




    1729    0.197426
    1777    0.195918
    1825    0.198505
    1873    0.203294
    1921    0.207952
    1969    0.205899
    2017    0.204820
    2065    0.205596
    2113    0.207423
    2161    0.206926
    2209    0.205679
    2257    0.210843
    dtype: float64




```python
# We want to do this for all of the London Boroughs. 
# First, let's make an empty dictionary, called final, where we'll store our ratios for each unique London_Borough.
final = {}
```


```python
# Now let's declare a for loop that will iterate through each of the unique elements of the 'London_Borough' column of our DataFrame dfg.
# Call the iterator variable 'b'. 
LB=df['London_Borough'].unique()

for b in LB:
    # Let's make our parameter to our create_price_ratio function: i.e., we subset dfg on 'London_Borough' == b. 
    borough = df[df['London_Borough'] == b]
    # Make a new entry in the final dictionary whose value's the result of calling create_price_ratio with the argument: borough
    final[b] = create_price_ratio(borough)
# We use the function and incorporate that into a new key of the dictionary 
#
#didn't quiet get what to do here
```

Now you have a dictionary with data about the ratio of average prices for each borough between 1998 and 2018,  but you can make it prettier by converting it to a DataFrame. 


```python
# Make a variable called df_ratios, and assign it the result of calling the DataFrame method on the dictionary final. 
df_ratios= pd.DataFrame(final)
```


```python
# Call the head() method on this variable to check it out. 
df_ratios.head()
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
      <th>City of London</th>
      <th>Barking &amp; Dagenham</th>
      <th>Barnet</th>
      <th>Bexley</th>
      <th>Brent</th>
      <th>Bromley</th>
      <th>Camden</th>
      <th>Croydon</th>
      <th>Ealing</th>
      <th>Enfield</th>
      <th>...</th>
      <th>Merton</th>
      <th>Newham</th>
      <th>Redbridge</th>
      <th>Richmond upon Thames</th>
      <th>Southwark</th>
      <th>Sutton</th>
      <th>Tower Hamlets</th>
      <th>Waltham Forest</th>
      <th>Wandsworth</th>
      <th>Westminster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1728</th>
      <td>0.155064</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1729</th>
      <td>NaN</td>
      <td>0.197426</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1730</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.21793</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1731</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.224684</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1732</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.189814</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>




```python
# All we need to do now is transpose it, and reset the index! 
df_ratios_T =df_ratios.T
#df_ratios = df_ratios_T.reset_index()
df_ratios.head()

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
      <th>City of London</th>
      <th>Barking &amp; Dagenham</th>
      <th>Barnet</th>
      <th>Bexley</th>
      <th>Brent</th>
      <th>Bromley</th>
      <th>Camden</th>
      <th>Croydon</th>
      <th>Ealing</th>
      <th>Enfield</th>
      <th>...</th>
      <th>Merton</th>
      <th>Newham</th>
      <th>Redbridge</th>
      <th>Richmond upon Thames</th>
      <th>Southwark</th>
      <th>Sutton</th>
      <th>Tower Hamlets</th>
      <th>Waltham Forest</th>
      <th>Wandsworth</th>
      <th>Westminster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1728</th>
      <td>0.155064</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1729</th>
      <td>NaN</td>
      <td>0.197426</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1730</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.21793</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1731</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.224684</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1732</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.189814</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>




```python
# Let's just rename the 'index' column as 'London_Borough', and the '0' column to '2018'.
df_ratios.rename(columns={'index':'London_Borough', 0:'2018'}, inplace=True)
df_ratios.head()
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
      <th>City of London</th>
      <th>Barking &amp; Dagenham</th>
      <th>Barnet</th>
      <th>Bexley</th>
      <th>Brent</th>
      <th>Bromley</th>
      <th>Camden</th>
      <th>Croydon</th>
      <th>Ealing</th>
      <th>Enfield</th>
      <th>...</th>
      <th>Merton</th>
      <th>Newham</th>
      <th>Redbridge</th>
      <th>Richmond upon Thames</th>
      <th>Southwark</th>
      <th>Sutton</th>
      <th>Tower Hamlets</th>
      <th>Waltham Forest</th>
      <th>Wandsworth</th>
      <th>Westminster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1728</th>
      <td>0.155064</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1729</th>
      <td>NaN</td>
      <td>0.197426</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1730</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.21793</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1731</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.224684</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1732</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.189814</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>




```python
# Let's sort in descending order and select the top 15 boroughs.
# Make a variable called top15, and assign it the result of calling sort_values() on df_ratios. 
top15= df.sort_values(['Average_price'])
top15.head(20)
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
      <th>London_Borough</th>
      <th>ID</th>
      <th>Month</th>
      <th>Average_price</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1995-01-01</td>
      <td>50460.226562</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>865</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1996-07-01</td>
      <td>50621.105469</td>
      <td>1996</td>
    </tr>
    <tr>
      <th>817</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1996-06-01</td>
      <td>50761.402344</td>
      <td>1996</td>
    </tr>
    <tr>
      <th>577</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1996-01-01</td>
      <td>50828.113281</td>
      <td>1996</td>
    </tr>
    <tr>
      <th>481</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1995-11-01</td>
      <td>50848.679688</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>529</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1995-12-01</td>
      <td>50945.183594</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1995-02-01</td>
      <td>51085.781250</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>913</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1996-08-01</td>
      <td>51104.691406</td>
      <td>1996</td>
    </tr>
    <tr>
      <th>97</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1995-03-01</td>
      <td>51268.968750</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>625</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1996-02-01</td>
      <td>51440.746094</td>
      <td>1996</td>
    </tr>
    <tr>
      <th>385</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1995-09-01</td>
      <td>51471.613281</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>433</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1995-10-01</td>
      <td>51513.757812</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>721</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1996-04-01</td>
      <td>51724.027344</td>
      <td>1996</td>
    </tr>
    <tr>
      <th>769</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1996-05-01</td>
      <td>51735.730469</td>
      <td>1996</td>
    </tr>
    <tr>
      <th>961</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1996-09-01</td>
      <td>51892.722656</td>
      <td>1996</td>
    </tr>
    <tr>
      <th>673</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1996-03-01</td>
      <td>51907.074219</td>
      <td>1996</td>
    </tr>
    <tr>
      <th>289</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1995-07-01</td>
      <td>52113.121094</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>1057</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1996-11-01</td>
      <td>52216.035156</td>
      <td>1996</td>
    </tr>
    <tr>
      <th>337</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1995-08-01</td>
      <td>52232.199219</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>1009</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1996-10-01</td>
      <td>52533.152344</td>
      <td>1996</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Let's plot the boroughs that have seen the greatest changes in price.
# Make a variable called ax. Assign it the result of filtering top15 on 'Borough' and '2018', then calling plot(), with
# the parameter kind = 'bar'. 
ax = top15.loc['London_Borough',2018].plot(kind="bar")
ax.set_xticklabels(['London_Borough'])
ax
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    C:\Users\toshiba\anaconda3\lib\site-packages\pandas\core\indexes\base.py in get_loc(self, key, method, tolerance)
       2894             try:
    -> 2895                 return self._engine.get_loc(casted_key)
       2896             except KeyError as err:
    

    pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()
    

    pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()
    

    pandas\_libs\index_class_helper.pxi in pandas._libs.index.Int64Engine._check_type()
    

    KeyError: 'London_Borough'

    
    The above exception was the direct cause of the following exception:
    

    KeyError                                  Traceback (most recent call last)

    <ipython-input-328-14104afd141f> in <module>
          2 # Make a variable called ax. Assign it the result of filtering top15 on 'Borough' and '2018', then calling plot(), with
          3 # the parameter kind = 'bar'.
    ----> 4 ax = top15.loc['London_Borough',2018].plot(kind="bar")
          5 ax.set_xticklabels(['London_Borough'])
          6 ax
    

    C:\Users\toshiba\anaconda3\lib\site-packages\pandas\core\indexing.py in __getitem__(self, key)
        871                     # AttributeError for IntervalTree get_value
        872                     pass
    --> 873             return self._getitem_tuple(key)
        874         else:
        875             # we by definition only have the 0th axis
    

    C:\Users\toshiba\anaconda3\lib\site-packages\pandas\core\indexing.py in _getitem_tuple(self, tup)
       1042     def _getitem_tuple(self, tup: Tuple):
       1043         try:
    -> 1044             return self._getitem_lowerdim(tup)
       1045         except IndexingError:
       1046             pass
    

    C:\Users\toshiba\anaconda3\lib\site-packages\pandas\core\indexing.py in _getitem_lowerdim(self, tup)
        784                 # We don't need to check for tuples here because those are
        785                 #  caught by the _is_nested_tuple_indexer check above.
    --> 786                 section = self._getitem_axis(key, axis=i)
        787 
        788                 # We should never have a scalar section here, because
    

    C:\Users\toshiba\anaconda3\lib\site-packages\pandas\core\indexing.py in _getitem_axis(self, key, axis)
       1108         # fall thru to straight lookup
       1109         self._validate_key(key, axis)
    -> 1110         return self._get_label(key, axis=axis)
       1111 
       1112     def _get_slice_axis(self, slice_obj: slice, axis: int):
    

    C:\Users\toshiba\anaconda3\lib\site-packages\pandas\core\indexing.py in _get_label(self, label, axis)
       1057     def _get_label(self, label, axis: int):
       1058         # GH#5667 this will fail if the label is not present in the axis.
    -> 1059         return self.obj.xs(label, axis=axis)
       1060 
       1061     def _handle_lowerdim_multi_index_axis0(self, tup: Tuple):
    

    C:\Users\toshiba\anaconda3\lib\site-packages\pandas\core\generic.py in xs(self, key, axis, level, drop_level)
       3489             loc, new_index = self.index.get_loc_level(key, drop_level=drop_level)
       3490         else:
    -> 3491             loc = self.index.get_loc(key)
       3492 
       3493             if isinstance(loc, np.ndarray):
    

    C:\Users\toshiba\anaconda3\lib\site-packages\pandas\core\indexes\base.py in get_loc(self, key, method, tolerance)
       2895                 return self._engine.get_loc(casted_key)
       2896             except KeyError as err:
    -> 2897                 raise KeyError(key) from err
       2898 
       2899         if tolerance is not None:
    

    KeyError: 'London_Borough'


### 4. Conclusion
Congratulation!  You're done. Excellent work.

What can you conclude? Type your conclusions below. 

We hope you enjoyed this practical project. It should have consolidated your data cleaning and pandas skills by looking at a real-world problem with the kind of dataset you might encounter as a budding data scientist. 


```python
#it was my first project. although jupyter gave me rough time, i finally completed
```
