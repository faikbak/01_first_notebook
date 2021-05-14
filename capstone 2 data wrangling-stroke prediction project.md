```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
#read csv from https://www.kaggle.com/fedesoriano/stroke-prediction-dataset
healthcare=pd.read_csv('healthcare-dataset-stroke-data.csv')
```


```python
#first observe general structure of data
healthcare.sample(5)
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
      <th>id</th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status</th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2333</th>
      <td>77</td>
      <td>Female</td>
      <td>13.0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>children</td>
      <td>Rural</td>
      <td>85.81</td>
      <td>18.6</td>
      <td>Unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2297</th>
      <td>58359</td>
      <td>Female</td>
      <td>71.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>129.97</td>
      <td>44.2</td>
      <td>smokes</td>
      <td>0</td>
    </tr>
    <tr>
      <th>904</th>
      <td>56089</td>
      <td>Female</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>63.64</td>
      <td>31.3</td>
      <td>formerly smoked</td>
      <td>0</td>
    </tr>
    <tr>
      <th>946</th>
      <td>62709</td>
      <td>Female</td>
      <td>47.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>204.63</td>
      <td>43.4</td>
      <td>never smoked</td>
      <td>0</td>
    </tr>
    <tr>
      <th>533</th>
      <td>26028</td>
      <td>Male</td>
      <td>51.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>98.41</td>
      <td>32.1</td>
      <td>never smoked</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



    -we see Unknown labels under smoking_status, we need to fix these.
    -maybe filtering avg_glucose_level above average or levels peoeple who have stroke will be helpful
    -grouping cases according to residence_type will also be helpful
    -ages below a certain age might not be out of our scope
    -looking for any correlation might be a good idea
    -bmi stands for body mass index


```python
#list column names
cols=healthcare.columns
cols
```




    Index(['id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
           'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
           'smoking_status', 'stroke'],
          dtype='object')



column names look alright but need some fixation to make them more understandable


```python
healthcare.rename(columns = {'heart_disease':'heart disease','ever_married':'ever married','work_type':'work type',
                                        'Residence_type':'residence type','avg_glucose_level':'avg glucose level',
                                       'bmi':'body mas','smoking_status':'smoking status'}, inplace = False)
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
      <th>id</th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart disease</th>
      <th>ever married</th>
      <th>work type</th>
      <th>residence type</th>
      <th>avg glucose level</th>
      <th>body mas</th>
      <th>smoking status</th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9046</td>
      <td>Male</td>
      <td>67.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>228.69</td>
      <td>36.6</td>
      <td>formerly smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>51676</td>
      <td>Female</td>
      <td>61.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Rural</td>
      <td>202.21</td>
      <td>NaN</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>31112</td>
      <td>Male</td>
      <td>80.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>105.92</td>
      <td>32.5</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60182</td>
      <td>Female</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>171.23</td>
      <td>34.4</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1665</td>
      <td>Female</td>
      <td>79.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Rural</td>
      <td>174.12</td>
      <td>24.0</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5105</th>
      <td>18234</td>
      <td>Female</td>
      <td>80.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>83.75</td>
      <td>NaN</td>
      <td>never smoked</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5106</th>
      <td>44873</td>
      <td>Female</td>
      <td>81.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Urban</td>
      <td>125.20</td>
      <td>40.0</td>
      <td>never smoked</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5107</th>
      <td>19723</td>
      <td>Female</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Rural</td>
      <td>82.99</td>
      <td>30.6</td>
      <td>never smoked</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5108</th>
      <td>37544</td>
      <td>Male</td>
      <td>51.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>166.29</td>
      <td>25.6</td>
      <td>formerly smoked</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5109</th>
      <td>44679</td>
      <td>Female</td>
      <td>44.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Govt_job</td>
      <td>Urban</td>
      <td>85.28</td>
      <td>26.2</td>
      <td>Unknown</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5110 rows × 12 columns</p>
</div>




```python
#see datatypes
healthcare.dtypes
```




    id                   category
    gender               category
    age                   float64
    hypertension         category
    heart_disease        category
    ever_married         category
    work_type            category
    Residence_type       category
    avg_glucose_level     float64
    bmi                   float64
    smoking_status       category
    stroke               category
    dtype: object



some column dtypes need to be changed. we typecast categorical features to a category dtype because they make the operations on such columns much faster than the object dtype.


```python

col_names=['id','heart_disease','hypertension','gender','ever_married','work_type','Residence_type','smoking_status','stroke']
for col in col_names:
    healthcare[col] = healthcare[col].astype('category',copy=False)
```


```python
healthcare.dtypes
```




    id                   category
    gender               category
    age                   float64
    hypertension         category
    heart_disease        category
    ever_married         category
    work_type            category
    Residence_type       category
    avg_glucose_level     float64
    bmi                   float64
    smoking_status       category
    stroke               category
    dtype: object




```python
#describe to columns
healthcare.info()
healthcare.shape
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5110 entries, 0 to 5109
    Data columns (total 12 columns):
     #   Column             Non-Null Count  Dtype   
    ---  ------             --------------  -----   
     0   id                 5110 non-null   category
     1   gender             5110 non-null   category
     2   age                5110 non-null   float64 
     3   hypertension       5110 non-null   category
     4   heart_disease      5110 non-null   category
     5   ever_married       5110 non-null   category
     6   work_type          5110 non-null   category
     7   Residence_type     5110 non-null   category
     8   avg_glucose_level  5110 non-null   float64 
     9   bmi                4909 non-null   float64 
     10  smoking_status     5110 non-null   category
     11  stroke             5110 non-null   category
    dtypes: category(9), float64(3)
    memory usage: 370.7 KB
    




    (5110, 12)




```python
healthcare.isna().sum()
```




    id                     0
    gender                 0
    age                    0
    hypertension           0
    heart_disease          0
    ever_married           0
    work_type              0
    Residence_type         0
    avg_glucose_level      0
    bmi                  201
    smoking_status         0
    stroke                 0
    dtype: int64




```python
#mean
healthcare.mean()
```




    age                   43.226614
    avg_glucose_level    106.147677
    bmi                   28.893237
    dtype: float64




```python
#median
healthcare.median()
```




    age                  45.000
    avg_glucose_level    91.885
    bmi                  28.100
    dtype: float64




```python
#median after dtype change
healthcare.median()
```




    age                  45.000
    avg_glucose_level    91.885
    bmi                  28.100
    dtype: float64




```python
healthcare.std()
```




    age                  22.612647
    avg_glucose_level    45.283560
    bmi                   7.854067
    dtype: float64




```python
#std after dtype change
healthcare.std()
```




    age                  22.612647
    avg_glucose_level    45.283560
    bmi                   7.854067
    dtype: float64




```python
healthcare.mean()
```




    age                   43.226614
    avg_glucose_level    106.147677
    bmi                   28.893237
    dtype: float64




```python
healthcare.median()
```




    age                  45.000
    avg_glucose_level    91.885
    bmi                  28.100
    dtype: float64




```python
healthcare.mode()
#The mode of a data values is the value that appears most often.
#It is the value at which the data is most likely to be sampled
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
      <th>id</th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status</th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>67</td>
      <td>Female</td>
      <td>78.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>93.88</td>
      <td>28.7</td>
      <td>never smoked</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77</td>
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
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84</td>
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
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>91</td>
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
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>99</td>
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
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5105</th>
      <td>72911</td>
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
      <td>NaN</td>
    </tr>
    <tr>
      <th>5106</th>
      <td>72914</td>
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
      <td>NaN</td>
    </tr>
    <tr>
      <th>5107</th>
      <td>72915</td>
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
      <td>NaN</td>
    </tr>
    <tr>
      <th>5108</th>
      <td>72918</td>
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
      <td>NaN</td>
    </tr>
    <tr>
      <th>5109</th>
      <td>72940</td>
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
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5110 rows × 12 columns</p>
</div>




```python
healthcare.nunique()
```




    id                   5110
    gender                  3
    age                   104
    hypertension            2
    heart_disease           2
    ever_married            2
    work_type               5
    Residence_type          2
    avg_glucose_level    3979
    bmi                   418
    smoking_status          4
    stroke                  2
    dtype: int64




```python
#What does this tell you about your data? What do you now need to investigate?
    #glucose level seem a little bit strange. i would study this category first
    #especially non categorical data might need to be cleaned
```


```python
healthcare['age'].max()
```




    82.0




```python
healthcare['age'].min()
```




    0.08



let's sort age column to see age distribution


```python
healthcare.sort_values(by='age')
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
      <th>id</th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status</th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3295</th>
      <td>29955</td>
      <td>Male</td>
      <td>0.08</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>children</td>
      <td>Rural</td>
      <td>70.33</td>
      <td>16.9</td>
      <td>Unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1614</th>
      <td>47350</td>
      <td>Female</td>
      <td>0.08</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>children</td>
      <td>Urban</td>
      <td>139.67</td>
      <td>14.1</td>
      <td>Unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3618</th>
      <td>22877</td>
      <td>Male</td>
      <td>0.16</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>children</td>
      <td>Urban</td>
      <td>114.71</td>
      <td>17.4</td>
      <td>Unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4021</th>
      <td>8247</td>
      <td>Male</td>
      <td>0.16</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>children</td>
      <td>Urban</td>
      <td>109.52</td>
      <td>13.9</td>
      <td>Unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3968</th>
      <td>41500</td>
      <td>Male</td>
      <td>0.16</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>children</td>
      <td>Rural</td>
      <td>69.79</td>
      <td>13.0</td>
      <td>Unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4590</th>
      <td>19271</td>
      <td>Female</td>
      <td>82.00</td>
      <td>1</td>
      <td>1</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Urban</td>
      <td>101.56</td>
      <td>31.5</td>
      <td>never smoked</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4094</th>
      <td>25510</td>
      <td>Male</td>
      <td>82.00</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Urban</td>
      <td>111.81</td>
      <td>19.8</td>
      <td>formerly smoked</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2341</th>
      <td>10649</td>
      <td>Female</td>
      <td>82.00</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>80.00</td>
      <td>33.6</td>
      <td>never smoked</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4716</th>
      <td>5387</td>
      <td>Female</td>
      <td>82.00</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>Private</td>
      <td>Rural</td>
      <td>96.98</td>
      <td>21.5</td>
      <td>never smoked</td>
      <td>0</td>
    </tr>
    <tr>
      <th>187</th>
      <td>67895</td>
      <td>Female</td>
      <td>82.00</td>
      <td>1</td>
      <td>1</td>
      <td>Yes</td>
      <td>Govt_job</td>
      <td>Urban</td>
      <td>215.94</td>
      <td>27.9</td>
      <td>formerly smoked</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5110 rows × 12 columns</p>
</div>



so 'work_type=children' are really children


```python
healthcare['smoking_status'].value_counts()
```




    never smoked       1892
    Unknown            1544
    formerly smoked     885
    smokes              789
    Name: smoking_status, dtype: int64



for column mode we use loc method,so that we can mode a sepsific column


```python
healthcare.loc[:,'smoking_status'].mode()
```




    0    never smoked
    Name: smoking_status, dtype: category
    Categories (4, object): ['Unknown', 'formerly smoked', 'never smoked', 'smokes']


