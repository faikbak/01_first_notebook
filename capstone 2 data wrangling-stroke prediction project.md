```python
import pandas as pd
import numpy as np
```


```python
#read csv from https://www.kaggle.com/fedesoriano/stroke-prediction-dataset
healthcare=pd.read_csv('healthcare-dataset-stroke-data.csv')
```


```python
#first observe general structure of data
healthcare.sample(20)
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
      <th>4533</th>
      <td>44799</td>
      <td>Female</td>
      <td>32.00</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>66.30</td>
      <td>47.5</td>
      <td>never smoked</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2094</th>
      <td>6199</td>
      <td>Female</td>
      <td>52.00</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Govt_job</td>
      <td>Rural</td>
      <td>107.27</td>
      <td>30.1</td>
      <td>Unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>808</th>
      <td>21397</td>
      <td>Female</td>
      <td>40.00</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Govt_job</td>
      <td>Urban</td>
      <td>122.74</td>
      <td>23.3</td>
      <td>Unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1139</th>
      <td>32733</td>
      <td>Female</td>
      <td>28.00</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>106.68</td>
      <td>29.3</td>
      <td>never smoked</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2800</th>
      <td>8655</td>
      <td>Female</td>
      <td>51.00</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Urban</td>
      <td>100.96</td>
      <td>33.4</td>
      <td>never smoked</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1097</th>
      <td>34326</td>
      <td>Male</td>
      <td>52.00</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>229.20</td>
      <td>35.6</td>
      <td>formerly smoked</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2742</th>
      <td>50681</td>
      <td>Female</td>
      <td>36.00</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>90.22</td>
      <td>28.7</td>
      <td>formerly smoked</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1231</th>
      <td>69355</td>
      <td>Male</td>
      <td>3.00</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>children</td>
      <td>Rural</td>
      <td>86.38</td>
      <td>22.8</td>
      <td>Unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1864</th>
      <td>29232</td>
      <td>Female</td>
      <td>56.00</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>114.33</td>
      <td>30.7</td>
      <td>smokes</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3333</th>
      <td>59872</td>
      <td>Female</td>
      <td>38.00</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>80.82</td>
      <td>49.3</td>
      <td>never smoked</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2372</th>
      <td>43827</td>
      <td>Female</td>
      <td>27.00</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>161.57</td>
      <td>25.7</td>
      <td>smokes</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4262</th>
      <td>17098</td>
      <td>Female</td>
      <td>12.00</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>children</td>
      <td>Urban</td>
      <td>116.06</td>
      <td>25.9</td>
      <td>Unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>753</th>
      <td>49529</td>
      <td>Female</td>
      <td>1.16</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>children</td>
      <td>Urban</td>
      <td>60.98</td>
      <td>17.2</td>
      <td>Unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>760</th>
      <td>37413</td>
      <td>Female</td>
      <td>39.00</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>77.54</td>
      <td>32.7</td>
      <td>Unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>452</th>
      <td>23221</td>
      <td>Male</td>
      <td>29.00</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>Private</td>
      <td>Urban</td>
      <td>83.51</td>
      <td>37.1</td>
      <td>never smoked</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1141</th>
      <td>30352</td>
      <td>Male</td>
      <td>57.00</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>90.06</td>
      <td>29.8</td>
      <td>Unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1863</th>
      <td>33692</td>
      <td>Female</td>
      <td>12.00</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>children</td>
      <td>Rural</td>
      <td>85.97</td>
      <td>35.7</td>
      <td>Unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3574</th>
      <td>7964</td>
      <td>Male</td>
      <td>24.00</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>Private</td>
      <td>Urban</td>
      <td>97.47</td>
      <td>24.2</td>
      <td>formerly smoked</td>
      <td>0</td>
    </tr>
    <tr>
      <th>528</th>
      <td>41800</td>
      <td>Female</td>
      <td>23.00</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>79.35</td>
      <td>39.4</td>
      <td>formerly smoked</td>
      <td>0</td>
    </tr>
    <tr>
      <th>732</th>
      <td>31308</td>
      <td>Female</td>
      <td>49.00</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>114.50</td>
      <td>35.9</td>
      <td>formerly smoked</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#list column names
healthcare.columns
```




    Index(['id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
           'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
           'smoking_status', 'stroke'],
          dtype='object')




```python
#see datatypes
healthcare.dtypes
```




    id                     int64
    gender                object
    age                  float64
    hypertension           int64
    heart_disease          int64
    ever_married          object
    work_type             object
    Residence_type        object
    avg_glucose_level    float64
    bmi                  float64
    smoking_status        object
    stroke                 int64
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
     0   id                 5110 non-null   int64  
     1   gender             5110 non-null   object 
     2   age                5110 non-null   float64
     3   hypertension       5110 non-null   int64  
     4   heart_disease      5110 non-null   int64  
     5   ever_married       5110 non-null   object 
     6   work_type          5110 non-null   object 
     7   Residence_type     5110 non-null   object 
     8   avg_glucose_level  5110 non-null   float64
     9   bmi                4909 non-null   float64
     10  smoking_status     5110 non-null   object 
     11  stroke             5110 non-null   int64  
    dtypes: float64(3), int64(4), object(5)
    memory usage: 479.2+ KB
    




    (5110, 12)




```python
healthcare.bmi.sample(10)
```




    1142    24.1
    4589    39.7
    1961    24.7
    331     30.5
    3648    37.5
    4213    24.1
    3444    25.0
    3429    27.9
    3038    22.2
    364     35.1
    Name: bmi, dtype: float64




```python
healthcare.isna().sample(10)
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
      <th>465</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2566</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4237</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3752</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2819</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3904</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3696</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1323</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1836</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>445</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
healthcare.dtypes
```




    id                     int64
    gender                object
    age                  float64
    hypertension           int64
    heart_disease          int64
    ever_married          object
    work_type             object
    Residence_type        object
    avg_glucose_level    float64
    bmi                  float64
    smoking_status        object
    stroke                 int64
    dtype: object




```python
#Ranges of values
#i didn't understand what to do here/
```


```python
#mean
healthcare.mean()
```




    id                   36517.829354
    age                     43.226614
    hypertension             0.097456
    heart_disease            0.054012
    avg_glucose_level      106.147677
    bmi                     28.893237
    stroke                   0.048728
    dtype: float64




```python
#median
healthcare.median()
```




    id                   36932.000
    age                     45.000
    hypertension             0.000
    heart_disease            0.000
    avg_glucose_level       91.885
    bmi                     28.100
    stroke                   0.000
    dtype: float64




```python
healthcare.std()
```




    id                   21161.721625
    age                     22.612647
    hypertension             0.296607
    heart_disease            0.226063
    avg_glucose_level       45.283560
    bmi                      7.854067
    stroke                   0.215320
    dtype: float64




```python
healthcare.avg_glucose_level.mean()
```




    106.14767710371804




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
#- Do your column names correspond to what those columns store?
    #no,i didn't quite get what bmi means for example. needs to be investigated in order to understand what it is

#- Check the data types of your columns. Are they sensible?
    #no,age column is float type,heart_disease should have been categorical type

#- Calculate summary statistics for each of your columns, such as mean, median, mode, standard deviation, range, and
#number of unique values. What does this tell you about your data? What do you now need to investigate?
    #glucose level seem a little bit strange. i would study this category first
    #especially non categorical data might need to be cleaned
```
