```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import random
import csv
```


```python
#Read the csv file
Google=pd.read_csv('googleplaystore.csv',thousands=' ')
#observe the first three entries
Google.head(3)
df=Google
```


```python
#store in it the path of the csv file that contains google dataset
#don't understand this one
```


```python
#Read the csv file
Apple =pd.read_csv("AppleStore.csv")
# Observe the first three entries
Apple.head(3)
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
      <th>id</th>
      <th>track_name</th>
      <th>size_bytes</th>
      <th>currency</th>
      <th>price</th>
      <th>rating_count_tot</th>
      <th>rating_count_ver</th>
      <th>user_rating</th>
      <th>user_rating_ver</th>
      <th>ver</th>
      <th>cont_rating</th>
      <th>prime_genre</th>
      <th>sup_devices.num</th>
      <th>ipadSc_urls.num</th>
      <th>lang.num</th>
      <th>vpp_lic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>281656475</td>
      <td>PAC-MAN Premium</td>
      <td>100788224</td>
      <td>USD</td>
      <td>3.99</td>
      <td>21292</td>
      <td>26</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>6.3.5</td>
      <td>4+</td>
      <td>Games</td>
      <td>38</td>
      <td>5</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>281796108</td>
      <td>Evernote - stay organized</td>
      <td>158578688</td>
      <td>USD</td>
      <td>0.00</td>
      <td>161065</td>
      <td>26</td>
      <td>4.0</td>
      <td>3.5</td>
      <td>8.2.2</td>
      <td>4+</td>
      <td>Productivity</td>
      <td>37</td>
      <td>5</td>
      <td>23</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>281940292</td>
      <td>WeatherBug - Local Weather, Radar, Maps, Alerts</td>
      <td>100524032</td>
      <td>USD</td>
      <td>0.00</td>
      <td>188583</td>
      <td>2822</td>
      <td>3.5</td>
      <td>4.5</td>
      <td>5.0.0</td>
      <td>4+</td>
      <td>Weather</td>
      <td>37</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#this isn't working i think,why? i couldn't see in the output
#Subset our DataFrame object Google variables ['Category', 'Rating', 'Reviews', 'Price']
Google=Google[['Category', 'Rating', 'Reviews', 'Price']]
Google.head(3)
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
      <th>Category</th>
      <th>Rating</th>
      <th>Reviews</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ART_AND_DESIGN</td>
      <td>4.1</td>
      <td>159</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ART_AND_DESIGN</td>
      <td>3.9</td>
      <td>967</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ART_AND_DESIGN</td>
      <td>4.7</td>
      <td>87510</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#select variables ['prime_genre', 'user_rating', 'rating_count_tot', 'price']
apple=Apple[['prime_genre', 'user_rating', 'rating_count_tot', 'price']]
#i didn't get what the problem here is

#check the first three entries
Apple.head(3)

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
      <th>id</th>
      <th>track_name</th>
      <th>size_bytes</th>
      <th>currency</th>
      <th>price</th>
      <th>rating_count_tot</th>
      <th>rating_count_ver</th>
      <th>user_rating</th>
      <th>user_rating_ver</th>
      <th>ver</th>
      <th>cont_rating</th>
      <th>prime_genre</th>
      <th>sup_devices.num</th>
      <th>ipadSc_urls.num</th>
      <th>lang.num</th>
      <th>vpp_lic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>281656475</td>
      <td>PAC-MAN Premium</td>
      <td>100788224</td>
      <td>USD</td>
      <td>3.99</td>
      <td>21292</td>
      <td>26</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>6.3.5</td>
      <td>4+</td>
      <td>Games</td>
      <td>38</td>
      <td>5</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>281796108</td>
      <td>Evernote - stay organized</td>
      <td>158578688</td>
      <td>USD</td>
      <td>0.00</td>
      <td>161065</td>
      <td>26</td>
      <td>4.0</td>
      <td>3.5</td>
      <td>8.2.2</td>
      <td>4+</td>
      <td>Productivity</td>
      <td>37</td>
      <td>5</td>
      <td>23</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>281940292</td>
      <td>WeatherBug - Local Weather, Radar, Maps, Alerts</td>
      <td>100524032</td>
      <td>USD</td>
      <td>0.00</td>
      <td>188583</td>
      <td>2822</td>
      <td>3.5</td>
      <td>4.5</td>
      <td>5.0.0</td>
      <td>4+</td>
      <td>Weather</td>
      <td>37</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#check out the data types of our Google
print(Google.dtypes)
#Google['Category', 'Rating', 'Reviews', 'Price'].dtypes
```

    Category     object
    Rating      float64
    Reviews      object
    Price        object
    dtype: object
    


```python
#Check again the unique values of Google
Google['Price'].unique()
```




    array(['0', '$4.99', '$3.99', '$6.99', '$1.49', '$2.99', '$7.99', '$5.99',
           '$3.49', '$1.99', '$9.99', '$7.49', '$0.99', '$9.00', '$5.49',
           '$10.00', '$24.99', '$11.99', '$79.99', '$16.99', '$14.99',
           '$1.00', '$29.99', '$12.99', '$2.49', '$10.99', '$1.50', '$19.99',
           '$15.99', '$33.99', '$74.99', '$39.99', '$3.95', '$4.49', '$1.70',
           '$8.99', '$2.00', '$3.88', '$25.99', '$399.99', '$17.99',
           '$400.00', '$3.02', '$1.76', '$4.84', '$4.77', '$1.61', '$2.50',
           '$1.59', '$6.49', '$1.29', '$5.00', '$13.99', '$299.99', '$379.99',
           '$37.99', '$18.99', '$389.99', '$19.90', '$8.49', '$1.75',
           '$14.00', '$4.85', '$46.99', '$109.99', '$154.99', '$3.08',
           '$2.59', '$4.80', '$1.96', '$19.40', '$3.90', '$4.59', '$15.46',
           '$3.04', '$4.29', '$2.60', '$3.28', '$4.60', '$28.99', '$2.95',
           '$2.90', '$1.97', '$200.00', '$89.99', '$2.56', '$30.99', '$3.61',
           '$394.99', '$1.26', 'Everyone', '$1.20', '$1.04'], dtype=object)




```python
#pick out just those rows whose value for the 'Price' column is NOT 'Everyone'
Google=Google[Google['Price']!='Everyone']
#Check again the unique values of Google
Google['Price'].unique()
```




    array(['0', '$4.99', '$3.99', '$6.99', '$1.49', '$2.99', '$7.99', '$5.99',
           '$3.49', '$1.99', '$9.99', '$7.49', '$0.99', '$9.00', '$5.49',
           '$10.00', '$24.99', '$11.99', '$79.99', '$16.99', '$14.99',
           '$1.00', '$29.99', '$12.99', '$2.49', '$10.99', '$1.50', '$19.99',
           '$15.99', '$33.99', '$74.99', '$39.99', '$3.95', '$4.49', '$1.70',
           '$8.99', '$2.00', '$3.88', '$25.99', '$399.99', '$17.99',
           '$400.00', '$3.02', '$1.76', '$4.84', '$4.77', '$1.61', '$2.50',
           '$1.59', '$6.49', '$1.29', '$5.00', '$13.99', '$299.99', '$379.99',
           '$37.99', '$18.99', '$389.99', '$19.90', '$8.49', '$1.75',
           '$14.00', '$4.85', '$46.99', '$109.99', '$154.99', '$3.08',
           '$2.59', '$4.80', '$1.96', '$19.40', '$3.90', '$4.59', '$15.46',
           '$3.04', '$4.29', '$2.60', '$3.28', '$4.60', '$28.99', '$2.95',
           '$2.90', '$1.97', '$200.00', '$89.99', '$2.56', '$30.99', '$3.61',
           '$394.99', '$1.26', '$1.20', '$1.04'], dtype=object)




```python
#find '$' and replace it with nothing

nosymb=Google['Price'].apply(str).replace('$','')
#doing same thing
#Google['Price'] = Google.Price.apply(lambda x: x.replace('$', ''))
Google.Price = pd.to_numeric(nosymb)
Google['Price'].unique()
```




    array([  0.  ,   4.99,   3.99,   6.99,   1.49,   2.99,   7.99,   5.99,
             3.49,   1.99,   9.99,   7.49,   0.99,   9.  ,   5.49,  10.  ,
            24.99,  11.99,  79.99,  16.99,  14.99,   1.  ,  29.99,  12.99,
             2.49,  10.99,   1.5 ,  19.99,  15.99,  33.99,  74.99,  39.99,
             3.95,   4.49,   1.7 ,   8.99,   2.  ,   3.88,  25.99, 399.99,
            17.99, 400.  ,   3.02,   1.76,   4.84,   4.77,   1.61,   2.5 ,
             1.59,   6.49,   1.29,   5.  ,  13.99, 299.99, 379.99,  37.99,
            18.99, 389.99,  19.9 ,   8.49,   1.75,  14.  ,   4.85,  46.99,
           109.99, 154.99,   3.08,   2.59,   4.8 ,   1.96,  19.4 ,   3.9 ,
             4.59,  15.46,   3.04,   4.29,   2.6 ,   3.28,   4.6 ,  28.99,
             2.95,   2.9 ,   1.97, 200.  ,  89.99,   2.56,  30.99,   3.61,
           394.99,   1.26,   1.2 ,   1.04])




```python
#check the data types for our Google dataframe again
print("dtypes:\n")
print(Google.dtypes)
Google.head(3)
```

    dtypes:
    
    Category     object
    Rating      float64
    Reviews      object
    Price       float64
    dtype: object
    




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
      <th>Category</th>
      <th>Rating</th>
      <th>Reviews</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ART_AND_DESIGN</td>
      <td>4.1</td>
      <td>159</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ART_AND_DESIGN</td>
      <td>3.9</td>
      <td>967</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ART_AND_DESIGN</td>
      <td>4.7</td>
      <td>87510</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Convert the 'Reviews' column to a numeric data type
Google['Reviews']=pd.to_numeric(Google['Reviews'])
```


```python
#check the data types of Google
print(Google.dtypes)
print(Apple.dtypes)
# Create a column called 'platform' in both the Apple and Google dataframes
Apple['platform']='apple'
Google['platform']='google'
```

    Category     object
    Rating      float64
    Reviews       int64
    Price       float64
    platform     object
    dtype: object
    A                     int64
    B                     int64
    C                    object
    D                     int64
    E                    object
    price               float64
    rating_count_tot      int64
    rating_count_ver      int64
    user_rating         float64
    user_rating_ver     float64
    ver                  object
    cont_rating          object
    prime_genre          object
    sup_devices.num       int64
    ipadSc_urls.num       int64
    lang.num              int64
    vpp_lic               int64
    platform             object
    dtype: object
    


```python
#store the column names of the Apple dataframe and replace
old_names=list(Apple.columns.values.tolist())
new_names=['A','B','C','D','E']
print(old_names)
#print columns
#cols = list(df.columns)
#print(cols)

Apple.rename(columns=dict(zip(old_names, new_names)), inplace=True)
Apple.rename(columns={'user_rating': 'Rating'}, inplace=False)

#changing column order if needed
#a, b = cols.index('LastName'), cols.index('MiddleName')
#cols[b], cols[a] = cols[a], cols[b]
#df = df[cols]
```

    ['A', 'B', 'C', 'D', 'E', 'price', 'rating_count_tot', 'rating_count_ver', 'user_rating', 'user_rating_ver', 'ver', 'cont_rating', 'prime_genre', 'sup_devices.num', 'ipadSc_urls.num', 'lang.num', 'vpp_lic', 'platform']
    




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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>price</th>
      <th>rating_count_tot</th>
      <th>rating_count_ver</th>
      <th>Rating</th>
      <th>user_rating_ver</th>
      <th>ver</th>
      <th>cont_rating</th>
      <th>prime_genre</th>
      <th>sup_devices.num</th>
      <th>ipadSc_urls.num</th>
      <th>lang.num</th>
      <th>vpp_lic</th>
      <th>platform</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>281656475</td>
      <td>PAC-MAN Premium</td>
      <td>100788224</td>
      <td>USD</td>
      <td>3.99</td>
      <td>21292</td>
      <td>26</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>6.3.5</td>
      <td>4+</td>
      <td>Games</td>
      <td>38</td>
      <td>5</td>
      <td>10</td>
      <td>1</td>
      <td>apple</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>281796108</td>
      <td>Evernote - stay organized</td>
      <td>158578688</td>
      <td>USD</td>
      <td>0.00</td>
      <td>161065</td>
      <td>26</td>
      <td>4.0</td>
      <td>3.5</td>
      <td>8.2.2</td>
      <td>4+</td>
      <td>Productivity</td>
      <td>37</td>
      <td>5</td>
      <td>23</td>
      <td>1</td>
      <td>apple</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>281940292</td>
      <td>WeatherBug - Local Weather, Radar, Maps, Alerts</td>
      <td>100524032</td>
      <td>USD</td>
      <td>0.00</td>
      <td>188583</td>
      <td>2822</td>
      <td>3.5</td>
      <td>4.5</td>
      <td>5.0.0</td>
      <td>4+</td>
      <td>Weather</td>
      <td>37</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>apple</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>282614216</td>
      <td>eBay: Best App to Buy, Sell, Save! Online Shop...</td>
      <td>128512000</td>
      <td>USD</td>
      <td>0.00</td>
      <td>262241</td>
      <td>649</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>5.10.0</td>
      <td>12+</td>
      <td>Shopping</td>
      <td>37</td>
      <td>5</td>
      <td>9</td>
      <td>1</td>
      <td>apple</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>282935706</td>
      <td>Bible</td>
      <td>92774400</td>
      <td>USD</td>
      <td>0.00</td>
      <td>985920</td>
      <td>5320</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>7.5.1</td>
      <td>4+</td>
      <td>Reference</td>
      <td>37</td>
      <td>5</td>
      <td>45</td>
      <td>1</td>
      <td>apple</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7192</th>
      <td>11081</td>
      <td>1187617475</td>
      <td>Kubik</td>
      <td>126644224</td>
      <td>USD</td>
      <td>0.00</td>
      <td>142</td>
      <td>75</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>1.3</td>
      <td>4+</td>
      <td>Games</td>
      <td>38</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>apple</td>
    </tr>
    <tr>
      <th>7193</th>
      <td>11082</td>
      <td>1187682390</td>
      <td>VR Roller-Coaster</td>
      <td>120760320</td>
      <td>USD</td>
      <td>0.00</td>
      <td>30</td>
      <td>30</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>0.9</td>
      <td>4+</td>
      <td>Games</td>
      <td>38</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>apple</td>
    </tr>
    <tr>
      <th>7194</th>
      <td>11087</td>
      <td>1187779532</td>
      <td>Bret Michaels Emojis + Lyric Keyboard</td>
      <td>111322112</td>
      <td>USD</td>
      <td>1.99</td>
      <td>15</td>
      <td>0</td>
      <td>4.5</td>
      <td>0.0</td>
      <td>1.0.2</td>
      <td>9+</td>
      <td>Utilities</td>
      <td>37</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>apple</td>
    </tr>
    <tr>
      <th>7195</th>
      <td>11089</td>
      <td>1187838770</td>
      <td>VR Roller Coaster World - Virtual Reality</td>
      <td>97235968</td>
      <td>USD</td>
      <td>0.00</td>
      <td>85</td>
      <td>32</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>1.0.15</td>
      <td>12+</td>
      <td>Games</td>
      <td>38</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>apple</td>
    </tr>
    <tr>
      <th>7196</th>
      <td>11097</td>
      <td>1188375727</td>
      <td>Escape the Sweet Shop Series</td>
      <td>90898432</td>
      <td>USD</td>
      <td>0.00</td>
      <td>3</td>
      <td>3</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>4+</td>
      <td>Games</td>
      <td>40</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>apple</td>
    </tr>
  </tbody>
</table>
<p>7197 rows × 18 columns</p>
</div>




```python
#append Apple to Google
Google.append(Apple) 
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
      <th>Category</th>
      <th>Rating</th>
      <th>Reviews</th>
      <th>Price</th>
      <th>platform</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>...</th>
      <th>rating_count_ver</th>
      <th>user_rating</th>
      <th>user_rating_ver</th>
      <th>ver</th>
      <th>cont_rating</th>
      <th>prime_genre</th>
      <th>sup_devices.num</th>
      <th>ipadSc_urls.num</th>
      <th>lang.num</th>
      <th>vpp_lic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>478</th>
      <td>DATING</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1.49</td>
      <td>google</td>
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
      <th>479</th>
      <td>DATING</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>2.99</td>
      <td>google</td>
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
      <th>621</th>
      <td>DATING</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>google</td>
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
      <th>623</th>
      <td>DATING</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>google</td>
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
      <th>627</th>
      <td>DATING</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>google</td>
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
      <th>7192</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>apple</td>
      <td>11081.0</td>
      <td>1.187617e+09</td>
      <td>Kubik</td>
      <td>126644224.0</td>
      <td>USD</td>
      <td>...</td>
      <td>75.0</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>1.3</td>
      <td>4+</td>
      <td>Games</td>
      <td>38.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7193</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>apple</td>
      <td>11082.0</td>
      <td>1.187682e+09</td>
      <td>VR Roller-Coaster</td>
      <td>120760320.0</td>
      <td>USD</td>
      <td>...</td>
      <td>30.0</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>0.9</td>
      <td>4+</td>
      <td>Games</td>
      <td>38.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7194</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>apple</td>
      <td>11087.0</td>
      <td>1.187780e+09</td>
      <td>Bret Michaels Emojis + Lyric Keyboard</td>
      <td>111322112.0</td>
      <td>USD</td>
      <td>...</td>
      <td>0.0</td>
      <td>4.5</td>
      <td>0.0</td>
      <td>1.0.2</td>
      <td>9+</td>
      <td>Utilities</td>
      <td>37.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7195</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>apple</td>
      <td>11089.0</td>
      <td>1.187839e+09</td>
      <td>VR Roller Coaster World - Virtual Reality</td>
      <td>97235968.0</td>
      <td>USD</td>
      <td>...</td>
      <td>32.0</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>1.0.15</td>
      <td>12+</td>
      <td>Games</td>
      <td>38.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7196</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>apple</td>
      <td>11097.0</td>
      <td>1.188376e+09</td>
      <td>Escape the Sweet Shop Series</td>
      <td>90898432.0</td>
      <td>USD</td>
      <td>...</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>4+</td>
      <td>Games</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>7793 rows × 22 columns</p>
</div>




```python
# Lets check first the dimesions of df before droping `NaN` values. Use the .shape feature.
print(Google.shape)
```

    (596, 5)
    


```python
#check 12 random points of your dataset
print("12 sample:\n ",Google.sample(12))

#first the dimensions of df before droping
print("dimension",Google.shape)
print(Google.size)
```

    12 sample:
                      Category  Rating  Reviews  Price platform
    5172             MEDICAL     NaN        0   0.00   google
    8106  NEWS_AND_MAGAZINES     NaN        0   0.00   google
    8871      ART_AND_DESIGN     NaN        0   0.00   google
    2425             MEDICAL     NaN        0   0.99   google
    2483             MEDICAL     NaN        0   0.00   google
    2479             MEDICAL     NaN        0   0.00   google
    5168             MEDICAL     NaN        0   0.00   google
    660               DATING     NaN        0   0.00   google
    6633       COMMUNICATION     NaN        0   0.00   google
    6277                GAME     NaN        0   0.99   google
    5843              FAMILY     NaN        0   0.00   google
    6959              FAMILY     NaN        0   0.00   google
    dimension (596, 5)
    2980
    


```python
#eliminate all the NaN values
Google.dropna()
print("after dropnan---------------------------------\n")

#Check the new dimesions of our dataframe
print(Google.size)
```

    after dropnan---------------------------------
    
    2980
    


```python
# Subset your df to pick out just those rows whose value for 'Reviews' is equal to 0. Do a count() on the result.
print("non-zero review count\n")
# Eliminate the points that have 0 reviews
Google=Google[Google['Reviews']==0]
print(len(Google))

```

    non-zero review count
    
    596
    


```python
#To summarize analytically, let's use the groupby() method on our df.

Google.groupby(['platform','Rating'])

# Call the boxplot() method on our df.
Google.boxplot(by='platform', column=['Rating'])
#why no Apple in the list? why no boxplot?
```




    <AxesSubplot:title={'center':'Rating'}, xlabel='platform'>




    
![png](output_19_1.png)
    



```python
#PART3
# Call the subsets 'apple' and 'google'
#unable to figure out what the problem is
contain_values = df[df['platform'].str.contains('apple')]
print (contain_values)
apple=df[df['platform']=='apple']['Rating']
google=df[df['platform']=='google']['Rating']
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    C:\Users\toshiba\anaconda3\lib\site-packages\pandas\core\indexes\base.py in get_loc(self, key, method, tolerance)
       2894             try:
    -> 2895                 return self._engine.get_loc(casted_key)
       2896             except KeyError as err:
    

    pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()
    

    pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()
    

    pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()
    

    pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()
    

    KeyError: 'platform'

    
    The above exception was the direct cause of the following exception:
    

    KeyError                                  Traceback (most recent call last)

    <ipython-input-222-684c1d5b5de1> in <module>
          1 #PART3
          2 # Call the subsets 'apple' and 'google'
    ----> 3 contain_values = df[df['platform'].str.contains('apple')]
          4 print (contain_values)
          5 apple=df[df['platform']=='apple']['Rating']
    

    C:\Users\toshiba\anaconda3\lib\site-packages\pandas\core\frame.py in __getitem__(self, key)
       2900             if self.columns.nlevels > 1:
       2901                 return self._getitem_multilevel(key)
    -> 2902             indexer = self.columns.get_loc(key)
       2903             if is_integer(indexer):
       2904                 indexer = [indexer]
    

    C:\Users\toshiba\anaconda3\lib\site-packages\pandas\core\indexes\base.py in get_loc(self, key, method, tolerance)
       2895                 return self._engine.get_loc(casted_key)
       2896             except KeyError as err:
    -> 2897                 raise KeyError(key) from err
       2898 
       2899         if tolerance is not None:
    

    KeyError: 'platform'



```python
#get an indication of whether the apple data are normally distributed
#the data are normally distributed, the lower the p-value in the result of this test, the more likely the data are to be non-normal.
# Save the result in a variable
apple_normal = stats.normaltest(apple)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-194-fb361dc981e4> in <module>
          2 #the data are normally distributed, the lower the p-value in the result of this test, the more likely the data are to be non-normal.
          3 # Save the result in a variable
    ----> 4 apple_normal = stats.normaltest(apple)
    

    C:\Users\toshiba\anaconda3\lib\site-packages\scipy\stats\stats.py in normaltest(a, axis, nan_policy)
       1700         return mstats_basic.normaltest(a, axis)
       1701 
    -> 1702     s, _ = skewtest(a, axis)
       1703     k, _ = kurtosistest(a, axis)
       1704     k2 = s*s + k*k
    

    C:\Users\toshiba\anaconda3\lib\site-packages\scipy\stats\stats.py in skewtest(a, axis, nan_policy)
       1515         a = np.ravel(a)
       1516         axis = 0
    -> 1517     b2 = skew(a, axis)
       1518     n = a.shape[axis]
       1519     if n < 8:
    

    C:\Users\toshiba\anaconda3\lib\site-packages\scipy\stats\stats.py in skew(a, axis, bias, nan_policy)
       1228         return mstats_basic.skew(a, axis, bias)
       1229 
    -> 1230     m2 = moment(a, 2, axis)
       1231     m3 = moment(a, 3, axis)
       1232     zero = (m2 == 0)
    

    C:\Users\toshiba\anaconda3\lib\site-packages\scipy\stats\stats.py in moment(a, moment, axis, nan_policy)
       1040         return np.array(mmnt)
       1041     else:
    -> 1042         return _moment(a, moment, axis)
       1043 
       1044 
    

    C:\Users\toshiba\anaconda3\lib\site-packages\scipy\stats\stats.py in _moment(a, moment, axis)
       1080 
       1081         # Starting point for exponentiation by squares
    -> 1082         a_zero_mean = a - np.expand_dims(np.mean(a, axis), axis)
       1083         if n_list[-1] == 1:
       1084             s = a_zero_mean.copy()
    

    <__array_function__ internals> in mean(*args, **kwargs)
    

    C:\Users\toshiba\anaconda3\lib\site-packages\numpy\core\fromnumeric.py in mean(a, axis, dtype, out, keepdims)
       3370             return mean(axis=axis, dtype=dtype, out=out, **kwargs)
       3371 
    -> 3372     return _methods._mean(a, axis=axis, dtype=dtype,
       3373                           out=out, **kwargs)
       3374 
    

    C:\Users\toshiba\anaconda3\lib\site-packages\numpy\core\_methods.py in _mean(a, axis, dtype, out, keepdims)
        160     ret = umr_sum(arr, axis, dtype, out, keepdims)
        161     if isinstance(ret, mu.ndarray):
    --> 162         ret = um.true_divide(
        163                 ret, rcount, out=ret, casting='unsafe', subok=False)
        164         if is_float16_result and out is None:
    

    TypeError: unsupported operand type(s) for /: 'str' and 'int'



```python
# Create a histogram of the apple reviews distribution
histoApple=plt.hist(Apple)

# Create a histogram of the google data
histoGoogle=plt.hist(Google)
```


```python
#Create a column called `Permutation1`, and assign to it the result of permuting (shuffling) the Rating column
df.groupby(by='platform')['Permutation1'].describe()

# Lets compare with the previous analytical summary
df.groupby(by='platform')['Rating'].describe()
#what else shpuld i do?

#make a list called difference
difference=list()

# Make a variable called 'histo', and assign to it the result of plotting a histogram of the difference list. 
histo = plt.hist(difference)
```


```python
# make a for loop that does the following 10,000 times:
# 1. makes a permutation of the 'Rating'
df.groupby(by='platform')['Permutation1'].describe()
# 2. calculates the difference in the mean rating for apple and the mean rating for google.
difference.append(np.mean(permutation[df['platform']=='apple']) - np.mean(permutation[df['platform']=='google']))

for i in range(1000):
    permutation = np.random.permutation(df[groupby(by='platform')['Permutation1'].describe()])
    difference.append(np.mean(permutation[df['platform']=='apple']) - np.mean(permutation[df['platform']=='google']))
```


```python
#make a variable called obs_difference, and assign it the result of the mean of our 'apple' variable and the mean of our 'google' variable
obs_difference = np.mean(apple)-np.mean(google)

# Make this difference absolute with the built-in abs() function
obs_difference =abs(obs_difference)

# Print out this value; it should be 0.1420605474512291. 
print("obs_difference="+obs_difference)
```


```python
#PART4
positiveExtremes = []
negativeExtremes = []
for i in range(len(difference)):
    if (difference[i] >= obs_difference):
        positiveExtremes.append(difference[i])
    elif (difference[i] <= -obs_difference):
        negativeExtremes.append(difference[i])

print(len(negativeExtremes))
print("burasi negative extreme")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-328-3bb2c1cb35db> in <module>
          2 positiveExtremes = []
          3 negativeExtremes = []
    ----> 4 for i in range(len(difference)):
          5     if (difference[i] >= obs_difference):
          6         positiveExtremes.append(difference[i])
    

    NameError: name 'difference' is not defined



```python

```
