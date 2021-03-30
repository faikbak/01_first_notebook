This exercise will require you to pull some data from the Qunadl API. Qaundl is currently the most widely used aggregator of financial market data.

As a first step, you will need to register a free account on the http://www.quandl.com website.

After you register, you will be provided with a unique API key, that you should store:


```python
# Store the API key as a string - according to PEP8, constants are always named in all upper case
API_KEY = 'RiQvfkr2G2rcZq1hECFo'
```

Qaundl has a large number of data sources, but, unfortunately, most of them require a Premium subscription. Still, there are also a good number of free datasets.

For this mini project, we will focus on equities data from the Frankfurt Stock Exhange (FSE), which is available for free. We'll try and analyze the stock prices of a company called Carl Zeiss Meditec, which manufactures tools for eye examinations, as well as medical lasers for laser eye surgery: https://www.zeiss.com/meditec/int/home.html. The company is listed under the stock ticker AFX_X.

You can find the detailed Quandl API instructions here: https://docs.quandl.com/docs/time-series

While there is a dedicated Python package for connecting to the Quandl API, we would prefer that you use the *requests* package, which can be easily downloaded using *pip* or *conda*. You can find the documentation for the package here: http://docs.python-requests.org/en/master/ 

Finally, apart from the *requests* package, you are encouraged to not use any third party Python packages, such as *pandas*, and instead focus on what's available in the Python Standard Library (the *collections* module might come in handy: https://pymotw.com/3/collections/ ).
Also, since you won't have access to DataFrames, you are encouraged to us Python's native data structures - preferably dictionaries, though some questions can also be answered using lists.
You can read more on these data structures here: https://docs.python.org/3/tutorial/datastructures.html

Keep in mind that the JSON responses you will be getting from the API map almost one-to-one to Python's dictionaries. Unfortunately, they can be very nested, so make sure you read up on indexing dictionaries in the documentation provided above.

# First, import the relevant modules


```python
import requests
import json
import numpy as np
```


```python
# Now, call the Quandl API and pull out a small sample of the data (only one day) to get a glimpse
# into the JSON structure that will be returned

url_='https://www.quandl.com/api/v3/datasets/FSE/AFX_X.json?api_key='+'RiQvfkr2G2rcZq1hECFo'+'&start_date=2018-01-01&end_date=2018-01-04'
r_= requests.get(url_)
```


```python
# Inspect the JSON structure of the object you created, and take note of how nested it is,
# as well as the overall structure
r_.json()
```




    {'dataset': {'id': 10095370,
      'dataset_code': 'AFX_X',
      'database_code': 'FSE',
      'name': 'Carl Zeiss Meditec (AFX_X)',
      'description': 'Stock Prices for Carl Zeiss Meditec (2020-11-02) from the Frankfurt Stock Exchange.<br><br>Trading System: Xetra<br><br>ISIN: DE0005313704',
      'refreshed_at': '2020-12-01T14:48:09.907Z',
      'newest_available_date': '2020-12-01',
      'oldest_available_date': '2000-06-07',
      'column_names': ['Date',
       'Open',
       'High',
       'Low',
       'Close',
       'Change',
       'Traded Volume',
       'Turnover',
       'Last Price of the Day',
       'Daily Traded Units',
       'Daily Turnover'],
      'frequency': 'daily',
      'type': 'Time Series',
      'premium': False,
      'limit': None,
      'transform': None,
      'column_index': None,
      'start_date': '2018-01-01',
      'end_date': '2018-01-04',
      'data': [['2018-01-02',
        52.05,
        52.4,
        51.2,
        51.4,
        None,
        54435.0,
        2807533.0,
        None,
        None,
        None]],
      'collapse': None,
      'order': None,
      'database_id': 6129}}



These are your tasks for this mini project:

1. Collect data from the Franfurt Stock Exchange, for the ticker AFX_X, for the whole year 2017 (keep in mind that the date format is YYYY-MM-DD).
2. Convert the returned JSON object into a Python dictionary.
3. Calculate what the highest and lowest opening prices were for the stock in this period.
4. What was the largest change in any one day (based on High and Low price)?
5. What was the largest change between any two days (based on Closing Price)?
6. What was the average daily trading volume during this year?
7. (Optional) What was the median trading volume during this year. (Note: you may need to implement your own function for calculating the median.)

# 1 Collect data from the Franfurt Stock Exchange, for the ticker AFX_X, for the whole year 2017


```python
url= requests.get('https://www.quandl.com/api/v3/datasets/FSE/AFX_X.json?api_key='+'RiQvfkr2G2rcZq1hECFo'+'&start_date=2017-01-01&end_date=2017-12-31')
```

# 2 Convert the returned JSON object into a Python dictionary


```python
url_dict=url.json()
type(url_dict)
```




    dict



# 3 Calculate what the highest and lowest opening prices were for the stock in this period.


```python
#highest opening price
import numpy as np
opening=[]
data_set=url_dict['dataset']['data']
for i in data_set:
    opening.append(i[1])
print('highest opening:',max(np.array(opening,dtype=float)))
print('lowest opening:',min(np.array(opening,dtype=float)))
```

    highest opening: 53.11
    lowest opening: 34.0
    

# 4 What was the largest change in any one day (based on High and Low price)?


```python
list_low=[]
list_high=[]
largest=[]
for i in data_set:
    list_low.append(i[3])
    list_high.append(i[2])
    largest.append(i[2]-i[3])
print("largest change:",max(largest))
```

    largest change: 2.8100000000000023
    

# 5 What was the largest change between any two days (based on Closing Price)?


```python
list_close=[]
#difference of closing prices in 2017
for i in data_set:
    list_close.append(i[4])

print("max:",max(list_close))
print("min:",min(list_close))

#max closing diffference betwween any two days
print(max(list_close)-min(list_close))
```

    max: 53.09
    min: 34.06
    19.03
    

# 6 What was the average daily trading volume during this year?


```python
average_daily=[]
for i in data_set:
    average_daily.append(i[6])
#print(average_daily)
print("The average daily trading volume:",sum(average_daily)/len(average_daily))
```

    The average daily trading volume: 89124.33725490196
    
