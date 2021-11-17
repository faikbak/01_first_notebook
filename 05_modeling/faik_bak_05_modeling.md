# Big Mountain's facilities modeling <a id='Big Mountain's facilities modeling'></a>

## 5.1 Contents<a id='5.1_Contents'></a>
* [5 Modeling](#5_Modeling)
  * [5.1 Contents](#5.1_Contents)
  * [5.2 Introduction](#5.2_Introduction)
  * [5.3 Imports](#5.3_Imports)
  * [5.4 Load Model](#5.4_Load_Model)
  * [5.5 Load Data](#5.5_Load_Data)
  * [5.6 Refit Model On All Available Data (excluding Big Mountain)](#5.6_Refit_Model_On_All_Available_Data_(excluding_Big_Mountain))
  * [5.7 Calculate Expected Big Mountain Ticket Price From The Model](#5.7_Calculate_Expected_Big_Mountain_Ticket_Price_From_The_Model)
  * [5.8 Big Mountain Resort In Market Context](#5.8_Big_Mountain_Resort_In_Market_Context)
    * [5.8.1 Ticket price](#5.8.1_Ticket_price)
    * [5.8.2 Vertical drop](#5.8.2_Vertical_drop)
    * [5.8.3 Snow making area](#5.8.3_Snow_making_area)
    * [5.8.4 Total number of chairs](#5.8.4_Total_number_of_chairs)
    * [5.8.5 Fast quads](#5.8.5_Fast_quads)
    * [5.8.6 Runs](#5.8.6_Runs)
    * [5.8.7 Longest run](#5.8.7_Longest_run)
    * [5.8.8 Trams](#5.8.8_Trams)
    * [5.8.9 Skiable terrain area](#5.8.9_Skiable_terrain_area)
  * [5.9 Modeling scenarios](#5.9_Modeling_scenarios)
    * [5.9.1 Scenario 1](#5.9.1_Scenario_1)
    * [5.9.2 Scenario 2](#5.9.2_Scenario_2)
    * [5.9.3 Scenario 3](#5.9.3_Scenario_3)
    * [5.9.4 Scenario 4](#5.9.4_Scenario_4)
  * [5.10 Summary](#5.10_Summary)
  * [5.11 Further work](#5.11_Further_work)


## 5.2 Introduction<a id='5.2_Introduction'></a>

In this notebook, we now take our model for ski resort ticket price and leverage it to gain some insights into what price Big Mountain's facilities might actually support as well as explore the sensitivity of changes to various resort parameters. Note that this relies on the implicit assumption that all other resorts are largely setting prices based on how much people value certain facilities. Essentially this assumes prices are set by a free market.

We can now use our model to gain insight into what Big Mountain's ideal ticket price could/should be, and how that might change under various scenarios.

## 5.3 Imports<a id='5.3_Imports'></a>


```python
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import cross_validate
```

## 5.4 Load Model<a id='5.4_Load_Model'></a>


```python
# This isn't exactly production-grade, but a quick check for development
# These checks can save some head-scratching in development when moving from
# one python environment to another, for example
expected_model_version = '1.0'
model_path = './ski_resort_pricing_model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    if model.version != expected_model_version:
        print("Expected model version doesn't match version loaded")
    if model.sklearn_version != sklearn_version:
        print("Warning: model created under different sklearn version")
else:
    print("Expected model not found")
```

    C:\Users\toshiba\anaconda3\lib\site-packages\sklearn\base.py:329: UserWarning: Trying to unpickle estimator SimpleImputer from version 0.22.1 when using version 0.23.2. This might lead to breaking code or invalid results. Use at your own risk.
      warnings.warn(
    C:\Users\toshiba\anaconda3\lib\site-packages\sklearn\base.py:329: UserWarning: Trying to unpickle estimator DecisionTreeRegressor from version 0.22.1 when using version 0.23.2. This might lead to breaking code or invalid results. Use at your own risk.
      warnings.warn(
    

    Warning: model created under different sklearn version
    

    C:\Users\toshiba\anaconda3\lib\site-packages\sklearn\base.py:329: UserWarning: Trying to unpickle estimator RandomForestRegressor from version 0.22.1 when using version 0.23.2. This might lead to breaking code or invalid results. Use at your own risk.
      warnings.warn(
    C:\Users\toshiba\anaconda3\lib\site-packages\sklearn\base.py:329: UserWarning: Trying to unpickle estimator Pipeline from version 0.22.1 when using version 0.23.2. This might lead to breaking code or invalid results. Use at your own risk.
      warnings.warn(
    

## 5.5 Load Data<a id='5.5_Load_Data'></a>


```python
ski_data = pd.read_csv('./data/ski_data_step3_features.csv')
```


```python
big_mountain = ski_data[ski_data.Name == 'Big Mountain Resort']
```


```python
big_mountain.T
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
      <th>124</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Name</th>
      <td>Big Mountain Resort</td>
    </tr>
    <tr>
      <th>Region</th>
      <td>Montana</td>
    </tr>
    <tr>
      <th>state</th>
      <td>Montana</td>
    </tr>
    <tr>
      <th>summit_elev</th>
      <td>6817</td>
    </tr>
    <tr>
      <th>vertical_drop</th>
      <td>2353</td>
    </tr>
    <tr>
      <th>base_elev</th>
      <td>4464</td>
    </tr>
    <tr>
      <th>trams</th>
      <td>0</td>
    </tr>
    <tr>
      <th>fastSixes</th>
      <td>0</td>
    </tr>
    <tr>
      <th>fastQuads</th>
      <td>3</td>
    </tr>
    <tr>
      <th>quad</th>
      <td>2</td>
    </tr>
    <tr>
      <th>triple</th>
      <td>6</td>
    </tr>
    <tr>
      <th>double</th>
      <td>0</td>
    </tr>
    <tr>
      <th>surface</th>
      <td>3</td>
    </tr>
    <tr>
      <th>total_chairs</th>
      <td>14</td>
    </tr>
    <tr>
      <th>Runs</th>
      <td>105</td>
    </tr>
    <tr>
      <th>TerrainParks</th>
      <td>4</td>
    </tr>
    <tr>
      <th>LongestRun_mi</th>
      <td>3.3</td>
    </tr>
    <tr>
      <th>SkiableTerrain_ac</th>
      <td>3000</td>
    </tr>
    <tr>
      <th>Snow Making_ac</th>
      <td>600</td>
    </tr>
    <tr>
      <th>daysOpenLastYear</th>
      <td>123</td>
    </tr>
    <tr>
      <th>yearsOpen</th>
      <td>72</td>
    </tr>
    <tr>
      <th>averageSnowfall</th>
      <td>333</td>
    </tr>
    <tr>
      <th>AdultWeekend</th>
      <td>81</td>
    </tr>
    <tr>
      <th>projectedDaysOpen</th>
      <td>123</td>
    </tr>
    <tr>
      <th>NightSkiing_ac</th>
      <td>600</td>
    </tr>
    <tr>
      <th>resorts_per_state</th>
      <td>12</td>
    </tr>
    <tr>
      <th>resorts_per_100kcapita</th>
      <td>1.12278</td>
    </tr>
    <tr>
      <th>resorts_per_100ksq_mile</th>
      <td>8.16104</td>
    </tr>
    <tr>
      <th>resort_skiable_area_ac_state_ratio</th>
      <td>0.140121</td>
    </tr>
    <tr>
      <th>resort_days_open_state_ratio</th>
      <td>0.129338</td>
    </tr>
    <tr>
      <th>resort_terrain_park_state_ratio</th>
      <td>0.148148</td>
    </tr>
    <tr>
      <th>resort_night_skiing_state_ratio</th>
      <td>0.84507</td>
    </tr>
    <tr>
      <th>total_chairs_runs_ratio</th>
      <td>0.133333</td>
    </tr>
    <tr>
      <th>total_chairs_skiable_ratio</th>
      <td>0.00466667</td>
    </tr>
    <tr>
      <th>fastQuads_runs_ratio</th>
      <td>0.0285714</td>
    </tr>
    <tr>
      <th>fastQuads_skiable_ratio</th>
      <td>0.001</td>
    </tr>
  </tbody>
</table>
</div>



## 5.6 Refit Model On All Available Data (excluding Big Mountain)<a id='5.6_Refit_Model_On_All_Available_Data_(excluding_Big_Mountain)'></a>

This next step requires some careful thought. We want to refit the model using all available data. But should we include Big Mountain data? On the one hand, we are _not_ trying to estimate model performance on a previously unseen data sample, so theoretically including Big Mountain data should be fine. One might first think that including Big Mountain in the model training would, if anything, improve model performance in predicting Big Mountain's ticket price. But here's where our business context comes in. The motivation for this entire project is based on the sense that Big Mountain needs to adjust its pricing. One way to phrase this problem: we want to train a model to predict Big Mountain's ticket price based on data from _all the other_ resorts! We don't want Big Mountain's current price to bias this. We want to calculate a price based only on its competitors.


```python
X = ski_data.loc[ski_data.Name != "Big Mountain Resort", model.X_columns]
y = ski_data.loc[ski_data.Name != "Big Mountain Resort", 'AdultWeekend']
```


```python
len(X), len(y)
```




    (276, 276)




```python
model.fit(X, y)
```




    Pipeline(steps=[('simpleimputer', SimpleImputer(strategy='median')),
                    ('standardscaler', None),
                    ('randomforestregressor',
                     RandomForestRegressor(n_estimators=69, random_state=47))])




```python
cv_results = cross_validate(model, X, y, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
```


```python
cv_results['test_score']
```




    array([-11.54734731,  -9.58081159, -11.44871673,  -7.87807115,
           -11.08494335])




```python
mae_mean, mae_std = np.mean(-1 * cv_results['test_score']), np.std(-1 * cv_results['test_score'])
mae_mean, mae_std
```




    (10.307978025597592, 1.4052910437391404)



These numbers will inevitably be different to those in the previous step that used a different training data set. They should, however, be consistent. It's important to appreciate that estimates of model performance are subject to the noise and uncertainty of data!

## 5.7 Calculate Expected Big Mountain Ticket Price From The Model<a id='5.7_Calculate_Expected_Big_Mountain_Ticket_Price_From_The_Model'></a>


```python
X_bm = ski_data.loc[ski_data.Name == "Big Mountain Resort", model.X_columns]
y_bm = ski_data.loc[ski_data.Name == "Big Mountain Resort", 'AdultWeekend']
```


```python
bm_pred = model.predict(X_bm).item()
```


```python
y_bm = y_bm.values.item()
```


```python
print(f'Big Mountain Resort modelled price is ${bm_pred:.2f}, actual price is ${y_bm:.2f}.')
print(f'Even with the expected mean absolute error of ${mae_mean:.2f}, this suggests there is room for an increase.')
```

This result should be looked at optimistically and doubtfully! The validity of our model lies in the assumption that other resorts accurately set their prices according to what the market (the ticket-buying public) supports. The fact that our resort seems to be charging that much less that what's predicted suggests our resort might be undercharging. 
But if ours is mispricing itself, are others? It's reasonable to expect that some resorts will be "overpriced" and some "underpriced." Or if resorts are pretty good at pricing strategies, it could be that our model is simply lacking some key data? Certainly we know nothing about operating costs, for example, and they would surely help.

## 5.8 Big Mountain Resort In Market Context<a id='5.8_Big_Mountain_Resort_In_Market_Context'></a>

Features that came up as important in the modeling (not just our final, random forest model) included:
* vertical_drop
* Snow Making_ac
* total_chairs
* fastQuads
* Runs
* LongestRun_mi
* trams
* SkiableTerrain_ac

A handy glossary of skiing terms can be found on the [ski.com](https://www.ski.com/ski-glossary) site. Some potentially relevant contextual information is that vertical drop, although nominally the height difference from the summit to the base, is generally taken from the highest [_lift-served_](http://verticalfeet.com/) point.

It's often useful to define custom functions for visualizing data in meaningful ways. The function below takes a feature name as an input and plots a histogram of the values of that feature. It then marks where Big Mountain sits in the distribution by marking Big Mountain's value with a vertical line using `matplotlib`'s [axvline](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.axvline.html) function. It also performs a little cleaning up of missing values and adds descriptive labels and a title.


```python
#Code task 1#
#Add code to the `plot_compare` function that displays a vertical, dashed line
#on the histogram to indicate Big Mountain's position in the distribution
#Hint: plt.axvline() plots a vertical line, its position for 'feature1'
#would be `big_mountain['feature1'].values, we'd like a red line, which can be
#specified with c='r', a dashed linestyle is produced by ls='--',
#and it's nice to give it a slightly reduced alpha value, such as 0.8.
#Don't forget to give it a useful label (e.g. 'Big Mountain') so it's listed
#in the legend.
def plot_compare(feat_name, description, state=None, figsize=(10, 5)):
    """Graphically compare distributions of features.
    
    Plot histogram of values for all resorts and reference line to mark
    Big Mountain's position.
    
    Arguments:
    feat_name - the feature column name in the data
    description - text description of the feature
    state - select a specific state (None for all states)
    figsize - (optional) figure size
    """
    
    plt.subplots(figsize=figsize)
    # quirk that hist sometimes objects to NaNs, sometimes doesn't
    # filtering only for finite values tidies this up
    if state is None:
        ski_x = ski_data[feat_name]
    else:
        ski_x = ski_data.loc[ski_data.state == state, feat_name]
    ski_x = ski_x[np.isfinite(ski_x)]
    plt.hist(ski_x, bins=30)
    plt.axvline(big_mountain[feat_name].values, c='r', ls='--', alpha=0.8, label='Big mountain')
    plt.xlabel(description)
    plt.ylabel('frequency')
    plt.title(description + ' distribution for resorts in market share')
    plt.legend()
```

### 5.8.1 Ticket price<a id='5.8.1_Ticket_price'></a>

Look at where Big Mountain sits overall amongst all resorts for price and for just other resorts in Montana.


```python
plot_compare('AdultWeekend', 'Adult weekend ticket price ($)')
```


    
![png](output_34_0.png)
    



```python
plot_compare('AdultWeekend', 'Adult weekend ticket price ($) - Montana only', state='Montana')
```


    
![png](output_35_0.png)
    


### 5.8.2 Vertical drop<a id='5.8.2_Vertical_drop'></a>


```python
plot_compare('vertical_drop', 'Vertical drop (feet)')
```


    
![png](output_37_0.png)
    


Big Mountain is doing well for vertical drop, but there are still quite a few resorts with a greater drop.

### 5.8.3 Snow making area<a id='5.8.3_Snow_making_area'></a>


```python
plot_compare('Snow Making_ac', 'Area covered by snow makers (acres)')
```


    
![png](output_40_0.png)
    


Big Mountain is very high up the league table of snow making area.

### 5.8.4 Total number of chairs<a id='5.8.4_Total_number_of_chairs'></a>


```python
plot_compare('total_chairs', 'Total number of chairs')
```


    
![png](output_43_0.png)
    


Big Mountain has amongst the highest number of total chairs, resorts with more appear to be outliers.

### 5.8.5 Fast quads<a id='5.8.5_Fast_quads'></a>


```python
plot_compare('fastQuads', 'Number of fast quads')
```


    
![png](output_46_0.png)
    


Most resorts have no fast quads. Big Mountain has 3, which puts it high up that league table. There are some values  much higher, but they are rare.

### 5.8.6 Runs<a id='5.8.6_Runs'></a>


```python
plot_compare('Runs', 'Total number of runs')
```


    
![png](output_49_0.png)
    


Big Mountain compares well for the number of runs. There are some resorts with more, but not many.

### 5.8.7 Longest run<a id='5.8.7_Longest_run'></a>


```python
plot_compare('LongestRun_mi', 'Longest run length (miles)')
```


    
![png](output_52_0.png)
    


Big Mountain has one of the longest runs. Although it is just over half the length of the longest, the longer ones are rare.

### 5.8.8 Trams<a id='5.8.8_Trams'></a>


```python
plot_compare('trams', 'Number of trams')
```


    
![png](output_55_0.png)
    


The vast majority of resorts, such as Big Mountain, have no trams.

### 5.8.9 Skiable terrain area<a id='5.8.9_Skiable_terrain_area'></a>


```python
plot_compare('SkiableTerrain_ac', 'Skiable terrain area (acres)')
```


    
![png](output_58_0.png)
    


Big Mountain is amongst the resorts with the largest amount of skiable terrain.

## 5.9 Modeling scenarios<a id='5.9_Modeling_scenarios'></a>

Big Mountain Resort has been reviewing potential scenarios for either cutting costs or increasing revenue (from ticket prices). Ticket price is not determined by any set of parameters; the resort is free to set whatever price it likes. However, the resort operates within a market where people pay more for certain facilities, and less for others. Being able to sense how facilities support a given ticket price is valuable business intelligence. This is where the utility of our model comes in.

The business has shortlisted some options:
1. Permanently closing down up to 10 of the least used runs. This doesn't impact any other resort statistics.
2. Increase the vertical drop by adding a run to a point 150 feet lower down but requiring the installation of an additional chair lift to bring skiers back up, without additional snow making coverage
3. Same as number 2, but adding 2 acres of snow making cover
4. Increase the longest run by 0.2 mile to boast 3.5 miles length, requiring an additional snow making coverage of 4 acres

The expected number of visitors over the season is 350,000 and, on average, visitors ski for five days. Assume the provided data includes the additional lift that Big Mountain recently installed.


```python
expected_visitors = 350_000
```


```python
all_feats = ['vertical_drop', 'Snow Making_ac', 'total_chairs', 'fastQuads', 
             'Runs', 'LongestRun_mi', 'trams', 'SkiableTerrain_ac']
big_mountain[all_feats]
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
      <th>vertical_drop</th>
      <th>Snow Making_ac</th>
      <th>total_chairs</th>
      <th>fastQuads</th>
      <th>Runs</th>
      <th>LongestRun_mi</th>
      <th>trams</th>
      <th>SkiableTerrain_ac</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>124</th>
      <td>2353</td>
      <td>600.0</td>
      <td>14</td>
      <td>3</td>
      <td>105.0</td>
      <td>3.3</td>
      <td>0</td>
      <td>3000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Code task 2#
#In this function, copy the Big Mountain data into a new data frame
#(Note we use .copy()!)
#And then for each feature, and each of its deltas (changes from the original),
#create the modified scenario dataframe (bm2) and make a ticket price prediction
#for it. The difference between the scenario's prediction and the current
#prediction is then calculated and returned.
#Complete the code to increment each feature by the associated delta
def predict_increase(features, deltas):
    """Increase in modelled ticket price by applying delta to feature.
    
    Arguments:
    features - list, names of the features in the ski_data dataframe to change
    deltas - list, the amounts by which to increase the values of the features
    
    Outputs:
    Amount of increase in the predicted ticket price
    """
    
    bm2 = X_bm.copy()
    for f, d in zip(features, deltas):
        bm2[features] += bm2
    return model.predict(bm2).item() - model.predict(X_bm).item()
```

### 5.9.1 Scenario 1<a id='5.9.1_Scenario_1'></a>

Close up to 10 of the least used runs. The number of runs is the only parameter varying.


```python
[i for i in range(-1, -11, -1)]
```




    [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]




```python
runs_delta = [i for i in range(-1, -11, -1)]
price_deltas = [predict_increase(['Runs'], [delta]) for delta in runs_delta]
```


```python
price_deltas
"""[0.0,
 -0.4057971014492807,
 -0.6666666666666714,
 -0.6666666666666714,
 -0.6666666666666714,
 -1.2608695652173907,
 -1.2608695652173907,
 -1.2608695652173907,
 -1.7101449275362341,
 -1.8115942028985472]"""
```




    '[0.0,\n -0.4057971014492807,\n -0.6666666666666714,\n -0.6666666666666714,\n -0.6666666666666714,\n -1.2608695652173907,\n -1.2608695652173907,\n -1.2608695652173907,\n -1.7101449275362341,\n -1.8115942028985472]'




```python
#Code task 3#
#Create two plots, side by side, for the predicted ticket price change (delta) for each
#condition (number of runs closed) in the scenario and the associated predicted revenue
#change on the assumption that each of the expected visitors buys 5 tickets
#There are two things to do here:
#1 - use a list comprehension to create a list of the number of runs closed from `runs_delta`
#2 - use a list comprehension to create a list of predicted revenue changes from `price_deltas`
runs_closed = [-1 * runs for runs in runs_delta] #1
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.subplots_adjust(wspace=0.5)
ax[0].plot(runs_closed, price_deltas, 'o-')
ax[0].set(xlabel='Runs closed', ylabel='Change ($)', title='Ticket price')
revenue_deltas = [5 * expected_visitors * runs_closed for  revenue_deltas in 'price_deltas'] #2
ax[1].plot(runs_closed, revenue_deltas, 'o-')
ax[1].set(xlabel='Runs closed', ylabel='Change ($)', title='Revenue');
```

The model says closing one run makes no difference. Closing 2 and 3 successively reduces support for ticket price and so revenue. If Big Mountain closes down 3 runs, it seems they may as well close down 4 or 5 as there's no further loss in ticket price. Increasing the closures down to 6 or more leads to a large drop. 

### 5.9.2 Scenario 2<a id='5.9.2_Scenario_2'></a>

In this scenario, Big Mountain is adding a run, increasing the vertical drop by 150 feet, and installing an additional chair lift.


```python
#Code task 4#
#Call `predict_increase` with a list of the features 'Runs', 'vertical_drop', and 'total_chairs'
#and associated deltas of 1, 150, and 1
ticket2_increase = predict_increase(['Runs', 'vertical_drop', 'total_chairs'], [1,150,1])
revenue2_increase = 5 * expected_visitors * ticket2_increase
```


```python
print(f'This scenario increases support for ticket price by ${ticket2_increase:.2f}')
print(f'Over the season, this could be expected to amount to ${revenue2_increase:.0f}')
```

### 5.9.3 Scenario 3<a id='5.9.3_Scenario_3'></a>

In this scenario, you are repeating the previous one but adding 2 acres of snow making.


```python
#Code task 5#
#Repeat scenario 2 conditions, but add an increase of 2 to `Snow Making_ac`
ticket3_increase = predict_increase(['Runs', 'vertical_drop', 'total_chairs', ___], [1, 150, 1, ___])
revenue3_increase = 5 * expected_visitors * ticket3_increase
```


```python
print(f'This scenario increases support for ticket price by ${ticket3_increase:.2f}')
print(f'Over the season, this could be expected to amount to ${revenue3_increase:.0f}')
```

Such a small increase in the snow making area makes no difference!

### 5.9.4 Scenario 4<a id='5.9.4_Scenario_4'></a>

This scenario calls for increasing the longest run by .2 miles and guaranteeing its snow coverage by adding 4 acres of snow making capability.


```python
#Code task 6#
#Predict the increase from adding 0.2 miles to `LongestRun_mi` and 4 to `Snow Making_ac`
predict_increase([___, ___], [___, ___])
```

No difference whatsoever. Although the longest run feature was used in the linear model, the random forest model (the one we chose because of its better performance) only has longest run way down in the feature importance list. 

## 5.10 Summary<a id='5.10_Summary'></a>

**Q: 1** Write a summary of the results of modeling these scenarios. Start by starting the current position; how much does Big Mountain currently charge? What does your modelling suggest for a ticket price that could be supported in the marketplace by Big Mountain's facilities? How would you approach suggesting such a change to the business leadership? Discuss the additional operating cost of the new chair lift per ticket (on the basis of each visitor on average buying 5 day tickets) in the context of raising prices to cover this. For future improvements, state which, if any, of the modeled scenarios you'd recommend for further consideration. Suggest how the business might test, and progress, with any run closures.

**A: 1** Your answer here

## 5.11 Further work<a id='5.11_Further_work'></a>

**Q: 2** What next? Highlight any deficiencies in the data that hampered or limited this work. The only price data in our dataset were ticket prices. You were provided with information about the additional operating cost of the new chair lift, but what other cost information would be useful? Big Mountain was already fairly high on some of the league charts of facilities offered, but why was its modeled price so much higher than its current price? Would this mismatch come as a surprise to the business executives? How would you find out? Assuming the business leaders felt this model was useful, how would the business make use of it? Would you expect them to come to you every time they wanted to test a new combination of parameters in a scenario? We hope you would have better things to do, so how might this model be made available for business analysts to use and explore?

**A: 2** Your answer here
