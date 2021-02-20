#!/usr/bin/env python
# coding: utf-8

# In[343]:


import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
from scipy.stats import stats
#scipi is a library for statistical tests and visualization
import random
#enables us to generate random numbers
from scipy.stats import normaltest   
import pylab as p


# In[344]:


#i didn't understand how to do this "Create a variable called google, and store in it the path of the csv file that contains your google dataset.
Google=pd.read_csv("googleplaystore.csv")
Google.head(3)


# In[348]:


Apple=pd.read_csv("AppleStore.csv")
Apple.head(3)


# In[467]:


# Subset our DataFrame object Google by selecting just the variables ['Category', 'Rating', 'Reviews', 'Price']
Google=Google[['Category', 'Rating', 'Reviews', 'Price']]
Google.head(3)


# In[351]:


Apple=Apple[['prime_genre', 'user_rating', 'rating_count_tot', 'price']]
Apple.head(3)


# In[352]:


#check out the data types within our Apple dataframe.
Apple.dtypes


# In[353]:


#the data types of our Google dataframe
Google.dtypes


# In[354]:


#unique() pandas method on the Price column
Google['Price'].unique()


# In[355]:


#check which data points have the value 'Everyone' for the 'Price' column
# Subset the Google dataframe on the price column. 
# To be sure: you want to pick out just those rows whose value for the 'Price' column is just 'Everyone'. 
Google[Google['Price']=='Everyone']


# In[356]:


# eliminate that row. 
# Subset Google dataframe to pick out just those rows whose value for the 'Price' column is NOT 'Everyone'. 
# Reassign that subset to the Google variable.
# Google['Price']= Google[Google['Price'] !='Everyone']
# Check again the unique values of Google
Google=Google[Google['Price'] !='Everyone']
Google['Price'].unique()


# In[357]:


Google['Price'] = Google['Price'].str.replace('$','')
Google['Price'].unique()


# In[360]:


# i. Make the values in the nosymb variable numeric using the to_numeric() pandas method.
# ii. Assign this new set of numeric, dollar-sign-less values to Google['Price']. 
# You can do this in one line if you wish.
Google['Price'] = pd.to_numeric(Google['Price'])


# In[361]:


Google.dtypes


# In[382]:


# Convert the 'Reviews' column to a numeric data type.
Google['Reviews'] = pd.to_numeric(Google['Reviews'])


# In[383]:


Google.dtypes


# In[474]:


# Create a column called 'platform' in both the Apple and Google dataframes
Apple['platform']="apple"
Google['platform']="google"
#rename user_rating as rating
Apple=Apple.rename({'user_rating':'Rating','price':'Price'},axis='columns')
# Let's use the append() method to append Apple to Google. 
# Make Apple the first parameter of append(), and make the second parameter just: ignore_index = True.
df=Google.append(Apple, ignore_index= True)


# In[476]:


df.columns.tolist()


# In[477]:


old_names=Apple.columns.tolist()
new_names=Google.columns.tolist()
#Use the rename() DataFrame method to change the columns names.
# In the columns parameter of the rename() method, use this construction: dict(zip(old_names,new_names)).
df.rename(columns=dict(zip(old_names,new_names)))


# In[478]:


# Using the sample() method with the number 12 passed to it, check 12 random points of your dataset.
df.sample(12)


# In[479]:


df.dropna(how="all")


# In[480]:


# Subset your df to pick out just those rows whose value for 'Reviews' is equal to 0. 
# Do a count() on the result. 
#df.groupby(Google).count()
df['Reviews'].describe()


# In[481]:


# Eliminate the points that have 0 reviews.
# An elegant way to do this is to assign df the result of picking out just those rows in df whose value for 'Reviews' is NOT 0.
df=df[df["Reviews"]!=0]
df


# In[482]:


# To summarize analytically, let's use the groupby() method on our df.
# For its parameters, let's assign its 'by' parameter 'platform', and then make sure we're seeing 'Rating' too. 
# Finally, call describe() on the result. We can do this in one line, but this isn't necessary. 
df.describe()
df.groupby('platform')['Rating'].describe()


# In[483]:


# Call the boxplot() method on our df.
# Set the parameters: by = 'platform' and column = ['Rating'].

df.boxplot(by='platform', column =['Rating'], grid=False, rot=45, fontsize=10)


# In[484]:


#Stage 3 - Modelling


# In[485]:


# Create a subset of the column 'Rating' by the different platforms.
# Call the subsets 'apple' and 'google' 
apple = df[df['platform'] == 'apple']['user_rating']
google = df[df['platform']=='google']['Rating']


# In[486]:


# Using the stats.normaltest() method, get an indication of whether the apple data are normally distributed
# Save the result in a variable called apple_normal, and print it out
#print(apple_normal)
x1 = np.linspace( -5, 12, 1000 ) 
y1 = 1./(np.sqrt(2.*np.pi)) * np.exp( -.5*(x1)**2  ) 
  
p.plot(x1, y1, '.') 

apple_normal=stats.normaltest(apple)


# In[487]:


# Do the same with the google data. 
# Save the result in a variable called google_normal
x1 = np.linspace( -5, 12, 1000 ) 
y1 = 1./(np.sqrt(2.*np.pi)) * np.exp( -.5*(x1)**2  ) 
  
p.plot(x1, y1, '.') 
google_normal = stats.normaltest(google)


# In[488]:


# Create a histogram of the apple reviews distribution
# You'll use the plt.hist() method here, and pass your apple data to it
histoApple = plt.hist(apple)


# In[489]:


# Create a histogram of the google data
histoGoogle =plt.hist(google)


# In[490]:


# Create a column called `Permutation1`, and assign to it the result of permuting (shuffling) the Rating column
# This assignment will use our numpy object's random.permutation() method
df['Permutation1'] = np.random.permutation(df['Rating'])


# In[491]:


# Lets compare with the previous analytical summary:
df.groupby(by='platform')['Permutation1'].describe()


# In[492]:


# First, make a list called difference.
list_difference=list()


# In[493]:


# make a for loop that does the following 10,000 times:
# 1. makes a permutation of the 'Rating'
df['Permutation1'] = np.random.permutation(df['Rating'])
# 2. calculates the difference in the mean rating for apple and the mean rating for google.
list_difference.append(np.mean(permutation[df['platform']=='apple']) - np.mean(permutation[df['platform']=='google']))

for i in range(10000):
    permutation2=np.random.permutation(df['Rating'])
    list_difference.append(np.mean(permutation2[df['platform']=='apple']) - np.mean(permutation2[df['platform']=='google']))


# In[494]:


# Make a variable called 'histo', and assign to it the result of plotting a histogram of the difference list. 
histo = plt.hist(list_difference)


# In[ ]:


'''
What do we know? 

Recall: The p-value of our observed data is just the proportion of the data given the null that's at least as extreme as that observed data.

As a result, we're going to count how many of the differences in our difference list are at least as extreme as our observed difference.

If less than or equal to 5% of them are, then we will reject the Null. 
'''


# In[ ]:


#make a variable called obs_difference, and assign it the result of the mean of our 'apple' variable and the mean of our 'google' variable
obs_difference = np.mean(apple)-np.mean(google)
# Make this difference absolute with the built-in abs() function
obs_difference=abs(obs_difference)

# Print out this value; it should be 0.1420605474512291. 
#this line isn't working
#print("obs_difference=",obs_difference())


# In[ ]:


positiveExtremes = []
negativeExtremes = []
for i in range(len(difference)):
    if (difference[i] >= obs_difference):
        positiveExtremes.append(difference[i])
    elif (difference[i] <= -obs_difference):
        negativeExtremes.append(difference[i])

print(len(positiveExtremes))
print(len(negativeExtremes))


# In[ ]:


#FINISHED FINALLY

