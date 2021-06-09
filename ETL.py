import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import os

# Input file & Output file
file_zomato = 'Zomato.csv'
file_cleaned = 'Stage_Clean.csv'

# Cleaning and transforming
###########################
# Read the input file
df_zomato = pd.read_csv(file_zomato)
print("Shape of the Initial dataset:   ", df_zomato.shape)

# (Uniqueness) Sorting based on Restaurant ID
df_zomato= df_zomato.sort_values(by=['Restaurant_ID'], ascending = True, na_position = 'last').drop_duplicates(['Restaurant_ID'],keep = 'first')
print("Shape of the dataset after Uniqueness quality check:   ", df_zomato.shape)

# (Value Completeness) Dropping records for Latitude and Cuisines is Null
df_zomato.dropna(subset = ['Latitude', 'Cuisines'], inplace = True)
# (Value Completeness) Fill some missing values based on ID
mask = (df_zomato['Restaurant_ID'] == 6300010) | (df_zomato['Restaurant_ID'] == 6308205)
df_zomato.loc[mask, 'Country_Code'] = 162
mask = (df_zomato['Restaurant_ID'] == 6600060)
df_zomato.loc[mask, 'Country_Code'] = 30
mask = (df_zomato['Restaurant_ID'] == 16613507)
df_zomato.loc[mask, 'Country_Code'] = 14
print("Shape of the dataset after Value Completeness quality check:   ", df_zomato.shape)

# (Redundancy Completeness) Drop an unwanted column
df_zomato.drop(['Locality Verbose'], axis = 1, inplace = True)
print("Shape of the dataset after Redundancy quality check:   ", df_zomato.shape)

# (Consistency) Normalising data for Rating text column
df_zomato['Rating text'] = df_zomato['Rating text'].str.replace('V Good','Very Good')
df_zomato['Rating text'] = df_zomato['Rating text'].str.replace('avg','Average')
print("Shape of the dataset after Consistency quality check:   ", df_zomato.shape)

# To fill up the remaining missing values: Creating a look-up table for Currency and Country Code based on City
df_Currency_Code = df_zomato.loc[ : , ['Country_Code' ,'City', 'Currency']]
# Drop all rows with null values
df_Currency_Code.dropna(axis = 'rows', inplace = True)
# Keep only unique values
df_Currency_Code.drop_duplicates(['City'], keep ='first', inplace = True)

lookup_table_Country = dict(zip(df_Currency_Code.City, df_Currency_Code.Country_Code))
lookup_table_Currency = dict(zip(df_Currency_Code.City, df_Currency_Code.Currency))
#print('Lookup table')
#print(lookup_table_Country)
#print(lookup_table_Currency)

# (Conformity) Hence fill up the missing values in columns "Currency" & "Country Code" using lookup table
df_zomato.Country_Code = df_zomato.Country_Code.fillna(df_zomato.City.map(lookup_table_Country))
df_zomato.Currency = df_zomato.Currency.fillna(df_zomato.City.map(lookup_table_Currency))

#Remove any remaining Nan value containing rows
df_zomato.dropna(axis = 'rows', inplace = True)
print('Shape of dataset after filling missing values using lookup-table: ',df_zomato.shape)

# Split column based on delimiter "," for the column: Cuisines
df_split = df_zomato["Cuisines"].str.split(';', n = 5, expand = True)
# Adding to df_zomato
df_zomato["Cuisine1"]= df_split[0]
df_zomato["Cuisine2"]= df_split[1]
df_zomato["Cuisine3"]= df_split[2]
df_zomato["Cuisine4"]= df_split[3]
df_zomato["Cuisine5"]= df_split[4]
df_zomato["Cuisine6"]= df_split[5]

print("Shape of the dataset after transformation: ",df_zomato.shape)
df_zomato.to_csv(file_cleaned, index = False)
print('############################################################')

# Creating star schema
######################
df_preprocessed = pd.read_csv("Stage_Clean.csv", header = 0)

# Table 1: DF for Location Dimension
df_star_Location = df_preprocessed.loc[ :, ['Address', 'Country_Code', 'City', 'Locality', 'Longitude', 'Latitude']]
print('Shape of the Location Dimension table:', df_star_Location.shape)
# Table 2: DF for Restaurant Dimension
df_star_Restaurant = df_preprocessed.loc[ :, ['Restaurant_ID', 'Restaurant Name', 'Cuisines', 'Cuisine1', 'Cuisine2', 'Cuisine3', 'Cuisine4', 'Cuisine5', 'Cuisine6', 'Average Cost for two', 'Price range']] 
print('Shape of the Restaurant Dimension table:', df_star_Restaurant.shape)
# Table 3: DF for Restaurant Rating Facts
df_star_RestRating = df_preprocessed.loc[ :, ['Restaurant_ID', 'Address', 'Aggregate rating', 'Rating color', 'Rating text', 'Votes']]
print('Shape of the Restaurant rating Facts table:', df_star_RestRating.shape)
# Table 4: DF for Rating Dimension
df_star_Rating = df_preprocessed.loc[ :, ['Aggregate rating', 'Rating color', 'Rating text']]
df_star_Rating = df_star_Rating.drop_duplicates(['Aggregate rating'],keep = 'first')
print('Shape of the Rating Dimension table:', df_star_Rating.shape)

### Question 1: Display the distribution of different types of Cuisines across Restaurants ###
print('Question 1: Display the distribution of different types of Cuisines across Restaurants')
df_Cuisine = df_preprocessed.loc[ :, ['Cuisine1']]
df_Cuisine2 = df_preprocessed.loc[ :, ['Cuisine2']]
df_Cuisine2.columns = ['Cuisine1']
df_Cuisine3 = df_preprocessed.loc[ :, ['Cuisine3']]
df_Cuisine3.columns = ['Cuisine1']
df_Cuisine4 = df_preprocessed.loc[ :, ['Cuisine4']]
df_Cuisine4.columns = ['Cuisine1']
df_Cuisine5 = df_preprocessed.loc[ :, ['Cuisine5']]
df_Cuisine5.columns = ['Cuisine1']
df_Cuisine6 = df_preprocessed.loc[ :, ['Cuisine6']]
df_Cuisine6.columns = ['Cuisine1']

df_Cuisine= df_Cuisine.append(df_Cuisine2, ignore_index=True)
df_Cuisine= df_Cuisine.append(df_Cuisine3, ignore_index=True)
df_Cuisine= df_Cuisine.append(df_Cuisine4, ignore_index=True)
df_Cuisine= df_Cuisine.append(df_Cuisine5, ignore_index=True)
df_Cuisine= df_Cuisine.append(df_Cuisine6, ignore_index=True)

df_Cuisine.dropna(axis = 'rows', inplace = True)

prob = df_Cuisine.value_counts(normalize=True)
threshold = 0.01
mask = prob > threshold
tail_prob = prob.loc[~mask].sum()
prob = prob.loc[mask]
prob['other'] = tail_prob
prob.plot(kind='bar')
plt.title('Frequency of Occurences of different Cuisines')
plt.xlabel('Cuisines')
plt.ylabel('Frequency %')
plt.xticks(rotation=25)
plt.show()

### Question 2: Show the distribution of the restaurants across the world along with their Aggregate rating and votes ###
print('Question 2: Show the distribution of the restaurants across the world along with their Aggregate rating and votes')
fig = go.Figure(px.scatter_geo(df_preprocessed, lat='Latitude', lon='Longitude', hover_name='Restaurant Name', size='Votes', color='Aggregate rating', color_continuous_scale=['#00b359','#ff3333'], opacity=1, hover_data=['City','Aggregate rating', 'Price range'])) #, scope='asia'
fig.update_geos(
    visible=False, resolution=50,
    showcountries=True, countrycolor="RebeccaPurple")
fig.update_layout(title = 'Zomato customers across the world (Hover for Rating and votes)', title_x=0.5)
fig.show()
if not os.path.exists("images"):
    os.mkdir("images")
fig.write_image("images/fig1.png")

### Question 3: Predict the aggregate rating for a restaurant provided its Country code, price range and votes ####
print('Question 3: Predict the aggregate rating for a restaurant provided its Country code, price range and votes')
df_preprocessed['Aggregate rating'] = df_preprocessed['Aggregate rating']*10

first_500 = df_preprocessed[500:]
features = first_500[['Country_Code', 'Price range', 'Votes']].values
labels = first_500[['Aggregate rating']].values
train, test, train_labels, test_labels = train_test_split(features,labels,test_size=0.33,random_state=42)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(train, train_labels.ravel())
Y_pred = decision_tree.predict(test)
df1= pd.DataFrame(test, columns= ['Country_Code', 'Price range', 'Votes'])
df1['Yhat (Rating)'] = Y_pred/10
print(df1)
df1.to_csv('Out.csv', index = False)
acc_decision_tree = round(decision_tree.score(train, train_labels) * 100, 2)
print('Accuracy for Decision tree',acc_decision_tree)
