#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Importing essential modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')


# In[9]:


#Reading Dataset
df = pd.read_csv('[DA] Pre-Screen Test-Data.csv')
df.head() #first 5 rows


# In[12]:


#Analyzing the Data
df.shape #the number of rows (observations) and columns (features) in the dataset.


# In[14]:


#understanding the dataset
df.info()


# In[16]:


# a statistical summary of the DataFrame
df.describe()


# In[18]:


#converts the column names of the DataFrame into a Python list
df.columns.tolist()


# In[98]:


#Checking Missing Values
print("\nMissing values check:")
print(df.isnull().sum())


# In[99]:


#Checking for the duplicate values
df.nunique()


# In[100]:


# Convert columns to appropriate types
df['Departure_Date'] = pd.to_datetime(df['Departure_Date']) 
#print(df['Departure_Date'].dtype)

print(df.dtypes)


# In[101]:


#Define categorical variables based on domain understanding
categorical_features = [
    'Gender', 'Nationality', 'Loyalty_Status', 'Booking_Channel',
    'Route', 'Scheduled_Time', 'Ticket_Class', 
    'Seat_Selection', 'Pre_Ordered_Meal'
]
# Cast categorical variables to 'category' dtype to improve memory efficiency
df[categorical_features] = df[categorical_features].apply(lambda x: x.astype('category'))

print(df[categorical_features].dtypes) #result


# In[54]:


# Check for duplicate nationality
df.loc[df.duplicated(subset=['Route'])]


# In[102]:


# Checking an example duplicate
print(df.query('Route == "BKK-SIN"'))


# In[103]:


# Create a categorical feature to classify flight delay performance
df['Delay_Status'] = pd.cut(
    df['Delay_Minutes'],
    bins=[-1, 0, 15, 60, np.inf],
    labels=['On Time', 'Slight Delay', 'Moderate Delay', 'Severe Delay']
)

print(df['Delay_Status'].value_counts())


# In[104]:


# Calculate the share of add-on revenue (seat + meal) in total revenue
df['AddOn_Revenue_Share'] = (
    (df['Seat_Extra_Charge_USD'] + df['Meal_Revenue_USD']) / df['Total_Revenue_USD']
).round(2)

print(df['AddOn_Revenue_Share'].describe())


# In[105]:


# Flag loyal customers (Silver, Gold, Platinum) for segmentation analysis
df['Is_Loyal'] = df['Loyalty_Status'].isin(['Silver', 'Gold', 'Platinum']).astype(int)

print(df['Is_Loyal'].value_counts())

# Extract flight origin and destination from route code
df[['Origin', 'Destination']] = df['Route'].str.split('-', expand=True)

print("\nRoute Components")
print(df[['Route', 'Origin', 'Destination']].head(10))

# Combine date and scheduled time into a unified departure segment
df['Departure_Segment'] = df['Departure_Date'].astype(str) + ' - ' + df['Scheduled_Time'].astype(str)

print("\nDeparture Segment Preview")
print(df['Departure_Segment'].head(10))


# In[106]:


# Group by Scheduled_Time
time_stats = df.groupby('Scheduled_Time').agg(
    Avg_Delay=('Delay_Minutes','mean'),
    Cancellation_Rate=('Cancellation','mean')
).round(3)
print("By Scheduled Time:\n", time_stats)

# Group by Route
route_stats = df.groupby('Route').agg(
    Avg_Delay=('Delay_Minutes','mean'),
    Cancellation_Rate=('Cancellation','mean')
).round(3)
print("\nBy Route:\n", route_stats)



# Group by Loyalty_Status
loyalty_stats = df.groupby('Loyalty_Status').agg(
    Avg_Delay=('Delay_Minutes','mean'),
    Cancellation_Rate=('Cancellation','mean')
).round(3)
print("\nBy Loyalty Status:\n", loyalty_stats)

# Group by Booking_Channel
channel_stats = df.groupby('Booking_Channel').agg(
    Avg_Delay=('Delay_Minutes','mean'),
    Cancellation_Rate=('Cancellation','mean')
).round(3)
print("\nBy Booking Channel:\n", channel_stats)


# In[108]:


# Visualize: Average delay by Scheduled_Time
plt.figure(figsize=(6,4))
sns.barplot(x=time_stats.index, y='Avg_Delay', data=time_stats.reset_index())
plt.title('Average Delay by Scheduled Time')
plt.ylabel('Avg Delay (min)')
plt.xlabel('Scheduled Time')
plt.show()


# In[107]:


# Visualize: Cancellation rate by Loyalty Status
plt.figure(figsize=(6,4))
sns.barplot(x=loyalty_stats.index, y='Cancellation_Rate', data=loyalty_stats.reset_index())
plt.title('Cancellation Rate by Loyalty Status')
plt.ylabel('Cancellation Rate')
plt.xlabel('Loyalty Status')
plt.show()


# In[109]:


# Correlation and Relationship Visualization


# Select numerical columns for correlation analysis
numeric_cols = ['Age', 'Delay_Minutes', 'Cancellation', 'Base_Ticket_Price_USD',
                'Seat_Extra_Charge_USD', 'Meal_Revenue_USD', 'Total_Revenue_USD']

# Compute correlation matrix
corr_matrix = df[numeric_cols].corr().round(2)

# Visualize correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Key Numerical Features', fontsize=13)
plt.show()


# In[110]:


# Calculate averages by Scheduled_Time
time_metrics = (
    df.groupby('Scheduled_Time')[['Delay_Minutes', 'Cancellation']]
      .mean()
      .rename(columns={'Delay_Minutes': 'Avg_Delay', 'Cancellation': 'Cancel_Rate'})
      .reset_index()
)

# Line plot showing both metrics on twin axes
fig, ax1 = plt.subplots(figsize=(7, 4))

ax2 = ax1.twinx()
sns.lineplot(data=time_metrics, x='Scheduled_Time', y='Avg_Delay', marker='o', color='tab:blue', ax=ax1, label='Avg Delay (min)')
sns.lineplot(data=time_metrics, x='Scheduled_Time', y='Cancel_Rate', marker='o', color='tab:red', ax=ax2, label='Cancellation Rate')

ax1.set_xlabel('Scheduled Time of Day')
ax1.set_ylabel('Average Delay (minutes)', color='tab:blue')
ax2.set_ylabel('Cancellation Rate', color='tab:red')
plt.title('Average Delay and Cancellation Rate by Scheduled Time', fontsize=13)
ax1.grid(True, linestyle='--', alpha=0.6)
plt.show()


# In[111]:


# Visualization: Histogram of Loyalty Status
plt.figure(figsize=(6,4))
sns.histplot(df['Loyalty_Status'], bins=10, kde=False, color='skyblue')
plt.title('Loyalty Distribution')
plt.xlabel('Status')
plt.ylabel('Count')
plt.show()


# In[112]:


# Mean total revenue by category
for col in ['Ticket_Class', 'Loyalty_Status', 'Seat_Selection', 'Pre_Ordered_Meal']:
    rev_mean = df.groupby(col)['Total_Revenue_USD'].mean().round(2)
    print(f"\nAverage Revenue by {col}:\n", rev_mean)


# In[113]:


# Illustrate revenue by Ticket Class 

#bar plt
plt.figure(figsize=(6,4))
sns.barplot(x='Ticket_Class', y='Total_Revenue_USD', data=df, estimator=np.mean, palette='pastel')
plt.title('Average Revenue by Ticket Class')
plt.ylabel('Mean Total Revenue (USD)')
plt.show()

#box plt
plt.figure(figsize=(6,4))
sns.boxplot(x='Ticket_Class', y='Total_Revenue_USD', data=df, palette='vlag')
plt.title('Revenue Distribution by Ticket Class')
plt.ylabel('Total Revenue (USD)')
plt.show()


# In[142]:


# High-Value Customer Segmentation
# Group by Age and Loyalty Status to find the average revenue, delay, and cancellation rate.
customer_analysis = df.groupby(['Age', 'Loyalty_Status']).agg(
    Avg_Revenue=('Total_Revenue_USD','mean'),
    Avg_Delay=('Delay_Minutes','mean'),
    Total_Flights=('Flight_ID', 'count'),
    Cancellations=('Cancellation','sum')
).reset_index()

customer_analysis['Cancel_Rate'] = (customer_analysis['Cancellations'] / customer_analysis['Total_Flights']) * 100


# Sort by Avg_Revenue to find top segments
top_segments = customer_analysis.sort_values(by='Avg_Revenue', ascending=False).head(5)

print("\nTop 5 High-Value Customer Segments by Avg.Revenue")
print(top_segments[['Age', 'Loyalty_Status', 'Avg_Revenue', 'Avg_Delay', 'Cancel_Rate']])
print("-" * 50)


# In[141]:


# Group by Booking Channel to assess financial and operational performance.
channel_analysis = df.groupby('Booking_Channel').agg(
    Total_Revenue=('Total_Revenue_USD', 'sum'),
    Avg_Delay=('Delay_Minutes', 'mean'),
    Total_Flights=('Flight_ID', 'count'),
    Cancellations=('Cancellation', 'sum')
)
channel_analysis['Cancel_Rate'] = (channel_analysis['Cancellations'] / channel_analysis['Total_Flights']) * 100

print("\nBooking Channel Performance")
print(channel_analysis[['Total_Revenue', 'Avg_Delay', 'Cancel_Rate']].sort_values(by='Total_Revenue', ascending=False))
print("-" * 50)

#figure
plt.figure(figsize=(8, 5))
sns.barplot(x=channel_analysis.index, y='Total_Revenue', data=channel_analysis, palette='viridis')
plt.title('Total Revenue by Booking Channel', fontsize=14)
plt.ylabel('Total Revenue USD', fontsize=12)
plt.xlabel('Booking Channel', fontsize=12)
plt.show()


# In[151]:


route_analysis = df.groupby('Route').agg(
    Total_Revenue=('Total_Revenue_USD', 'sum'),
    Avg_Delay=('Delay_Minutes', 'mean'),
    Total_Flights=('Flight_ID', 'count'),
    Cancellations=('Cancellation', 'sum')
)
route_analysis['Cancel_Rate'] = (route_analysis['Cancellations'] / route_analysis['Total_Flights']) * 100

top_routes = route_analysis.sort_values(by='Total_Revenue', ascending=False).head(5)

print("\nTop 5 Revenue-Generating Routes")
print(top_routes[['Total_Revenue', 'Avg_Delay', 'Cancel_Rate']])
print("-" * 50)

# Plot
plt.figure(figsize=(10, 6))

sns.barplot(
    x='Total_Revenue',
    y=top_routes.index,
    data=top_routes.reset_index(), 
    palette='rocket'
)
plt.title('Total Revenue by Top 5 Routes (Horizontal)', fontsize=14)
plt.xlabel('Total Revenue USD', fontsize=12)
plt.ylabel('Route', fontsize=12)
plt.show()


# In[ ]:





# In[ ]:




