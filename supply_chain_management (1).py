# -*- coding: utf-8 -*-
"""supply chain management.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PHWhoAjoDMhQ7iNyZ4C03zFKgvND_rXk
"""

import numpy as np # linear algebra
import pandas as pd

import numpy as np # linear algebra
import pandas as pd # Import the pandas library and alias it as 'pd'

df = pd.read_csv("/content/FMCG_data.csv.zip")

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline
import plotly.express as px
import plotly.subplots as sp
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import seaborn as sns
sns.set_style('whitegrid')

df

df.info()

wh = df.dropna(subset=['wh_est_year'])
pd.options.mode.copy_on_write = True

df['wh_est_year'] = wh['wh_est_year'].astype(int)
df.wh_est_year.unique()

df.shape
df.isna().sum()

df.describe()
df.select_dtypes(include='object').describe()

df.WH_capacity_size.mode()

df.WH_capacity_size.value_counts().reset_index()

df.Location_type.value_counts().reset_index()

df.groupby('zone')['retail_shop_num'].sum().reset_index()

df.groupby('WH_regional_zone')['workers_num'].mean().reset_index()

df.groupby('zone')['workers_num'].mean()

df.groupby('WH_capacity_size')['workers_num'].mean()

df['workers_num'] = wh['workers_num'].fillna(wh.groupby('WH_regional_zone')['workers_num'].transform('mean'))

df.workers_num.isna().sum()

df.electric_supply.value_counts()

total_warehouses = len(wh)

warehouses_with_electric_supply = df['electric_supply'].sum()

percentage_with_electric_supply = (warehouses_with_electric_supply / total_warehouses) * 100

print(f"Percentage of warehouses with electric supply: {percentage_with_electric_supply:.2f}%")

df.groupby('zone')['dist_from_hub'].mean().reset_index()

df.groupby(['zone', 'WH_regional_zone'])['dist_from_hub'].mean()

# Total number of warehouses
total_warehouses = len(wh)

# Total number of warehouses with storage issues in the last 3 months
total_issues = wh['storage_issue_reported_l3m'].sum()

# Count of warehouses with storage issues by zone and regional zone
issues_by_zone = wh[wh['storage_issue_reported_l3m'] > 0].groupby('zone').size().reset_index(name='issues_count')
issues_by_regional_zone = wh[wh['storage_issue_reported_l3m'] > 0].groupby('WH_regional_zone').size().reset_index(name='issues_count')

print(f"Total warehouses: {total_warehouses}")
print(f"Total warehouses with storage issues: {total_issues}")

print("Issues by Zone:")
print(issues_by_zone)

print("\nIssues by Regional Zone:")
print(issues_by_regional_zone)

df.groupby('WH_regional_zone')['num_refill_req_l3m'].sum().sort_values(ascending=False).head(3)

df.groupby(['zone','WH_regional_zone'])['govt_check_l3m'].mean()

df.approved_wh_govt_certificate.value_counts()

correlation = wh['workers_num'].corr(wh['storage_issue_reported_l3m'])
print(f"The correlation between the number of workers and the number of reported storage issues is: {correlation}")

df.pivot_table(index='WH_capacity_size', columns='num_refill_req_l3m', values='Ware_house_ID', aggfunc='count')

df.groupby('zone')['transport_issue_l1y'].mean().round(decimals=2)

filtered_wh = wh[wh['temp_reg_mach']==1]
avg_weight = filtered_wh['product_wg_ton'].mean().round(decimals=2)
print('\nAverage product weight per ton for warehouses with temperature regulation machinery:', avg_weight)

df.groupby('govt_check_l3m')['storage_issue_reported_l3m'].sum().sort_values(ascending=False).head().reset_index()

df.groupby('Location_type')['workers_num'].mean().round(decimals=2)

cor = wh['transport_issue_l1y'].corr(wh['dist_from_hub'])
print(cor)

fig = px.scatter(wh, x='transport_issue_l1y', y='dist_from_hub', size='dist_from_hub')
fig.show()

df.pivot_table(index='Competitor_in_mkt', columns='num_refill_req_l3m', values='Ware_house_ID', aggfunc='count')

df.groupby('Competitor_in_mkt')['num_refill_req_l3m'].sum()

df.pivot_table(index='storage_issue_reported_l3m', columns='approved_wh_govt_certificate', values='Ware_house_ID', aggfunc='count')

df.groupby('approved_wh_govt_certificate')['storage_issue_reported_l3m'].sum()

plt.figure(figsize=(14,8), dpi=80, frameon=True)
df.plot(kind='line', x='wh_est_year', y='wh_breakdown_l3m')
plt.show()

df.columns

wh.isna().sum()

wh.to_csv('Amazon_FMCG.csv')

"""1. Setting Up Agents with LangChain and OpenAGI"""

from langchain.agents import initialize_agent, Tool

from exa_py import Exa

exa = Exa(api_key="a04814e6-d805-4006-9235-49065cbd3286")

result = exa.search(
  "blog post about Rust",
  type="auto"
)

pip install tavily-python

from tavily import TavilyClient

# Step 1. Instantiating your TavilyClient
tavily_client = TavilyClient(api_key="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")

# Step 2. Executing a simple search query
response = tavily_client.search("Who is Leo Messi?")

# Step 3. That's it! You've done a Tavily Search!
print(response)

# Function to perform Tavily search and print results
def search_tavily(query):
  response = tavily_client.search(query)
  print(response)

# Example usage before your other code:
search_tavily("FMCG industry trends in India")

from langchain.agents import initialize_agent, Tool
from langchain.tools import BaseTool
# You likely need to install these if you haven't already
!pip install tavily-python exa-py -q

from tavily import TavilyClient
from exa_py import Exa

# Custom Tavily Tool
class TavilySearchTool(BaseTool):
    name: str = "Tavily Search"  # Added type annotation for 'name'
    description: str = "A tool for searching the web using Tavily API"  # Added type annotation for 'description'

    def __init__(self, api_key):
        self.tavily_client = TavilyClient(api_key=api_key)

    def _run(self, query: str):
        response = self.tavily_client.search(query)
        return response  # You might need to format this further

    async def _arun(self, query: str):
        raise

"""2. Demand Forecasting Model Using Prophet"""

from prophet import Prophet
import pandas as pd

# Load data
data = pd.read_csv("/content/Amazon_FMCG.csv")

# Data preprocessing (e.g., handling intermittent demand, outliers, seasonality)

# Rename columns (correcting the syntax error)
data.rename(columns={"govt_check_l3m": "govt_check_l3m", "storage_issue_reported_l3m": "storage_issue_reported_l3m"}, inplace=True)

# Check if 'Date' or a similar column exists, and adjust accordingly
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
elif 'wh_est_year' in data.columns:  # Assuming 'wh_est_year' represents the date
    data['Date'] = pd.to_datetime(data['wh_est_year'], format='%Y') # Assuming year only
    # If you have month/day, adjust format accordingly (e.g., '%Y-%m-%d')
else:
    raise KeyError("No suitable date column found. Please specify the correct column name.")

data = data[['Date', 'govt_check_l3m']]  # Replace 'TargetColumn' with your actual sales column
data.columns = ['ds', 'y']

# Train forecasting model
model = Prophet()
model.fit(data)

# Generate future predictions
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot predictions
model.plot(forecast)

"""Deployment with Streamlit"""
