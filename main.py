import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

file_path = 'ca_real_estate.csv'
data = pd.read_csv(file_path).query("City == 'Toronto'")

Q1 = data['Price'].quantile(0.25)
Q3 = data['Price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['Price'] >= lower_bound) & (data['Price'] <= upper_bound)]

data['Age'] = 2024 - data['Year_Built'] 
data = pd.get_dummies(data, columns=['Province', 'Type'])

average_price_by_age_bedrooms = data.groupby(['Age', 'Bedrooms'])['Price'].mean().reset_index()

projection_years = 10
current_year = 2024
future_years = [current_year + i for i in range(projection_years + 1)]
bedroom_categories = average_price_by_age_bedrooms['Bedrooms'].unique()

plt.figure(figsize=(10, 6))

for bedroom in bedroom_categories:
    bedroom_data = average_price_by_age_bedrooms[average_price_by_age_bedrooms['Bedrooms'] == bedroom]
    bedroom_data = bedroom_data.sort_values(by='Age')
    bedroom_data['Price_Increase_Rate'] = bedroom_data['Price'].pct_change()
    annual_increase = bedroom_data['Price_Increase_Rate'].mean()

    projected_prices = []
    if not np.isnan(annual_increase):
        current_avg_price = bedroom_data['Price'].iloc[-1]
        for year in future_years:
            future_price = current_avg_price * ((1 + annual_increase) ** (year - current_year))
            projected_prices.append(future_price)
        
        plt.plot(future_years, projected_prices, marker='o', label=f"{bedroom} Bedrooms")

plt.title("Projected Real Estate Prices in GTA")
plt.xlabel("Year")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()
