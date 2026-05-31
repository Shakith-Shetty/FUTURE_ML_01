import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

from google.colab import files
uploaded = files.upload()

df = pd.read_csv("superstoredataset.csv", encoding='latin1')

df.head()
df.info()

df.isnull().sum()

df.duplicated().sum()
df.drop_duplicates(inplace=True)

df['Order Date'] = pd.to_datetime(
    df['Order Date'],
    errors='coerce'
)

df['Ship Date'] = pd.to_datetime(
    df['Ship Date'],
    errors='coerce'
)

df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Day'] = df['Order Date'].dt.day
df['Quarter'] = df['Order Date'].dt.quarter
df['Weekday'] = df['Order Date'].dt.dayofweek

monthly_sales = df.groupby('Month')['Sales'].sum()

plt.figure(figsize=(10,5))
monthly_sales.plot(marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(x='Region',y='Sales',data=df)
plt.title('Region Wise Sales')
plt.show()

category_sales = df.groupby('Category')['Sales'].sum()

category_sales.plot(kind='bar')
plt.title('Category Wise Sales')
plt.show()

sns.histplot(df['Profit'],bins=30)
plt.title("Profit Distribution")
plt.show()

top_states = df.groupby('State')['Sales'].sum().sort_values(ascending=False).head(10)

top_states.plot(kind='bar')
plt.title('Top 10 States by Sales')
plt.show()

daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()
daily_sales['Year'] = daily_sales['Order Date'].dt.year
daily_sales['Month'] = daily_sales['Order Date'].dt.month
daily_sales['Day'] = daily_sales['Order Date'].dt.day

X = daily_sales[['Year','Month','Day']]
y = daily_sales['Sales']

X_train,X_test,y_train,y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)

print("MAE :",mae)
print("RMSE :",rmse)
print("R2 :",r2)

future = pd.DataFrame({
    'Year':[2026,2026,2026,2026,2026,2026],
    'Month':[1,2,3,4,5,6],
    'Day':[1,1,1,1,1,1]
})

future_sales = model.predict(future)

future['Forecast Sales'] = future_sales

print(future)

plt.figure(figsize=(10,5))

plt.plot(
    future['Month'],
    future['Forecast Sales'],
    marker='o'
)

plt.title('Future Sales Forecast')
plt.xlabel('Month')
plt.ylabel('Forecast Sales')
plt.grid(True)

plt.show()

