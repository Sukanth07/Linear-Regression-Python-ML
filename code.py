#This code is primarily written on google colab
#Reading the Dataset
import pandas as pd
df = pd.read_csv("canada_per_capita_income.csv")

#Performing EDA (Exploratory Data Analysis)
print("Head of the dataset :\n",df.head())
print("\nShape :  ",df.shape)
print("\nSize :  ",df.size)
print("\nInfo :  ",df.info())
print("\nDescribe :  \n",df.describe())

#Fitting the model with respect to dataset
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(df[['year']],df['income'])

#Visualization using graph and best fit line
import matplotlib.pyplot as plt
plt.scatter(df.year,df.income,marker="+",color="red")
plt.xlabel("Year")
plt.ylabel("Income")
plt.title("Canada Per Capita Income")
plt.plot(df.year,model.predict(df[['year']]),color="blue")
plt.show()

#Predicting the income for year 2020
print("\nPrediction year (2020) :  ",model.predict([[2020]]))

# Equation of the line for predicting ---- y = mx + c
m = model.coef_
c = model.intercept_
x = 2020
y = m*x+c
print("\nPrediction year (2020) using line eqn :  ",y)

#Calculating the accuracy of the model
score = model.score(df[['year']],df['income'])
print("\nAccuracy of the model :  ",score*100)
