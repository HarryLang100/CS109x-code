"""
Created on Sat Nov 18 17:30:19 2023

"""

# Import statements
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()

# Read data
filename = "C:\\Users\\harry\\OneDrive\\Learning\\Leiter resources\Credit.csv"
df = pd.read_csv(filename)

# Extract/create the required columns
y = df["Balance"]
x = pd.DataFrame(columns=["Sex_indicator"])
x = pd.DataFrame(df["Sex"].apply(lambda x: 1 if x == 'Female' else 0))

# Initialise and fit the model
lreg = LinearRegression()
lreg.fit(x, y)

# What are the coefficients?
intercept = lreg.intercept_
gradient = float(lreg.coef_)
text = f"Intercept (beta_0): {intercept:.0f}, gradient (beta_1): {gradient:.0f}"

# Make a plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y, "k+", label="Data")
ax.plot(x, lreg.predict(x), "bx-", label="Predictions")
plt.xlabel("Sex indicator - 1 is Female")
plt.ylabel("Balance")
plt.legend()
plt.text(0, 0, text)
plt.show()


