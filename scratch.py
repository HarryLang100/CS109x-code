# Import statements
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
filename = "C:\\Users\\harry\\OneDrive\\Learning\\Leiter resources\\colinearity.csv"
df = pd.read_csv(filename)

# Make the heatmap
corrMatrix = df.drop("y", axis=1).corr()
sns.heatmap(corrMatrix, annot=True)
plt.title("Correlation Matrix Heatmap")
plt.show()