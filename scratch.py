import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filename = "C:\\Users\\harry\\OneDrive\\Learning\\Leiter resources\\Credit.csv"
df = pd.read_csv(filename)

corrMatrix = df.drop(["y"], axis=1).corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()