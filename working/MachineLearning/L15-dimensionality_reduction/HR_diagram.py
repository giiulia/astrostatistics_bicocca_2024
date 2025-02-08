import urllib.request
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

urllib.request.urlretrieve("https://raw.githubusercontent.com/nshaud/ml_for_astro/main/stars.csv", "stars.csv")
df_stars = pd.read_csv("stars.csv") #Read a comma-separated values (csv) file into DataFrame.

print(df_stars)

le = LabelEncoder() # Assign unique integers from 0 to 6 to each star type
df_stars['Star type'] = le.fit_transform(df_stars['Star type'])

labels = le.inverse_transform(df_stars['Star type']) #Array of all the 240 labels in order
class_names = le.classes_ #Extraxt only the 6 different classes

print(df_stars)

fig = plt.figure(figsize=(8, 8))
sns.scatterplot(data=df_stars, x='Temperature (K)', y='Luminosity(L/Lo)', hue = labels) #plot data from pandas

plt.xscale('log')
plt.yscale('log')
plt.xticks([5000, 10000, 50000])
plt.xlim(5e4, 1.5e3)
plt.show()

columns = pd.read_csv("stars.csv", nrows = 1).select_dtypes("number").columns #select only columns with numbers
#or, do: df_stars_data = df_stars.drop(columns=["Star color", "Star type", "Spectral Class"])
df_stars = pd.read_csv("stars.csv", usecols = columns) #Read a comma-separated values (csv) file into DataFrame; only numbers

X = np.array(df_stars) #foundamental step

#Apply PCA, scatterplot on first 2 pc
pca = PCA(n_components = 2) #as said in the assignment
x_new = pca.fit_transform(X) #important use of X, with df_stars doesn't work

fig = plt.figure(figsize=(8, 8))
sns.scatterplot(x = x_new[:,0], y = x_new[:,1], hue = labels)

plt.show()

#use StandardScaler because the last one was unsatisfactory
print(df_stars)
scaler = StandardScaler()
X = scaler.fit_transform(df_stars)

pca = PCA(n_components = 4) #4 is the default
x_new = pca.fit_transform(X)
evals = pca.explained_variance_ratio_ 


fig = plt.figure(figsize=(8, 8))
sns.scatterplot(x = x_new[:,0], y = x_new[:,1], hue = labels)

plt.show()
s = 0
for idx, r in enumerate(evals):
    s += r
    print(f"Component {idx+1} explains {100*r:.1f}% of the variance (cumulative = {100*s:.1f})")