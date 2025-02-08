import requests
from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt

# Download file
r = requests.get('https://user-web.icecube.wisc.edu/~grbweb_public/Summary_table.txt') #web request
with open("Summary_table.txt", 'wb') as f:
    f.write(r.content)

# Read content
gb = np.loadtxt("Summary_table.txt", dtype='str',unpack='True')

print(gb.shape)

fluence = np.array(gb[7], dtype=float)
print(fluence)
redshift = np.array(gb[11], dtype=float)
print(redshift)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot()


ax.scatter(gb[7, :100], gb[11, :100], marker=".", color='black')
ax.set_xlabel('fluence')
ax.set_ylabel('redshift')
plt.show()

clf = KMeans(n_clusters=2,n_init='auto') #Try 2 clusters to start with
clf.fit(gb)
centers = clf.cluster_centers_ #location of the clusters
labels = clf.predict(gb) #labels for each of the points

# plot the data color-coded by cluster id
colors = ['C1', 'C0', 'C2']
for ii in range(3):
    plt.scatter(gb[labels==ii,0], gb[labels==ii,1], color=colors[ii],alpha=0.5)

# To get some information on these try:
# KMeans?
# help(clf)
plt.title('Clustering output');
plt.show()
