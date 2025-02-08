from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import Isomap
import numpy as np
import matplotlib.pyplot as plt

digits = datasets.load_digits()
print(digits.images.shape)
print(digits.keys())
print(digits.DESCR)

my_classification = np.array([8, 9, 8, 1, 2, 6, 9, 4, 9, 4, 0, 6, 1, 7, 6, 9, 5, 4, 4, 8, 4, 0, 5, 6, 1, 7, 9, 3, 2, 4, 0, 8, 3, 9, 6, 1, 1, 2, 0, 5, 4, 4, 9, 6, 2, 6, 1, 0, 0])
print(my_classification.shape)

fig, axes = plt.subplots(7,7, figsize=(10, 10))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

np.random.seed(42)
mychoices = np.random.choice(digits.images.shape[0], 100)

for i, ax in enumerate(axes.flat):
    ax.imshow((digits.images[mychoices[i]]), cmap='binary')
    ax.text(0.05, 0.05, str(digits.target[mychoices[i]]),transform=ax.transAxes, color='green', fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()

print(digits.data.shape)

embedding = Isomap(n_components = 36)
digits_transformed = embedding.fit_transform(digits.data[:100])
digits_transformed.shape

print(digits_transformed.shape)

digits_train, digits_test = train_test_split(digits_transformed.data, test_size = 0.2)
clf = LogisticRegression(random_state = 0, solver = 'sag').fit(digits_train)
pred = clf.predict(digits_train[:, :])

print(accuracy_score(pred, digits_train))

pred_new = clf.predict(digits_test[:, :])
print(accuracy_score(pred_new, digits_test))
