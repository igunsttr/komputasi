import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score  

import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
iris = load_iris()
X = iris.data[:, :2]  # Ambil 2 fitur pertama untuk visualisasi
y = iris.target

# Bagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat model KNN
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)

# Prediksi
y_pred = knn.predict(X_test)

# Evaluasi
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi:", accuracy)

# Visualisasi data pelatihan
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=iris.target_names[y], style=iris.target_names[y])
plt.title('Visualisasi Data Iris')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()