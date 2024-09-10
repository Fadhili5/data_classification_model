#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

#Load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

#split the dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#Train a classification model using a Support vector machine (SVM) Algorithm
model = SVC(kernel='linear',random_state=44)
model.fit(X_train, y_train)

#Test the model on test data and report accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))

#Visualize classification results using simple plot
#Reduce data to 2 components for easy visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

#Split PCA-transformed data into train and test data
X_pca_train,X_pca_test = train_test_split(X_pca,test_size=0.3,random_state=42)

#Plot
plt.scatter(X_pca[0,0],X_pca[0,1], color='blue',label=iris.target_names[0])
plt.scatter(X_pca[1,0],X_pca[1,1], color='red',label=iris.target_names[1])
plt.scatter(X_pca[2,0],X_pca[2,1], color='green',label=iris.target_names[2])
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.title('The Iris Dataset PCA')
plt.show()

#Test model using the sample data points
sample_data = np.array([7.0, 3.2, 4.7, 1.4])
predicted_class = model.predict(sample_data.reshape(1,-1))
predicted_species = iris.target_names[predicted_class][0]

#Output resultant prediction
print(f'Predicted class for the sample data {sample_data} is {predicted_species}')