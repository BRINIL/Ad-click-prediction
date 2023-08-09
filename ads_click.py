import numpy as np 
import pandas as pd 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


data = pd.read_csv("C:\\Users\\Brinil john\\Desktop\\Resume update\\Model deployment\\FastAPI\\Social_Network_Ads.csv")

data = data.drop('User ID' , axis =1)

Sex  = pd.get_dummies(data['Gender'] , drop_first = True)
data['Sex'] = Sex
data = data.drop('Gender' , axis =1)


x = data[['Age','EstimatedSalary','Sex']]
y = data['Purchased']

npX = np.array(x).copy()
npy = np.array(y).copy()

X_train  ,X_test , y_train , y_test = train_test_split( x , y , test_size = 0.3 , random_state = 50)
clf_nb = GaussianNB()
model = clf_nb.fit(X_train,y_train)
'''
age= 51
gender= 1
EstimatedSalary= 23000
val=[age,EstimatedSalary,gender]

print(model.predict([val])[0])
'''

with open('model_pkl.pkl', 'wb') as files:
    pickle.dump(model, files)