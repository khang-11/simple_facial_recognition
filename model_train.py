from numpy import reshape
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 
import pickle

data = load('faces.npz')
x, y = data['arr_0'], data['arr_1']

x = x.squeeze()

in_encoder = Normalizer(norm='l2')
x = in_encoder.transform(x)

model = SVC(kernel='linear', probability=True)
# model = KNeighborsClassifier(n_neighbors = 1)
model.fit(x, y)

pickle.dump(model, open('model.sav', 'wb'))
