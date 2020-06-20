import numpy as np
X = np.array([[1,1,1,1],[1,1,1,2],[1,1,2,1],[1,1,2,2],[1,2,1,1],
[1,2,1,2],[1,2,2,1],[1,2,2,2],[2,1,1,1],[2,1,1,2],[2,1,2,1],
[2,1,2,2],[2,2,1,1],[2,2,1,2],[2,2,2,1],[2,2,2,2],[3,1,1,1],
[3,1,1,2],[3,1,2,1]])
Y = np.array([3,2,3,1,3,2,3,1,3,2,3,1,3,2,3,3,3,3,3])
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X,Y)
print("ID = 20")
print(clf.predict([[3,1,2,2]]))
print("ID = 21")
print(clf.predict([[3,2,1,1]]))
print("ID =22")
print(clf.predict([[3,2,1,2]]))
print("ID =23")
print(clf.predict([[3,2,2,1]]))
print("ID = 24")
print(clf.predict([[3,2,2,2]]))