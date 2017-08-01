from BayesianNNCopyCat import *

copycat_agent = BayesianNNCopyCat()

X = np.array([[1, 2],[3, 4]])
print(copycat_agent.predict(X))