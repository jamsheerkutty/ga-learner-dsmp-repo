# --------------
import matplotlib.pyplot as plt

# code starts here        
cols = X_train.columns

fig, axes = plt.subplots(nrows = 3 , ncols = 3, figsize=(20,20))

print(axes)
for i in range(3):
    for j in range(3):
        col = cols[i*3+j]
        axes[i][j].scatter(X_train[col],y_train)
     



# code ends here



# --------------
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
# code starts here

df = pd.read_csv(path)
print(df.head())

X = df.drop(['list_price'],axis=1)
#print(X.head())
y = df['list_price']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=6)




# code ends here



# --------------
# Code starts here
corr = X_train.corr()

X_train.drop(['play_star_rating','val_star_rating'],axis=1,inplace=True)

X_test.drop(['play_star_rating','val_star_rating'],axis=1,inplace=True)



# Code ends here


# --------------
# Code starts here
residual=y_test - y_pred
plt.hist(residual)




# Code ends here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Code starts here
regressor = LinearRegression()
# fit model on training data
regressor.fit(X_train, y_train)
# predict on test features
y_pred = regressor.predict(X_test)
mse=(mean_squared_error(y_test,y_pred))
print(mse)
r2=r2_score(y_test,y_pred)
print(r2)


# Code ends here

