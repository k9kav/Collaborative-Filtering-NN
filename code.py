import tensorflow as tf
from tensorflow import keras  
from sklearn import metrics
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from operator import add

#------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------IMPORT DATA------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------

movie_cols = ['userId', 'movieId', 'rating', 'timestamp']
ratings = pd.read_csv('http://files.grouplens.org/datasets/movielens/ml-100k/u.data', 
                    sep = '\t',
                    names = movie_cols,
                    encoding = 'latin-1')


#------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------PREPROCESSING----------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------

ratings = ratings.drop(columns = 'timestamp')
# Take every unique user id and map it to a contiguous user .
u_uniq = ratings.userId.unique()
user2idx = {o:i for i,o in enumerate(u_uniq)}
# Replace that userid with contiguous number.
ratings.userId = ratings.userId.apply(lambda x: user2idx[x]) 
ratings.rating = ratings.rating

#Do the same for movies
m_uniq = ratings.movieId.unique()
movie2idx = {o:i for i,o in enumerate(m_uniq)}
ratings.movieId = ratings.movieId.apply(lambda x: movie2idx[x] + 943)

x = ratings.drop(columns = "rating")
x = x.values

y = ratings.rating
y = y.values

#find average rating
df1 = ratings.groupby('userId')['rating'].agg(['count','mean']).reset_index()
avRating = df1.values 

#centering
for i in range(100000):
   y[i] = y[i] - avRating[x[i, 0]][2]


#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------MODEL-------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------

nOfepochs = 4000
nOfFolds = 5

#rmse, mae, lss, vrmse, vmae, vlss will hold average loss, RMSE and MAE values per epoch for training and validation data
rmse = vrmse = mae = vmae = lss = vlss = np.zeros((nOfepochs))

#out of sample predictions and y will be used to calculate total out of sample RMSE and MAE
oos_y = []
oos_pred = []

kf = KFold(nOfFolds, shuffle = True, random_state = 42)
fold = 0

for train, test in kf.split(x):
   fold+=1
   print(f"Fold #{fold}")

   x_train = x[train]
   y_train = y[train]
   x_test = x[test]
   y_test = y[test]

   #create model
   model = keras.Sequential()
   model.add(keras.layers.Embedding(input_dim = 2625, output_dim = 20, input_length =2))
   model.add(keras.layers.Flatten())
   model.add(keras.layers.Dense(units = 20, activation = 'relu'))
   model.add(keras.layers.Dense(1, activation = 'linear'))

   
   optimizer = keras.optimizers.SGD(lr=0.001)
   model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics=['RootMeanSquaredError', 'MeanAbsoluteError']) #RootMeanSquaredError MeanAbsoluteError
   history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=nOfepochs, batch_size=1000, verbose = 1)

   pred = model.predict(x_test)

   oos_y.append(y_test)
   oos_pred.append(pred)

   
   #this fold's RMSE 
   rmseScore = np.sqrt(metrics.mean_squared_error(pred, y_test))
   #this fold's MAE
   maeScore = metrics.mean_absolute_error(pred, y_test)
   print(f"Fold score (RMSE): {rmseScore}")
   print(f"Fold score (MAE): {maeScore}")

   #collect average training and validation data rmse for every fold's epochs
   r = history.history['RootMeanSquaredError']
   vr = history.history['val_RootMeanSquaredError']
   r[:] = [x/nOfFolds for x in r]
   rmse = list(map(add, rmse, r)) 
   vr[:] = [x/nOfFolds for x in vr]
   vrmse = list(map(add, vrmse, vr)) 

   #collect average training and validation data mae for every fold's epochs
   m = history.history['MeanAbsoluteError']
   vm = history.history['val_MeanAbsoluteError']
   m[:] = [x/nOfFolds for x in m]
   mae = list(map(add, mae, m)) 
   vm[:] = [x/nOfFolds for x in vm]
   vmae = list(map(add, vmae, vm)) 

   #collect average training and validation data loss for every fold's epochs
   l = history.history['loss']
   vl = history.history['val_loss']
   l[:] = [x/nOfFolds for x in l]
   lss = list(map(add, lss, l)) 
   vl[:] = [x/nOfFolds for x in vl]
   vlss = list(map(add, vlss, vl))

   

#oos prediction list and error
oos_y = np.concatenate(oos_y)
oos_pred = np.concatenate(oos_pred)
rmseScore = np.sqrt(metrics.mean_squared_error(oos_pred, oos_y))
maeScore = metrics.mean_absolute_error(oos_pred, oos_y)
print(f"Final, out of sample score (RMSE): {rmseScore}")
print(f"Final, out of sample score (MAE): {maeScore}")

#Plot for RMSE
plt.plot(rmse)
plt.plot(vrmse)
plt.title('Root Mean Squared Error')
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.legend(['train RMSE', 'test RMSE'], loc='lower left')

#Plot for MAE
plt.plot(mae)
plt.plot(vmae)
plt.title('Mean Absolute Error')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train MAE', 'test MAE'], loc='lower left')

#Plot for loss
plt.plot(lss)
plt.plot(vlss)
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train Loss', 'test Loss'], loc='lower left')
