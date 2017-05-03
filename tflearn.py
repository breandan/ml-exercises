import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.metrics import r2_score
import tflearn
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('housing_train.csv')
labels = train['SalePrice']

test = pd.read_csv('housing_test.csv')
ids = test['Id']

tf.reset_default_graph()
r2 = tflearn.R2()
net = tflearn.input_data(shape=[None, train.shape[1]])
net = tflearn.fully_connected(net, 20, activation='linear')
# net = tflearn.fully_connected(net, 10, activation = 'linear')
net = tflearn.fully_connected(net, 1, activation = 'linear')
sgd = tflearn.SGD(learning_rate=0.1, lr_decay = 0.01, decay_step=100)
net = tflearn.regression(net, optimizer=sgd, loss='mean_square', metric=r2)
model = tflearn.DNN(net)

model.fit(train, labels, show_metric=True, validation_set=0.2, shuffle = True, n_epoch=50)

predictions = model.predict(test)
predictions = np.exp(predictions)
predictions = predictions.reshape(-1,)

output = pd.DataFrame({"id":ids, "SalePrice": predictions})
output