import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
seed(0)
############################################## Scientific packages
import tensorflow as tf
tf.random.set_seed(0)
import keras
from keras.models import *
from keras.layers import *
from sklearn.metrics import accuracy_score
from sklearn import datasets

############################################## Making Datasets
n_points = 8
circles = datasets.make_circles(n_samples=1000, noise=.05, factor=0.3, random_state=3)
circles_test = datasets.make_circles(n_samples=10000, noise=0, factor=0.3, random_state=1)
inds = list(np.where(circles[1] == 0)[0][:n_points]) + list(np.where(circles[1] == 1)[0][:n_points])

X_train = circles[0][inds]
Y_train = circles[1][inds]
Y_train_cat = keras.utils.to_categorical(circles[1][inds])

X_test = circles_test[0][inds]
Y_test = circles_test[1][inds]
Y_test_cat = keras.utils.to_categorical(circles_test[1][inds])

n_class = int(np.max(Y_train) + 1) # Total number of class present : max value is 1 in our case

############################################## Plot Datasets
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, s=20, cmap='winter', edgecolors='None', alpha=0.005)
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, s=20, cmap='winter', edgecolors='k')
plt.show()

def plot_model_predictions(m):
    xx, yy = np.meshgrid(np.arange(-1.4, 1.4, 0.1),
                         np.arange(-1.8, 1.4, 0.1))

    Z = m.predict(np.c_[xx.ravel(), yy.ravel()]).argmax(-1)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='Greens')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, s=20, cmap='winter', edgecolor='none', alpha=0.005)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, s=20, cmap='winter', edgecolor='k')
    plt.show()

############################################## Model without VAT
model = Sequential()
model.add( Dense(100 ,activation='relu' ,  input_shape=(2,)))
model.add( Dense(2 , activation='softmax' ))
model.compile( 'sgd' ,  'categorical_crossentropy'  ,  metrics=['accuracy'])

model.fit(  np.concatenate([X_train]*10000) , np.concatenate([Y_train_cat]*10000)  )

y_pred  = model.predict( X_test ).argmax(-1)
print("Test accruracy " , accuracy_score(Y_test , y_pred  ))

############################################## Plot the Model
plot_model_predictions(model)

############################################## Model without VAT
def compute_kld(p_logit, q_logit):
    p = tf.nn.softmax(p_logit)
    q = tf.nn.softmax(q_logit)
    return tf.reduce_sum(p*(tf.math.log(p + 1e-16) - tf.math.log(q + 1e-16)), axis=1)


def make_unit_norm(x):
    return x/(tf.reshape(tf.sqrt(tf.reduce_sum(tf.pow(x, 2.0), axis=1)), [-1, 1]) + 1e-16)

network = Sequential()
network.add( Dense(100 ,activation='relu' ,  input_shape=(2,)))
network.add( Dense(2   ))

model_input = Input((2,))
p_logit = network( model_input )
p = Activation('softmax')( p_logit )

r = tf.random.normal(shape=tf.shape( model_input ))
r = make_unit_norm( r )
p_logit_r = network( model_input + 10*r  )


kl = tf.reduce_mean(compute_kld( p_logit , p_logit_r ))
grad_kl = tf.GradientTape( kl , [r ])
r_vadv = tf.stop_gradient(grad_kl)
r_vadv = make_unit_norm( r_vadv )/3.0


p_logit_no_gradient = tf.stop_gradient(p_logit)
p_logit_r_adv = network( model_input  + r_vadv )
vat_loss =  tf.reduce_mean(compute_kld( p_logit_no_gradient, p_logit_r_adv ))


model_vat = Model(model_input , p )
model_vat.add_loss( vat_loss   )

model_vat.compile( 'sgd' ,  'categorical_crossentropy'  ,  metrics=['accuracy'])

model_vat.metrics_names.append('vat_loss')
model_vat.metrics_tensors.append( vat_loss )

model_vat.fit(  np.concatenate([X_train]*10000) , np.concatenate([Y_train_cat]*10000)  )

y_pred  = model_vat.predict( X_test ).argmax(-1)
print( "Test accruracy " , accuracy_score(Y_test , y_pred  ))

plot_model_predictions( model_vat  )