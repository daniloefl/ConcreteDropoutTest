import tensorflow as tf
import numpy as np
from utils import *
import gc
from scipy.ndimage.filters import gaussian_filter

from sklearn.datasets import make_moons
x_train, y_train = make_moons(n_samples = 5000, noise = 0.20)
x_test, y_test = make_moons(n_samples = 2000, noise = 0.20)
y_test = y_test.astype(np.float32)
y_train = y_train.astype(np.float32)

batch_size = 20
Nepochs = 200
train_size = len(x_train)
L = x_train.shape[1]

lengthscale = 1e-4
wd = lengthscale**2/train_size
dd = 2./train_size
layers = [200, 100, 50, 5]

x_in = tf.keras.layers.Input(shape = (L,))
x = x_in
for il, l in enumerate(layers):
    x = ConcreteDropout(tf.keras.layers.Dense(l, activation = tf.nn.relu), weight_regularizer = wd, dropout_regularizer = dd)(x)
m = ConcreteDropout(tf.keras.layers.Dense(1, activation = None), weight_regularizer = wd, dropout_regularizer = dd)(x)

def hloss(y_true, y_pred):
    mean_ = y_pred[:,0]
    y_ = y_true[:,0]
    return tf.square(y_ - mean_)

def mse(y_true, y_pred):
    mean_ = y_pred[:,0]
    y_ = y_true[:,0]
    return tf.reduce_mean(tf.square(y_ - mean_), -1)

def mae(y_true, y_pred):
    mean_ = y_pred[:,0]
    y_ = y_true[:,0]
    return tf.reduce_mean(tf.abs(y_ - mean_), -1)

model = tf.keras.Model(x_in, m)
model.compile(loss = hloss, optimizer = tf.keras.optimizers.Adam(1e-3), metrics = [mse, mae])
model.fit(x_train, y_train, epochs = Nepochs, batch_size = batch_size, shuffle = True, validation_data = (x_test, y_test), callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'loss', min_delta = 0, patience = 5, verbose = 1, mode = 'auto', restore_best_weights = True)])
p_list = [tf.nn.sigmoid(x) for x in model.trainable_weights if "p_logit" in x.name]
print("Dropout probabilities:", p_list)

def getMeanStd(model, x_test, Npred = 50):
    pred_m = np.zeros( (len(x_test),) )
    pred_s = np.zeros( (len(x_test),) )
    for i in range(0, Npred):
        tmp = model.predict(x_test)[:,0]
        pred_m += tmp
        pred_s += tmp**2
        del tmp
        gc.collect()
    pred_m /= float(Npred)
    pred_s /= float(Npred)
    pred_s -= pred_m**2
    pred_s[pred_s <= 0] = 1e-6
    pred_s = np.sqrt(pred_s)
    return pred_m, pred_s

#Npred = 50
#pred_m, pred_s = getMeanStd(model, x_test, Npred = Npred)

import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

def plot_contour(model, x, y):
    # make countour
    mins = [np.min(x[:,0]), np.min(x[:,1])]
    maxs = [np.max(x[:,0]), np.max(x[:,1])]
    step = [(maxs[0] - mins[0])/100.0, (maxs[1] - mins[1])/100.0]
    bx, by = np.mgrid[mins[0]:(maxs[0]+0.5*step[0]):step[0], mins[1]:(maxs[1]+0.5*step[0]):step[1]]
    inputs = np.vstack([bx.flatten(), by.flatten()]).T
    inputs = inputs.astype(np.float32)

    pred_m, pred_s = getMeanStd(model, inputs, Npred = 50)
    pred_m_2d = pred_m.reshape( (-1, bx.shape[1]) )
    pred_s_2d = pred_s.reshape( (-1, bx.shape[1]) )

    # if one wants to smoothen the results
    #for data in [pred_m_2d, pred_s_2d]:
    #    data = gaussian_filter(data, 0.1)

    fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (10, 8))
    cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)
    contour_s = ax[0].contourf(bx, by, pred_s_2d, cmap = cmap)
    cbar_s = plt.colorbar(contour_s, ax = ax[0])
    cbar_s.ax.set_ylabel('Unc.')
    contour_m = ax[1].contourf(bx, by, pred_m_2d, cmap = cmap)
    cbar_m = plt.colorbar(contour_m, ax = ax[1])
    cbar_m.ax.set_ylabel('Mean')
    for a in [ax[0], ax[1]]:
        a.scatter(x[y == 1,0], x[y == 1,1], color = 'r', marker = 's', s = 5, label = 'y = 1')
        a.scatter(x[y == 0,0], x[y == 0,1], color = 'b', marker = 's', s = 5, label = 'y = 0')
        a.set(xlabel = 'A', ylabel = 'B', title = '')
        a.set_xlim([mins[0], maxs[0]])
        a.set_ylim([mins[1], maxs[1]])
        a.legend(frameon = True)

    ax[0].set_xlabel('')
    fig.subplots_adjust(hspace = 0)
    fig.tight_layout()
    plt.show()

plot_contour(model, x_test, y_test)

