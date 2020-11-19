import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
tfk = tf.keras
tfkl = tf.keras.layers
#import glob
tfku = tf.keras.utils

def scale_field(field_in, field_type, inv=False):
    '''
    function to scale Lagrangian, Eulerian, and reionization fields
    from their physical units to the range (0,1), and invert this scaling if needed.
    Can choose other scalings from e.g. https://en.wikipedia.org/wiki/Feature_scaling - right now does min-max scaling.
    Keyword arguments:
    field_in -- the field to scale
    field_type -- type of field: 'lag', 'eul', or 'reion'
    inv -- False=forward scaling from physical units to normalized, True=inverse
    """
    '''
    field_types =  ['eul', 'lag', 'reion']
    field_mins  = np.array([ -1.        , -23.45174217,   6.69999981])
    field_maxs  = np.array([11.14400196, 22.52767944, 16.        ])
    ind = field_types.index(field_type)
    fmin = field_mins[ind]
    fmax = field_maxs[ind]
    #min max scaling
    if not inv:
        return (field_in - fmin)/(fmax - fmin)
    if inv:
        return field_in * (fmax - fmin) + fmin
    
N = 512
datadir = '/global/cscratch1/sd/tmedan/notebooks/'


def cubify(arr, newshape):
    '''stolen from https://stackoverflow.com/questions/42297115/numpy-split-cube-into-cubes'''
    oldshape = np.array(arr.shape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.column_stack([repeats, newshape]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    
    # newshape must divide oldshape evenly or else ValueError will be raised
    return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)

def uncubify(arr, oldshape):
    '''stolen from https://stackoverflow.com/questions/42297115/numpy-split-cube-into-cubes'''
    N, newshape = arr.shape[0], arr.shape[1:]
    oldshape = np.array(oldshape)    
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.concatenate([repeats, newshape])
    order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
    return arr.reshape(tmpshape).transpose(order).reshape(oldshape)

n_filters = 36

img_shape = 32

Input_shape=(img_shape,img_shape,img_shape,1)

model = tfk.Sequential()
    
model.add(tfkl.Conv3D(filters=n_filters,
                     kernel_size=(5,5,5),
                     strides=(1,1,1),
                     use_bias=True,
                     input_shape=Input_shape,
                      padding = 'same'))
model.add(tfkl.BatchNormalization())
model.add(tfkl.Activation('relu'))

model.add(tfkl.Conv3D(filters=n_filters,
                     kernel_size=(3,3,3),
                     strides=(1,1,1),
                     use_bias=True,
                     input_shape=Input_shape,
                      padding = 'same'))
model.add(tfkl.BatchNormalization())
model.add(tfkl.Activation('relu'))

model.add(tfkl.Conv3D(filters=n_filters,
                     kernel_size=(3,3,3),
                     strides=(1,1,1),
                     use_bias=True,
                     input_shape=Input_shape,
                      padding = 'same'))
model.add(tfkl.BatchNormalization())
model.add(tfkl.LeakyReLU(alpha = 0.3))

model.add(tfkl.Conv3D(filters=n_filters,
                     kernel_size=(1,1,1),
                     strides=(1,1,1),
                     use_bias=True,
                     input_shape=Input_shape,
                      padding = 'same'))
model.add(tfkl.BatchNormalization())
model.add(tfkl.LeakyReLU(alpha = 0.3))

model.add(tfkl.Conv3D(filters=1,
                     kernel_size=(1,1,1),
                     strides=(1,1,1),
                     use_bias=False,padding= 'same'))
model.add(tfkl.Activation('relu'))


#training data

def data_feeder(n_files, img_shape):
    sim_size  =  512
    n_subcube = int((sim_size//img_shape)**3)
    # make empty array first.
    densE = np.zeros((n_files, n_subcube, img_shape, img_shape, img_shape))
    densL = np.zeros((n_files, n_subcube, img_shape, img_shape, img_shape))
    reion = np.zeros((n_files, n_subcube, img_shape, img_shape, img_shape))
    for i in range(n_files):
        # load in simulation
        densEfile_i = datadir+'density_Eul/dens_{:02d}'.format(i)
        densE_i = np.fromfile(open(densEfile_i),count=sim_size**3,dtype=np.float32).reshape(sim_size, sim_size, sim_size)
        
        densLfile_i = datadir+'density_Lag/dens_{:02d}'.format(i)
        densL_i = np.fromfile(open(densLfile_i),count=sim_size**3,dtype=np.float32).reshape(sim_size, sim_size, sim_size)
        
        reionfile_i = datadir+'reionization/reion_{:02d}'.format(i)
        reion_i = np.fromfile(open(reionfile_i),count=sim_size**3,dtype=np.float32).reshape(sim_size, sim_size, sim_size)
        #reshape to subcubes
        densE[i] = cubify(densE_i,(img_shape,img_shape,img_shape))
        densL[i] = cubify(densL_i,(img_shape,img_shape,img_shape))
        reion[i] = cubify(reion_i,(img_shape,img_shape,img_shape))
    # add additional axis for number of channels
    densE = densE[..., np.newaxis]
    densL = densL[..., np.newaxis]
    reion = reion[..., np.newaxis]
    # scale all fields at once
    densE_scaled = scale_field(densE,'eul')
    densL_scaled = scale_field(densL,'lag')
    reion_scaled = scale_field(reion,'reion')
    return densE_scaled,densL_scaled,reion_scaled

data = data_feeder(3,32)
densE_train = data[0]
densL_train = data[1]
reion_train = data[2]

# get test data
freion = open(datadir+'reionization/reion_08')
fdens  = open(datadir+'density_Eul/dens_08')

x_test  = np.fromfile(fdens, count=N**3, dtype=np.float32).reshape(N,N,N)
y_test  = np.fromfile(freion, count=N**3, dtype=np.float32).reshape(N,N,N)


x_test = cubify(x_test,  (img_shape, img_shape, img_shape))[..., np.newaxis]
y_test = cubify(y_test,  (img_shape, img_shape, img_shape))[..., np.newaxis]

x_test = scale_field(x_test,'eul')
y_test = scale_field(y_test,'reion')

reion_test         = scale_field(y_test,'reion', inv=True)

optimizer = tfk.optimizers.Adam(1e-2)
model.compile(optimizer=optimizer, loss='mse',metrics=['mae'])

loss = []
val_loss = []

for i in range(densE_train.shape[0]):
    history = model.fit(densE_train[i],reion_train[i],
                    epochs=50,
                    validation_split=0.1)
    loss.append(history.history['loss'])
    val_loss.append(history.history['val_loss'])

loss = np.array(loss).flatten()
val_loss = np.array(val_loss).flatten()

# summarize history for loss
plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.yscale('log')
plt.xscale('log')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('5_layer_CNN_scaled_trainplot')
plt.show()

model.save('5_layer_CNN_scaled')


y_test_predict = model.predict(x_test)

reion_test_predict = scale_field(y_test_predict,'reion', inv=True)


yshow_predict = reion_test_predict[0,img_shape//2, ..., 0]
yshow = reion_test[0,img_shape//2, ..., 0]
xshow = x_test[0,img_shape//2, ..., 0]

ymin = yshow.min()
ymax = yshow.max()

plt.figure()
plt.scatter(yshow.flatten(),yshow_predict.flatten(),c='k',s=0.5)
xx = np.linspace(yshow.min(), yshow.max(), 100)
plt.plot(xx,xx, 'k')
plt.savefig('5_layer_CNN_scaled_scatter')
#this is our transormation plot, actual versus predicted redshifts

plt.figure()
plt.imshow(yshow_predict, vmin=ymin, vmax=ymax)
plt.title('Prediction')
plt.colorbar()
plt.savefig('5_layer_CNN_scaled_predict')


plt.figure()
plt.imshow(yshow)
plt.title('Actual 21CM Run')
plt.colorbar()
plt.savefig('5_layer_CNN_scaled_actual')

diff = yshow_predict-yshow
plt.figure()
plt.title('Difference Plot')
plt.imshow(diff, vmin=-np.abs(diff).max(), vmax=np.abs(diff).max(), cmap=plt.get_cmap('coolwarm'))
plt.colorbar()
plt.savefig('5_layer_CNN_scaled_diff')

