from resnet import ResNet50
from keras.utils.visualize_util import plot
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard
import os
import sys 


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    if len(sys.argv) != 3:
        print("usage: python test.py [weights file] [output numpy name]")
        sys.exit(1)

    weights_path = sys.argv[1]
    numpy_name = sys.argv[2]
    model = ResNet50(input_tensor=None, input_shape=(6000, 1))
    model.load_weights(weights_path)
    plot(model, to_file="../../figures/resnetaudio.png", show_shapes=True)

    model.compile(loss='mean_squared_error',
                  optimizer='Adam',
                  metrics=['accuracy'])

    x_train = np.load('/home/data/urasam/sounds/VCTKnumpy/xcutest.npy')
    y_train = np.load('/home/data/urasam/sounds/VCTKnumpy/xtest.npy')

    x_train = x_train.reshape((len(x_train), 6000, 1))
    y_train = y_train.reshape((len(y_train), 6000, 1))

    print(x_train.shape, y_train.shape)

    sr_data = model.predict(x_train)
    np.save('/home/data/urasam/sounds/output/numpy/' + numpy_name, sr_data)

    #save_dir = '/home/data/urasam/sounds/weights/resasr/'
    #fpath = save_dir + 'weights_epoch{epoch:02d}-loss{loss:.2f}-acc{acc:.2f}.hdf5'
    #cp_cb = ModelCheckpoint(fpath)
    #tb_cb = TensorBoard(log_dir='/home/data/urasam/sounds/logs/resasr/', histogram_freq=0)

    #history = model.fit(x_train, y_train, batch_size=32, nb_epoch=500, callbacks=[cp_cb, tb_cb])
