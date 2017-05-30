import os
import wave
import sys
from tqdm import tqdm
import math
import numpy as np
import array
from scipy import signal, interpolate


# {{{
def clopfunc(f_name, data_dir_path, save_dir_path, pixel, stride):
    img = cv2.imread(data_dir_path+f_name)
    # print(img.shape)
    clp_name = f_name.split(".")[0]
    height, width, channels = img.shape
    high = 0
    while high+pixel < height:
        wide = 0
        while wide+pixel < width:
            clp = img[high:(high+pixel), wide:(wide+pixel)]
            # print(high, wide)
            cv2.imwrite(save_dir_path+clp_name + "-" +
                        str(high) + "-"+str(wide) + ".jpg", clp)
            wide += stride
        high += stride


def down_scale(f_name, data_dir_path, save_dir_path):
    img = cv2.imread(data_dir_path+f_name)
    down_name = f_name.split(".")[0]
    I = resize(img, (math.floor(img.shape[0]*0.5),
                     math.floor(img.shape[1]*0.5)))
    IN = imresize(I, (I.shape[0]*2+1, I.shape[1]*2+1), interp='bicubic')
    cv2.imwrite(save_dir_path+down_name+".jpg", IN)


def save_numpy(numpy_name, data_dir_path, save_dir_path):
    X_target = []
    file_list = os.listdir(data_dir_path)
    for file_name in tqdm(file_list):
        if file_name.endswith('.jpg') and not file_name.startswith('._'):
            image_path = str(data_dir_path) + str(file_name)
            image = cv2.imread(image_path)
            image = image/255.
            X_target.append(image)
    np.save(save_dir_path+numpy_name, np.array(X_target))
    print(np.array(X_target).shape)
# }}}


def down_sampe(wave_file):
    # input:wave file, return:np.array
    x = wave_file.readframes(wave_file.getnframes())
    x = np.frombuffer(x, dtype="int16")
    x = x[::2]
    return x


def save_wav(x, save_dir_path, file_name, sample_rate):
    # input:np.array, output:wav file
    w = wave.Wave_write(save_dir_path + file_name)
    w.setparams((
        1,
        2,
        sample_rate,
        len(x),
        "NONE", "not commpressed"
        ))
    w.writeframes(array.array('h', x).tostring())
    w.close()


def clopwave():
    dir_list = os.listdir(data_dir_raw_path)
    for dir_name in tqdm(dir_list):
        if dir_name.startswith("p"):
            file_list = os.listdir(data_dir_raw_path+dir_name)
            for file_name in file_list:
                if file_name.startswith("p") and file_name.endswith("wav"):
                    # print(data_dir_raw_path+dir_name+"/"+file_name)
                    wave_file = wave.open(data_dir_raw_path+dir_name+"/"+file_name)
                    x = wave_file.readframes(wave_file.getnframes())
                    x = np.frombuffer(x, dtype="int16")
                    for i in range(len(x)/data_length):
                        save_wav(x[i*data_length:(i+1)*data_length], save_dir_clp_path, file_name.split(".")[0]+str(i)+".wav", sample_rate_high)


def down_and_save():
    # save numpy array
    X_target = []
    for file_name in tqdm(os.listdir(data_dir_train)):
        if file_name.startswith("p") and file_name.endswith("wav"):
            wave_file = wave.open(data_dir_train+file_name)
            x = down_sampe(wave_file)
            save_wav(x, data_dir_low_path+"train/", file_name, sample_rate_low)
            X_target.append(x)
    np.save(numpy_dir+"xlowtrain.npy", np.array(X_target))
    print(np.array(X_target).shape)


def mvfiles():
    """mv file dayo"""
    file_list = os.listdir(save_dir_clp_path)
    for file_name in tqdm(file_list):
        if file_name.startswith("p3"):
            os.system('mv '+save_dir_clp_path+file_name+" "+data_dir_test)


def up_sample_cubic(x):
    # imput:np.array low data, return:np.array hith data
    x_high = []
    for i in tqdm(range(len(x))):
        # print(x[i,:].shape)
        y_interf = interpolate.CubicSpline(np.linspace(0, data_length/2-1, data_length/2),
                                           x[i,:])
        y_inter = y_interf(np.linspace(0, data_length/2-1, data_length)).astype(np.int16)
        x_high.append(y_inter)
    return np.array(x_high)

if __name__ == '__main__':
    sample_rate_low = 24000
    sample_rate_high = 48000
    data_length = 6000

    # clop wave file
    home_dir = "/home/data/urasam/"
    data_dir_raw_path = home_dir + "sounds/VCTK-Corpus/wav48/" # raw data
    save_dir_clp_path = home_dir + "sounds/VCTKclpraw/" # all 6000 length data
    # down sample
    data_dir_low_path = home_dir + "sounds/VCTKlow/" # down sample data
    data_dir_train = home_dir + "sounds/VCTKclptrain/" # clp data
    data_dir_test = home_dir + "sounds/VCTKclptest/" # clp data
    numpy_dir = home_dir + "sounds/VCTKnumpy/"

    X_target = []
    file_list = os.listdir(data_dir_train)
    for file_name in tqdm(file_list):
        if file_name.startswith("p") and file_name.endswith("wav"):
            wave_file = wave.open(data_dir_train+file_name)
            x = wave_file.readframes(wave_file.getnframes())
            x = np.frombuffer(x, dtype="int16")
            X_target.append(x)
    np.save(numpy_dir+"xtrain.npy", np.array(X_target))
    print(np.array(X_target).shape)

    # mvfiles()
    # down_and_save()
    # up_sample_cubic()
    # x = np.load(numpy_dir+"xlowtrain.npy")
    # up_sample_cubic(x)
    # x_up = up_sample_cubic(x)
    # np.save(numpy_dir+"xcutrain.npy", x_up)
    # print(x_up.shape)

    # t_low = np.linspace(0, 2999, 3000)
    # t_high = np.linspace(0, 2999, 6000)
    # y = np.sin(t_low)
    # y = t_low
    # cs = interpolate.CubicSpline(t_low, y)
    # y_inter = cs(t_high)
    #print(y)
    #print(y_inter)
