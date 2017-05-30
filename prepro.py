import os
import wave
import sys
from tqdm import tqdm
import math
import numpy as np
import array


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
    X_target = []
    for file_name in tqdm(os.listdir(save_dir_clp_path)):
        if file_name.startswith("p") and file_name.endswith("wav"):
            wave_file = wave.open(save_dir_clp_path+file_name)
            x = down_sampe(wave_file)
            X_target.append(x)
    np.save(numpy_dir+"xlow.npy", np.array(X_target))
    print(np.array(X_target).shape)


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
    numpy_dir = home_dir + "sounds/VCTKnumpy/"
    down_and_save()


    # # down scale and upscale with bicubic
    # data_dir_clp_path = save_dir_clp_path
    # save_dir_data_path = home_dir + "images/srcnndata/inputdata/"
    # file_list = os.listdir(data_dir_clp_path)
    # for file_name in tqdm(file_list):
    #     down_scale(file_name, data_dir_clp_path, save_dir_data_path)

    # save_numpy("Xtrain33.npy",
    #            save_dir_data_path, '/home/data/urasam/images/srcnndata/')
    # save_numpy("Ytrain33.npy",
    #            save_dir_clp_path, '/home/data/urasam/images/srcnndata/')
