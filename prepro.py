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


def save_wav(x, save_dir_path, file_name):
    # input:np.array, output:wav file
    w = wave.Wave_write(save_dir_path + file_name)
    w.setparams((
        1,
        2,
        sample_rate_low,
        len(x),
        "NONE", "not commpressed"
        ))
    w.writeframes(array.array('h', x).tostring())
    w.close()


if __name__ == '__main__':
    sample_rate_low = 24000
    if len(sys.argv) != 3:
        print("usage: python prepro.py [sound file] savedir")
        sys.exit(1)
    file_name = sys.argv[1]
    save_dir_path = sys.argv[2]
    save_dir_path = save_dir_path + "/"

    wave_file = wave.open(file_name, "r")
    print("param")
    print(wave_file.getparams())

    x = down_sampe(wave_file)
    save_wav(x, save_dir_path, "test.wav")

    # clop pictures
    home_dir = "/home/data/urasam/"
    data_dir_raw_path = home_dir + "Image-Super-Resolution/input_images/"
    save_dir_clp_path = home_dir + "images/srcnndata/valclp_raw33/"
    # file_list = os.listdir(data_dir_raw_path)
    # for file_name in tqdm(file_list):
    #     # print(file_name)
    #     clopfunc(file_name, data_dir_raw_path, save_dir_clp_path, 33, 14)

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
