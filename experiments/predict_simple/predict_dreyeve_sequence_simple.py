import os; os.environ['KERAS_BACKEND'] = 'theano'# ; os.environ['THEANO_FLAGS'] = 'device=cuda,floatX=float32'
# os.environ['CPLUS_INCLUDE_PATH'] = '/home/tobias/anaconda3/envs/dreyeve/include:/usr/local/cuda-10.1/targets/x86_64-linux/include'
# os.environ['LIBRARY_PATH'] = '/home/tobias/anaconda3/envs/dreyeve/lib'

import numpy as np
import cv2

import argparse

import os
from tqdm import tqdm
from os.path import join

import keras
keras.backend.set_image_dim_ordering('th')

from models import DreyeveNet
from utils import read_image, normalize, resize_tensor, stitch_together


def makedirs(dir_list):
    """
    Helper function to create a list of directories.

    :param dir_list: a list of directories to be created
    """

    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)


def load_dreyeve_sample(sequence_dir, sample, mean_dreyeve_image, frames_per_seq=16, h=448, w=448):
    """
    Function to load a dreyeve_sample.

    :param sequence_dir: string, sequence directory (e.g. 'Z:/DATA/04/').
    :param sample: int, sample to load in (15, 7499). N.B. this is the sample where prediction occurs!
    :param mean_dreyeve_image: mean dreyeve image, subtracted to each frame.
    :param frames_per_seq: number of temporal frames for each sample
    :param h: h
    :param w: w
    :return: a dreyeve_sample like I, OF, SEG
    """

    h_c = h_s = h // 4
    w_c = w_s = h // 4

    I_ff = np.zeros(shape=(1, 3, 1, h, w), dtype='float32')
    I_s = np.zeros(shape=(1, 3, frames_per_seq, h_s, w_s), dtype='float32')
    I_c = np.zeros(shape=(1, 3, frames_per_seq, h_c, w_c), dtype='float32')

    for fr in xrange(0, frames_per_seq):
        offset = sample - frames_per_seq + 1 + fr + 1  # tricky

        # read image
        x = read_image(join(sequence_dir, 'frames', '{:06d}.png'.format(offset)),
                       channels_first=True, resize_dim=(h, w)) - mean_dreyeve_image
        I_s[0, :, fr, :, :] = resize_tensor(x, new_shape=(h_s, w_s))

    I_ff[0, :, 0, :, :] = x

    return [I_ff, I_s, I_c]


if __name__ == '__main__':
    frames_per_seq, h, w = 16, 448, 448
    verbose = False

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int)
    parser.add_argument("--pred_dir", type=str)
    args = parser.parse_args()

    assert args.seq is not None, 'Please provide a correct dreyeve sequence'
    assert args.pred_dir is not None, 'Please provide a correct pred_dir'

    dreyeve_dir = '/media/storage_hdd/Datasets/DREYEVE_DATA'

    # load mean dreyeve image
    mean_dreyeve_image = read_image('dreyeve_mean_frame.png',
                                    channels_first=True, resize_dim=(h, w))

    # get the models
    dreyevenet_model = DreyeveNet(frames_per_seq=frames_per_seq, h=h, w=w)
    dreyevenet_model.compile(optimizer='adam', loss='kld')  # do we need this?
    dreyevenet_model.load_weights('dreyevenet_model.h5')  # load weights

    image_branch = [l for l in dreyevenet_model.layers if l.name == 'image_saliency_branch'][0]

    # set up some directories
    image_pred_dir = join(args.pred_dir, '{:02d}'.format(int(args.seq)), 'image_branch')

    makedirs([image_pred_dir])

    sequence_dir = join(dreyeve_dir, '{:02d}'.format(int(args.seq)))
    num_images = len(os.listdir(join(sequence_dir, 'frames')))
    for sample in tqdm(range(frames_per_seq - 1, num_images - 1)):
        X = load_dreyeve_sample(sequence_dir=sequence_dir, sample=sample, mean_dreyeve_image=mean_dreyeve_image,
                                frames_per_seq=frames_per_seq, h=h, w=w)

        Y_image = image_branch.predict(X[:3])[0]  # predict on image

        # save model output
        np.savez_compressed(join(image_pred_dir, '{:06d}'.format(sample)), Y_image)

        if verbose:
            # visualization
            x_stitch = stitch_together([normalize(X[0][0, :, 0, :, :].transpose(1, 2, 0))], layout=(1, 1),
                                       resize_dim=(720, 720))

            y_tot = np.tile(normalize(resize_tensor(Y_image[0], new_shape=(720, 720)).transpose(1, 2, 0)),
                            reps=(1, 1, 3))

            cv2.imshow('prediction', stitch_together([x_stitch, y_tot], layout=(1, 2),
                                                     resize_dim=(500, 1000)))
            cv2.waitKey(1)
