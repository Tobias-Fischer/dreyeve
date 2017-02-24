import numpy as np

palette = np.array([[128, 64, 128],
                    [244, 35, 232],
                    [70, 70, 70],
                    [102, 102, 156],
                    [190, 153, 153],
                    [153, 153, 153],
                    [250, 170, 30],
                    [220, 220, 0],
                    [107, 142, 35],
                    [152, 251, 152],
                    [70, 130, 180],
                    [220, 20, 60],
                    [255, 0, 0],
                    [0, 0, 142],
                    [0, 0, 70],
                    [0, 60, 100],
                    [0, 80, 100],
                    [0, 0, 230],
                    [119, 11, 32]], dtype='uint8')


def seg_to_colormap(seg):
    """
    Function to turn segmentation PREDICTION (not probabilities) to colormap.

    :param seg: the prediction image, having shape (h,w)
    :return: the colormap image, having shape (h,w,3)
    """
    h, w = seg.shape
    color_image = palette[seg.ravel()].reshape(h, w, 3)

    return color_image