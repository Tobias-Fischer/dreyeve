import numpy as np
import cv2
import os.path as path

# cityscapes dataset palette
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


def seg_to_colormap(seg, channels_first):
    """
    Function to turn segmentation PREDICTION (not probabilities) to colormap.

    :param seg: the prediction image, having shape (h,w)
    :param channels_first: if true, returns (c,h,w) rather than (h,w,c)
    :return: the colormap image, having shape (h,w,3)
    """
    h, w = seg.shape
    color_image = palette[seg.ravel()].reshape(h, w, 3)

    if channels_first:
        color_image = color_image.transpose(2, 0, 1)

    return color_image


def read_lines_from_file(filename):
    """
    Function to read lines from file

    :param filename: The text file to be read.
    :return: content: A list of strings
    """
    with open(filename) as f:
        content = f.readlines()

    content = [x.strip() for x in content]
    return content


def get_branch_from_experiment_id(experiment_id):
    """
    Function to return model branch name given experiment_id.
    :param experiment_id: experiment id
    :return: a string among ['all','image','optical_flow','semseg']
    """

    assert isinstance(experiment_id, basestring), "Experiment ID must be a string."

    branch = None
    if experiment_id.lower().startswith('dreyeve'):
        branch = "all"
    elif experiment_id.lower().startswith('color'):
        branch = "image"
    elif experiment_id.lower().startswith('flow'):
        branch = "optical_flow"
    elif experiment_id.lower().startswith('segm'):
        branch = "semseg"

    return branch


def read_image(img_path, channels_first, color=True, color_mode='BGR', dtype=np.float32, resize_dim=None):

    """
    Reads and returns an image as a numpy array
    Parameters
    ----------
    img_path : string
        Path of the input image
    channels_first: bool
        If True, channel dimension is moved in first position
    color: bool, optional
        If True, image is loaded in color: grayscale otherwise
    color_mode: "RGB", "BGR", optional
        Whether to load the color image in RGB or BGR format
    dtype: dtype, optional
        Array is casted to this data type before being returned
    resize_dim: tuple, optional
        Resize size following convention (new_h, new_w) - interpolation is linear
    Returns
    -------
    image : np.array
        Loaded Image as numpy array of type dtype
    """

    if not path.exists(img_path):
        raise ValueError('Provided path "{}" does NOT exist.'.format(img_path))

    image = cv2.imread(img_path, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)

    if color and color_mode == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if resize_dim is not None:
        image = cv2.resize(image, dsize=resize_dim[::-1], interpolation=cv2.INTER_LINEAR)

    if color and channels_first:
        image = np.transpose(image, (2, 0, 1))

    return image.astype(dtype)


def normalize(img):
    """
    Normalizes an image between 0 and 255 and returns it as uint8.
    Parameters
    ----------
    img : ndarray
        Image that has to be normalized
    Returns
    ----------
    img : ndarray
        The normalized image
    """
    img = img.astype(np.float32)
    img -= img.min()
    img /= img.max()
    img *= 255
    img = img.astype(np.uint8)

    return img


def resize_tensor(tensor, new_shape):
    """
    Resize a numeric input 3D tensor with opencv. Each channel is resized independently from the others.
    
    Parameters
    ----------
    tensor: ndarray
        Numeric 3D tensor of shape (channels, h, w)
    new_shape: tuple
        Tuple (new_h, new_w)
    Returns
    -------
    new_tensor: ndarray
        Resized tensor having size (channels, new_h, new_w)
    """
    channels = tensor.shape[0]
    new_tensor = np.zeros(shape=(channels,) + new_shape)
    for i in range(0, channels):
        new_tensor[i] = cv2.resize(tensor[i], dsize=new_shape[::-1])

    return new_tensor


def stitch_together(input_images, layout, resize_dim=None, off_x=None, off_y=None, bg_color=(0, 0, 0)):
    """
    Stitch together N input images into a bigger frame, using a grid layout.
    Input images can be either color or grayscale, but must all have the same size.
    
    Parameters
    ----------
    input_images : list
        List of input images
    layout : tuple
        Grid layout of the stitch expressed as (rows, cols) 
    resize_dim : couple 
        If not None, stitch is resized to this size
    off_x : int
        Offset between stitched images along x axis
    off_y : int
        Offset between stitched images along y axis
    bg_color : tuple
        Color used for background
        
    Returns
    -------
    stitch : ndarray
        Stitch of input images
    """

    if len(set([img.shape for img in input_images])) > 1:
        raise ValueError('All images must have the same shape')

    if len(set([img.dtype for img in input_images])) > 1:
        raise ValueError('All images must have the same data type')
    
    # determine if input images are color (3 channels) or grayscale (single channel)
    if len(input_images[0].shape) == 2:
        mode = 'grayscale'
        img_h, img_w = input_images[0].shape
    elif len(input_images[0].shape) == 3:
        mode = 'color'
        img_h, img_w, img_c = input_images[0].shape
    else:
        raise ValueError('Unknown shape for input images')

    # if no offset is provided, set to 10% of image size
    if off_x is None:
        off_x = img_w // 10
    if off_y is None:
        off_y = img_h // 10

    # create stitch mask
    rows, cols = layout
    stitch_h = rows * img_h + (rows + 1) * off_y
    stitch_w = cols * img_w + (cols + 1) * off_x
    if mode == 'color':
        bg_color = np.array(bg_color)[None, None, :]  # cast to ndarray add singleton dimensions
        stitch = np.uint8(np.repeat(np.repeat(bg_color, stitch_h, axis=0), stitch_w, axis=1))
    elif mode == 'grayscale':
        stitch = np.zeros(shape=(stitch_h, stitch_w), dtype=np.uint8)

    for r in range(0, rows):
        for c in range(0, cols):

            list_idx = r * cols + c

            if list_idx < len(input_images):
                if mode == 'color':
                    stitch[r * (off_y + img_h) + off_y: r*(off_y+img_h) + off_y + img_h,
                           c * (off_x + img_w) + off_x: c * (off_x + img_w) + off_x + img_w,
                           :] = input_images[list_idx]
                elif mode == 'grayscale':
                    stitch[r * (off_y + img_h) + off_y: r*(off_y+img_h) + off_y + img_h,
                           c * (off_x + img_w) + off_x: c * (off_x + img_w) + off_x + img_w]\
                        = input_images[list_idx]

    if resize_dim:
        stitch = cv2.resize(stitch, dsize=(resize_dim[::-1]))

    return stitch
