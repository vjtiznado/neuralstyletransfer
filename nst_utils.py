import os
import sys
import scipy.io
import scipy.misc
from imageio import imwrite
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf

class CONFIG:
    IMAGE_WIDTH = 800
    IMAGE_HEIGHT = 600
    COLOR_CHANNELS = 3
    NOISE_RATIO = 0.6
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    VGG_MODEL = "./imagenet-vgg-verydeep-19.mat"
    STYLE_IMAGE = "./starry_night.jpg"
    CONTENT_IMAGE = "./content_image_01_800x600.jpg"
    OUTPUT_DIR = "./output"


def save_style_layers(output_path, STYLE_LAYERS):
    with open(output_path + "/style_layers.txt", "w") as output:
        for row in STYLE_LAYERS:
            output.write(str(row) + "\n")

    return None


def create_output_folder(content_imgname, style_imgname):
    """
    Creates the folder where output frames are saved. It takes its name
    based on names of content and style images, respectively

    arguments:
        content_imgname -- filename or full path of the content image
        style_imgname -- filename or full path of the style image

    returns:
        output_folder -- string with the name of the outputs' folder
    """
    content_basename = os.path.splitext(os.path.basename(content_imgname))[0]
    style_basename = os.path.splitext(os.path.basename(style_imgname))[0]

    output_folder = (content_basename + "_" + style_basename)
    os.makedirs(output_folder, exist_ok=True)

    return output_folder


def load_vgg_model(path):
    """
    Returns a model for the purpose of "painting" the picture.
    Takes only the convolution layer weights and wrap using the tensofrlow
    Conv2d, Relu and AveragePooling layer. VGG actually uses maxpool but
    the paper indicates that using AveragePooling yields better results.
    The last few fully connected layers are not used.
    """
    vgg = scipy.io.loadmat(path)
    vgg_layers = vgg['layers']

    def _weights(layer, expected_layer_name):
        """
        Returns the weights and bias from the VGG model for a given layer
        """
        wb = vgg_layers[0][layer][0][0][2]
        W = wb[0][0]
        b = wb[0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        assert layer_name == expected_layer_name

        return W, b

    def _relu(conv2d_layer):
        """
        Returns the RELU function wrapped over a tensorflow layer. It
        expects a Conv2d layer input.
        """
        return tf.nn.relu(conv2d_layer)

    def _conv2d(prev_layer, layer, layer_name):
        """
        Returns the Conv2D layer using the weights, biases from the VGG model
        at "layer".
        """
        W, b = _weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1],
                            padding="SAME")+b

    def _conv2d_relu(prev_layer, layer, layer_name):
        """
        Returns the Conv2D + RELU layer using the weights, biases from the VGG
        model ay "layer".
        """
        return _relu(_conv2d(prev_layer, layer, layer_name))

    def _avgpool(prev_layer):
        """
        Returns the AveragePooling layer.
        """

        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding="SAME")
    # Constructs the graph model.
    graph = {}
    graph["input"] = tf.Variable(np.zeros((1, CONFIG.IMAGE_HEIGHT,
                                           CONFIG.IMAGE_WIDTH,
                                           CONFIG.COLOR_CHANNELS)),
                                 dtype="float32")
    graph["conv1_1"] = _conv2d_relu(graph["input"], 0, "conv1_1")
    graph["conv1_2"] = _conv2d_relu(graph["conv1_1"], 2, "conv1_2")
    graph["avgpool1"] = _avgpool(graph["conv1_2"])
    graph["conv2_1"] = _conv2d_relu(graph["avgpool1"], 5, "conv2_1")
    graph["conv2_2"] = _conv2d_relu(graph["conv2_1"], 7, "conv2_2")
    graph["avgpool2"] = _avgpool(graph["conv2_2"])
    graph["conv3_1"] = _conv2d_relu(graph["avgpool2"], 10, "conv3_1")
    graph["conv3_2"] = _conv2d_relu(graph["conv3_1"], 12, "conv3_2")
    graph["conv3_3"] = _conv2d_relu(graph["conv3_2"], 14, "conv3_3")
    graph["conv3_4"] = _conv2d_relu(graph["conv3_3"], 16, "conv3_4")
    graph["avgpool3"] = _avgpool(graph["conv3_4"])
    graph["conv4_1"] = _conv2d_relu(graph["avgpool3"], 19, "conv4_1")
    graph["conv4_2"] = _conv2d_relu(graph["conv4_1"], 21, "conv4_2")
    graph["conv4_3"] = _conv2d_relu(graph["conv4_2"], 23, "conv4_3")
    graph["conv4_4"] = _conv2d_relu(graph["conv4_3"], 25, "conv4_4")
    graph["avgpool4"] = _avgpool(graph["conv4_4"])
    graph["conv5_1"] = _conv2d_relu(graph["avgpool4"], 28, "conv5_1")
    graph["conv5_2"] = _conv2d_relu(graph["conv5_1"], 30, "conv5_2")
    graph["conv5_3"] = _conv2d_relu(graph["conv5_2"], 32, "conv5_3")
    graph["conv5_4"] = _conv2d_relu(graph["conv5_3"], 34, "conv5_4")
    graph["avgpool5"] = _avgpool(graph["conv5_4"])

    return graph

def generate_noise_image(content_image, noise_ratio = CONFIG.NOISE_RATIO):
    """
    Generates a noisy image by adding random noise to the content_image
    """
    noise_image = np.random.uniform(-20, 20,
                                    (1, CONFIG.IMAGE_HEIGHT,
                                     CONFIG.IMAGE_WIDTH,
                                     CONFIG.COLOR_CHANNELS)).astype("float32")
    # set the input image to be a weighted average of the content image and
    # a noise image
    input_image = noise_image*noise_ratio + content_image*(1-noise_ratio)

    return input_image


def reshape_and_normalize_image(image):
    """
    Reshapes and normalizes the input image (content or style).
    It resizes the image to match the CONFIG dimensions, and
    adds a fourth dimension in the first position of the image's shape.
    Then, it subtracts the mean values obtained from the VGG model
    """
    # resizing the image to match CONFIG dimensions
    PIL_image = Image.fromarray(np.uint8(image)).convert("RGB")
    resized = PIL_image.resize((CONFIG.IMAGE_WIDTH,CONFIG.IMAGE_HEIGHT))
    resized = np.array(resized)

    # reshape image to match expected input of VGG16
    reshaped = np.reshape(resized, ((1,) + resized.shape))

    # subtract the mean to match the expected input of VGG16
    normalized = reshaped - CONFIG.MEANS

    return normalized


def save_image(path, image):

    # un-normalize the image so that it looks good
    image = image + CONFIG.MEANS

    # clip and save the image
    image = np.clip(image[0], 0, 255).astype("uint8")
    # scipy.misc.imsave(path, image)
    imwrite(path, image, "JPEG", dpi=(300,300))
