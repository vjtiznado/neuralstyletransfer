import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.pyplot import imread
from matplotlib.pyplot import show
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
import pprint
from tqdm import tqdm
import cv2
from time import sleep
# %matplotlib inline


def compute_content_loss(a_C, a_G):
    """
    Computes the content loss between the Content image and the
    Generated image. This refers to the actual content of the image.

    arguments:
        a_C -- tensor of dimension (1, n_H, n_H, n_C).
                hidden layer activations representing content of the image C
        a_G -- tensor of dimension (1, n_H, n_W, n_C).
                hidden layer activations representing content of the image G

    returns:
        J_content -- scalar that you compute using loss equation
    """
    # retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # reshape a_C and a_G
    a_C_unrolled = tf.reshape(a_C, shape=[m, -1, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[m, -1, n_C])

    # compute the cost with tensorflow
    J_content = tf.reduce_sum(tf.square(
    tf.subtract(a_C_unrolled, a_G_unrolled)))/(4*n_H*n_W*n_C)

    return J_content


def gram_matrix(A):
    """
    Generates the Gram matrix, which refers to the "style matrix".

    Since the Style of an image is represented as how correlated are the
    different filter features of an image. For computing the Style Loss
    we have to measure those correlations. This is made by constructing
    the Gram matrix, which multuplies the values of all the different
    pairs of features to get an idea of how correlated they are.
    For example, a Gij value measures how similar the activations of
    filter i are to the activations of filter j, and so on.

    arguments:
        A -- matrix of shape (n_C, n_H*n_W). The "unrolled" filter matrix
            (where each row contains all the values of a given filter)

    returns:
        GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    GA = tf.matmul(A, tf.transpose(A))

    return GA


def compute_layer_style_cost(a_S, a_G):
    """
    Computes the style loss between the style image S and the gram matrix G
    of the Generated image.
    This will be done for a single layer

    arguments:
        a_S -- tensor of dimension (1, n_H, n_W, n_C)
                hidden layer activations representing style of the image S
        a_G -- tensor of dimension (1, n_H, n_W, n_C)
                hidden layer activations representing style of the image G

    returns:
        J_style_layer -- tensor representing a scalar value, the style cost
    """
    # retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # reshape the images to have them of shape (n_C, n_H*n_W)
    a_S = tf.reshape(tf.transpose(a_S, [0,3,1,2]), shape=[n_C, n_H*n_W])
    a_G = tf.reshape(tf.transpose(a_G, [0,3,1,2]), shape=[n_C, n_H*n_W])

    # computing gram_matrices for both images S and G
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # compute the style loss
    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG)))
    J_style_layer /= (2*n_C*n_W*n_H)**2

    return J_style_layer


def compute_style_cost(model, STYLE_LAYERS):
    """
    Compuyes the overall style cost from several chosen layers.

    arguments:
        model -- our tensorflow model
        STYLE_LAYERS -- A python list containing:
                            - the names of the layer we would like to extract
                                the style from
                            - a coefficient for each of them (lambda)

    returns:
        J_style -- tensor representing a scalar value, style cost
    """
    # initialize the overall style cost
    J_style = 0

    for layer_name, lambd in STYLE_LAYERS:
        # select the output tensor of the currently selected layer
        out = model[layer_name]

        # set a_S to be the hidden layer activation from the layer we have
        # selected, by running the session on out
        a_S = sess.run(out)

        # set a_G to be hidden layer activation from same layer. Here, a_G
        # references model[layer_name] and isn't evaluated yet. Later in the
        # code, we'll assign the image G as the model input, so that when we
        # run the session, this will be the activations drawn from the
        # appropiate layer, with G as input.
        a_G = out

        # compute style cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff*J of this layer to overall style cost
        J_style += lambd * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha=10, beta=40):
    """
    Computes the total cost function, which is the sum of the content loss
    and the style loss, each of them having a defined weight (alpha and beta).

    arguments:
        J_content -- content cost
        J_style -- style cost
        alpha -- hyperparameter weighting the importance of the content loss
        beta -- hyperparameter weighting the importance of the style loss

    returns:
        J -- total cost
    """
    J = alpha*J_content + beta*J_style

    return J


def initialize_model(sess, model, content_image, style_image):
    """
    Initializes the model by processing content and style images, creating
    the generated image, and computing its activations and losses.

    arguments:
        content_frame -- numpy array of the content image
        style_image -- numpy array of the style image

    returns:
        sess -- tensorflow session after initialization
        content_image -- content image for being run in the model
    """
    # process content and style images
    content_image = reshape_and_normalize_image(content_image)
    style_image = reshape_and_normalize_image(style_image)

    # initialize a "generated" image slightly correlateed with content_image
    generated_image = generate_noise_image(content_image)

    # 1. compute content loss
    sess.run(model["input"].assign(content_image))  # content image on the model
    out = model["conv4_2"]  # define the layer for computing the activations
    a_C = sess.run(out)  # compute the activations for the content image
    a_G = out  # define generated image's activations for being the same layer
    J_content = compute_content_loss(a_C, a_G)

    # 2. compute the style loss
    sess.run(model["input"].assign(style_image))  # run style image on the model
    J_style = compute_style_cost(model, STYLE_LAYERS)  # compute the cost

    # 3. compute the total cost
    J = total_cost(J_content, J_style, alpha=10, beta=40)

    # 4. run the optimizer
    optimizer = tf.train.AdamOptimizer(2.0)  # define the optimizer
    train_step = optimizer.minimize(J)  # define the training operation

    return sess, generated_image, train_step


def model_nn(sess, input_image, train_step, count, nframes,
             num_iterations=200, output_dir="./"):
    # initialize global variables (you need to run the session on the
    # initializer)
    sess.run(tf.global_variables_initializer())

    # run the noisy input image (initial generated image) through the model.
    # Use assign()
    sess.run(model["input"].assign(input_image))

    clear = lambda: os.system("clear")
    clear()
    print("running the model on frame " + str(count)+"/"+str(nframes)+" ...\n")
    for i in tqdm(range(num_iterations)):
        # Run the session on the train_step to minimize the total cost
        sess.run(train_step)

        # Compute the generated image by running the session on the current model["input"]
        generated_image = sess.run(model["input"])

    print("model run successfully.\n")
    save_image(output_dir + "/generated_image.jpg", generated_image)

    return generated_image


if __name__ == "__main__":
    # Reset the graph and create a tensorflow's interactive session
    # tf.reset_default_graph()
    # sess = tf.InteractiveSession()
    # create the folder where all the output frames will be saved
    output_folder = create_output_folder(sys.argv[1], sys.argv[2])

    # load the VGG19 model
    # model = load_vgg_model("./imagenet-vgg-verydeep-19.mat")

    # define the layers and its contributions (lambda) for computing the style loss
    STYLE_LAYERS = [("conv1_1", 0.2),  # (layer_name, lambda)
                    ("conv1_2", 0.2),
                    ("conv2_1", 0.2),
                    ("conv2_2", 0.2),
                    ("conv5_1", 0.2)]
    save_style_layers(output_folder, STYLE_LAYERS)

    # read the style image
    style_image = imread(sys.argv[2])

    # load the video
    video = cv2.VideoCapture(sys.argv[1])
    videoframe_shape = (int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        int(video.get(cv2.CAP_PROP_FRAME_WIDTH)))

    nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    print("\n \n")
    while(video.isOpened()):
        # Reset the graph and create a tensorflow's interactive session
        tf.reset_default_graph()
        sess = tf.InteractiveSession()
        model = load_vgg_model("./imagenet-vgg-verydeep-19.mat")

        still, frame = video.read()
        if still == False:
            break

        # initialize the model
        sess, initialized_image, train_step = initialize_model(sess, model,
                                                               frame, style_image)

        # run the model
        model_nn(sess, initialized_image, train_step, count+1, nframes,
                 300, output_folder)

        # resizing the generated image to the original size of the content image
        generated_image = imread(output_folder + "/generated_image.jpg")
        pil_generated_image = Image.fromarray(np.uint8(generated_image)).convert("RGB")
        resized = pil_generated_image.resize((videoframe_shape[1],
                                              videoframe_shape[0]))
        resized = np.array(resized)
        imwrite(output_folder + "/frame_" + str(count+1).zfill(3) + ".jpg",
                resized, dpi=(300,300))
        count += 1
        sess.close()
    video.release()
    # i need to be able to generate a video after finished
    # optimize zfill to authomatically set the number depending on the number
    # of frames of the video
