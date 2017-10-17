import tensorflow as tf
import model_Autoencoder as an
import numpy as np
import utils
import os

def load_variable():
    # Build model/graph then load variables
    tf.reset_default_graph()
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    model = an.Autoencoder()
    model.build_model()
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(model.init)
    saver = ut.initialize(sess, resume=True)
    return sess, model.decoded, model.train

###################################################################################
###### Load graph without building the model first
def load_graph(meta_file, checkpoint):
    # Load graph and variable at once
    tf.reset_default_graph()
    sess = tf.Session()
    saver = tf.train.import_meta_graph(meta_file)
    # saver.restore(sess, tf.train.latest_checkpoint('.\Checkpoints'))
    saver.restore(sess, checkpoint)

    graph = tf.get_default_graph()
    decoded = tf.get_collection("predict")
    #decoded = graph.get_operation_by_name("decoded")
    # Get placeholder tensor within the graph
    # Tensor names must be of the form "<op_name>:<output_index>"
    # The graph only accepts a holder with the same name defined in the original graph
    # Creating a new place holder with the same name won't work, because the graph will rename the
    # accepted placeholder name when seeing another variable with the same name
    train = graph.get_tensor_by_name("train:0")
    # output = sess.run(decoded, feed_dict={train: image})
    # The output has five dimensions somehow
    # output = np.array(output)
    # output = output.squeeze(axis=0)
    # Remove the first dimension
    return sess, decoded, train

def scale(image, target):
    threshold = -3.9
    bg = image[image<threshold].mean()
    image -= bg

    fg_image = image.mean()
    fg_target = target.mean()

    if (fg_image) != 0:
        ratio = fg_target / (fg_image)
        image *= ratio
    return image

if __name__ == "__main__":
    ut = utils.utils()
    ut.init_file_directory()

    # patch_validation = ut.normalize(ut.get_validation_image(ut.patch_height, ut.patch_width, 1))
    # output = load_graph(patch_validation)
    # image = output.reshape([ut.patch_height, ut.patch_width])

    normalized_full_low_dose_dir = ut.OUTPUT_DIR + "\\LowDose\\"
    normalized_full_normal_dose_dir = ut.OUTPUT_DIR + "\\NormalDose\\"
    output_dir = ut.OUTPUT_DIR + "\\Results\\"
    if not os.path.exists(normalized_full_low_dose_dir):
        os.makedirs(normalized_full_low_dose_dir)
    if not os.path.exists(normalized_full_normal_dose_dir):
        os.makedirs(normalized_full_normal_dose_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    meta_files, checkpoints = ut.get_checkpoint_list(".\Checkpoints/")
    sess, predict, train = load_graph(meta_files[-1], checkpoints[-1])

    n_files = len(ut.validation_low_dose_files)
    mean = -100
    std = 45
    for i in range(n_files):

        name = ut.extract_filename(ut.validation_low_dose_files[i])

        normalized_full_normal_dose_file = normalized_full_normal_dose_dir + name + ".img"
        full_normal_dose = ut.get_full_image(ut.validation_normal_dose_files, i, 1)
        full_normal_dose = np.clip(full_normal_dose, -300, 1000)
        full_normal_dose = (full_normal_dose - mean) / std
        full_normal_dose.tofile(normalized_full_normal_dose_file)

        normalized_full_low_dose_file = normalized_full_low_dose_dir + name + ".img"
        full_low_dose = ut.get_full_image(ut.validation_low_dose_files, i, 1)
        full_low_dose = np.clip(full_low_dose, -300, 1000)
        full_low_dose = (full_low_dose - mean) / std
        full_low_dose.tofile(normalized_full_low_dose_file)

        # subtraction_file = "D:\DeepLearning\Outputs/Subtraction/" + name + ".img"
        # subtraction = full_low_dose - full_normal_dose
        # subtraction.tofile(subtraction_file)

        output = sess.run(predict, feed_dict={train: full_low_dose})
        output = np.array(output).squeeze(axis=0)
        output = output.reshape([ut.full_height, ut.full_width])

        # file = "D:\DeepLearning\Outputs\Results_Model2\KBCT110001L_CBCT{:03d}.img".format(i+1)
        # image = ut.read_image(file, ut.full_height, ut.full_width, dtype='float32')
        # output = scale(output, image)
        output_file = output_dir + name + ".img"
        output.tofile(output_file)

        print(output_file)
