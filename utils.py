import os
import scipy.misc
import numpy as np
import tensorflow as tf
import random

class utils(object):
    def __init__(self):
        # input matrix format [batch, height, width, channels]
        self.full_width = 1024
        self.full_height = 1024
        self.patch_width = 64
        self.patch_height = 64
        self.channels = 1
        self.epochs = 50
        self.batch_size = 128
        self.CKPT_STEP = 50
        self.CKPT_DIR = './Checkpoints/'
        self.GRAPH_DIR = './Graphs/'
        self.OUTPUT_DIR = 'D:/DeepLearning/Outputs/'
        self.real_time_validation_patch = 'D:/DeepLearning/ValidationImages/'
        self.patch_low_dose_path = "D:\DeepLearning\KBCT-150\Patch_Size064_Stride32_Threshold-180"
        self.patch_normal_dose_path = "D:\DeepLearning\KBCT-300\Patch_Size064_Stride32_Threshold-180"
        self.validation_low_dose_path = "D:\DeepLearning\KBCT-150\Validation_Phantom"
        self.validation_normal_dose_path = "D:\DeepLearning\KBCT-300\Validation_Phantom"

    def init_file_directory(self):
        extension = ['.img', '.raw']
        self.patch_low_dose_files = self.get_files(self.patch_low_dose_path, extension)
        self.patch_normal_dose_files = self.get_files(self.patch_normal_dose_path, extension)
        self.patch_file_number = len(self.patch_normal_dose_files)
        self.validation_low_dose_files = self.get_files(self.validation_low_dose_path, extension)
        self.validation_normal_dose_files = self.get_files(self.validation_normal_dose_path, extension)
        if not os.path.exists(self.CKPT_DIR):
            os.makedirs(self.CKPT_DIR)
        if not os.path.exists(self.OUTPUT_DIR):
            os.makedirs(self.OUTPUT_DIR)

    def get_files(self, path, extension):
        files = []
        for file in os.listdir(path):
            ext = os.path.splitext(file)[-1]
            if ext and ext in extension:
                files.append(os.path.join(path, file))
        return files

    def read_image(self, file, height, width, dtype='int16'):
        shape = (height, width)
        fid = open(file, 'rb')
        data = np.fromfile(fid, np.dtype(dtype))
        fid.close()
        return data.reshape(shape)

    def get_patch_batch(self, files, index, batch_size):
        batch = []
        for i in range(batch_size):
            batch.append(self.read_image(files[index * batch_size + i], self.patch_height, self.patch_width))
        batch = np.array(batch).astype(np.float32)
        return np.reshape(batch, [batch_size, self.patch_height, self.patch_width, self.channels])

    def save_patch_subtraction(self):
        output_dir = self.OUTPUT_DIR + "\\Subtraction\\"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_number = len(self.patch_normal_dose_files)
        for i in range(file_number):
            image = self.get_patch_batch(self.patch_normal_dose_files, i, 1) \
                    - self.get_patch_batch(self.patch_low_dose_files, i, 1)
            file_name = output_dir + self.extract_filename(self.patch_normal_dose_files[i]) + '.img'
            image.tofile(file_name)
            print(file_name)

    def shuffle_patch_files(self):
        combination = list(zip(self.patch_low_dose_files, self.patch_normal_dose_files))
        random.shuffle(combination)
        self.patch_low_dose_files[:], self.patch_normal_dose_files[:] = zip(*combination)

    def get_full_image(self, files, start, batch_size):
        batch = []
        for i in range(batch_size):
            batch.append(self.read_image(files[start + i], self.full_height, self.full_width))
        batch = np.array(batch).astype(np.float32)
        return np.reshape(batch, [batch_size, self.full_height, self.full_width, self.channels])

    def get_real_time_validation_image(self, height, width):
        extension = ['.img', '.raw']
        batch_low_dose = []
        batch_normal_dose = []
        batch_low_dose_size = 0
        batch_normal_dose_size = 0
        files = self.get_files(self.real_time_validation_patch, extension)
        for file in files:
            if "lowdose" in file.lower():
                batch_low_dose_size += 1
                batch_low_dose.append(self.read_image(file, height, width))
            if "normaldose" in file.lower():
                batch_normal_dose_size += 1
                batch_normal_dose.append(self.read_image(file, height, width))
        batch_low_dose = np.array(batch_low_dose).astype(np.float32)
        batch_normal_dose = np.array(batch_normal_dose).astype(np.float32)
        return batch_low_dose.reshape([batch_low_dose_size, height, width, self.channels]),\
            batch_normal_dose.reshape([batch_normal_dose_size, height, width, self.channels])

    def extract_filename(self, file):
        idx1 = file.rfind("\\")
        idx2 = file.rfind(".")
        return file[idx1+1:idx2]

    def standardize(self, image1, image2=np.array([])):
        # Standardize image1 image by image
        # Standardize image2 using image1's means and stds
        batch_size = image1.shape[0]
        minv = -300
        maxv = 200
        image1 = np.clip(image1, minv, maxv)
        if image2.any():
            image2 = np.clip(image2, minv, maxv)
        for i in range(batch_size):
            img = image1[i,:,:,:]
            # Remove background when calculating std and mean for full image
            img = img[img>minv]
            image1[i,:,:,:] = (image1[i,:,:,:]-img.mean()) / img.std()
            if image2.any():
                image2[i, :, :, :] = (image2[i, :, :, :] - img.mean()) / img.std()
        return image1, image2

    def clip(self, image):
        minv = -200
        maxv = 100
        image = np.clip(image, minv, maxv)
        image = (image - minv) / (maxv - minv)
        return image

    def get_checkpoint_list(self, dir=""):
        if not dir:
            dir = self.CKPT_DIR
        extension = '.meta'
        meta_files = self.get_files(dir, extension)
        ckpt_files = [file[:-len(extension)] for file in meta_files]
        return meta_files, ckpt_files

    def denormalize(self, data):
        minv = -300
        maxv = 125
        data = data * (maxv - minv) + minv
        return data

    def initialize(self, sess, resume=False):
        saver = tf.train.Saver(max_to_keep=10)
        # FileWriter creates an event file in a given directory and add summaries and events to it.
        # The class updates the file contents asynchronously.
        writer = tf.summary.FileWriter(self.GRAPH_DIR, sess.graph)
        # Restore checkpoint and continue?
        if resume:
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.CKPT_DIR))
            # ckpt.all_model_checkpoint_paths contains all checkpoints
            # ckpt.model_checkpoint_path contains the latest checkpoints
            if ckpt and ckpt.model_checkpoint_path:
                # Restore variables, it requires a session in which the graph was launched
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("No checkpoint found.")
        return saver

    def imsave(self, filename, image):
        scipy.misc.imsave(self.OUTPUT_DIR+filename+'.png', image)

    def split(self, arr, size):
        arrs = []
        while len(arr) > size:
            pice = arr[:size]
            arrs.append(pice)
            arr = arr[size:]
        arrs.append(arr)
        return arrs


