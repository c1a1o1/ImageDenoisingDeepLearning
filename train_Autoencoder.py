import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import model_Autoencoder as an
import utils
import PIL
from skimage import measure

ut = utils.utils()
ut.init_file_directory()

model = an.Autoencoder()
model.build_model()

sess = tf.Session()
sess.run(model.init)
saver = ut.initialize(sess, resume=False)
# Finalize graph in case it's modified later
# Graph grows when using tf's image standardization
# tf.get_default_graph().finalize()
# mean = -120
# std = 45
height = 128
width = 128

validation_low_dose, validation_normal_dose = ut.get_real_time_validation_image(height, width)
validation_low_dose, validation_normal_dose = ut.standardize(validation_low_dose, validation_normal_dose)
validation_number = validation_low_dose.shape[0]
ut.imsave("LowDose", validation_low_dose.reshape([height*validation_number, width]))
ut.imsave("NormalDose", validation_normal_dose.reshape([height*validation_number, width]))

# validation_low_dose = (validation_train - mean) / std

loss = []
for epoch in range(ut.epochs):
    ut.shuffle_patch_files()
    n_batch = ut.patch_file_number // ut.batch_size
    for b in range(n_batch):
        batch_input = ut.get_patch_batch(ut.patch_low_dose_files, b, ut.batch_size)
        batch_target = ut.get_patch_batch(ut.patch_normal_dose_files, b, ut.batch_size)

        # batch_input = ut.clip(batch_input)
        # batch_target = ut.clip(batch_target)

        # Standardizing per image uses different mean and std for different images
        # Resulting in non uniform backgrounds
        batch_input, batch_target = ut.standardize(batch_input, batch_target)

        # batch_input = (batch_input - mean) / std
        # batch_target = (batch_target - mean) / std

        batch_cost, _ = sess.run([model.cost, model.opt], feed_dict={model.train: batch_input,
                                                         model.target: batch_target})

        print("Epoch: {}/{}, Batch: {}/{}".format(epoch + 1, ut.epochs, b+1, n_batch),
              "Training loss: {:.8f}".format(batch_cost))

        loss.append(batch_cost)

        # save check point
        if (b + 1) % ut.CKPT_STEP == 0:
            # `save` method will call `export_meta_graph` implicitly and save the graph
            saver.save(sess, ut.CKPT_DIR, epoch * n_batch + b)
            image = sess.run(model.decoded, feed_dict={model.train: validation_low_dose})
            image = image.reshape([height*validation_number, width])
            ut.imsave('val_{}_{}'.format(epoch + 1, b + 1), image)

            # image.tofile(ut.OUTPUT_DIR+'val_{}_{}.img'.format(epoch + 1, b + 1))
            # Peak signal-to-noise ratio:
            # ratio between the maximum possible power of a signal and the power of corrupting noise
            psnr = measure.compare_psnr(validation_normal_dose.reshape([height*validation_number, width]), image, data_range=1)
            # Structural similarity, [-1, 1], 1 means identical
            ssim = measure.compare_ssim(validation_normal_dose.reshape([height*validation_number, width]), image, data_range=1, win_size=9)
            print("PSNR: {}, SSIM: {}...".format(psnr, ssim))

            np.array(loss).tofile(ut.OUTPUT_DIR+"loss.img")

loss = ut.read_image(ut.OUTPUT_DIR+"loss.img", 1, 30882, dtype='float32')
plt.plot(loss[0][:30000])