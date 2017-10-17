import time
import matplotlib.pyplot as plt
import model_GAN as GAN
from skimage import measure
import numpy as np
import tensorflow as tf
import utils

ut = utils.utils()
ut.init_file_directory()

model = GAN.GAN()
model.build_model()

sess = tf.Session()
sess.run(model.init)
saver = ut.initialize(sess, resume=False)

initial_step = model.global_step.eval(sess)

height = 128
width = 128
validation_low_dose, validation_normal_dose = ut.get_real_time_validation_image(height, width)
validation_low_dose, validation_normal_dose = ut.standardize(validation_low_dose, validation_normal_dose)
validation_number = validation_low_dose.shape[0]
ut.imsave("LowDose", validation_low_dose.reshape([height*validation_number, width]))
ut.imsave("NormalDose", validation_normal_dose.reshape([height*validation_number, width]))

# validation_low_dose = (validation_train - mean) / std

loss = []
start_time = time.time()
for epoch in range(ut.epochs):
    ut.shuffle_patch_files()
    n_batch = ut.patch_file_number // ut.batch_size
    for b in range(n_batch):
        batch_input = ut.get_patch_batch(ut.patch_low_dose_files, b, ut.batch_size)
        batch_target = ut.get_patch_batch(ut.patch_normal_dose_files, b, ut.batch_size)
        batch_input, batch_target = ut.standardize(batch_input, batch_target)
        _, d_loss_cur = sess.run([model.d_solver, model.d_loss], feed_dict={model.train: batch_input, model.target: batch_target})
        # d_loss_cur = 0
        _, g_loss_cur = sess.run([model.g_solver, model.g_loss], feed_dict={model.train: batch_input, model.target: batch_target})

        print("Epoch: {}/{}, Batch: {}/{}".format(epoch + 1, ut.epochs, b + 1, n_batch),
             "D loss: {:.8f}, G loss: {:.8f}".format(d_loss_cur, g_loss_cur))
        loss.append(g_loss_cur)

        # save check point
        if (b + 1) % ut.CKPT_STEP == 0:
            # `save` method will call `export_meta_graph` implicitly and save the graph
            saver.save(sess, ut.CKPT_DIR, epoch * n_batch + b)
            image = sess.run(model.Gz, feed_dict={model.train: validation_low_dose})
            image = image.reshape([height * validation_number, width])
            ut.imsave('val_{}_{}'.format(epoch + 1, b + 1), image)

            # Peak signal-to-noise ratio:
            # ratio between the maximum possible power of a signal and the power of corrupting noise
            psnr = measure.compare_psnr(validation_normal_dose.reshape([height * validation_number, width]), image,
                                        data_range=1)
            # Structural similarity, [-1, 1], 1 means identical
            ssim = measure.compare_ssim(validation_normal_dose.reshape([height * validation_number, width]), image,
                                        data_range=1, win_size=9)
            print("PSNR: {}, SSIM: {}...".format(psnr, ssim))
            np.array(loss).tofile(ut.OUTPUT_DIR + "loss.img")

end_time = time.time()
print("Elipsed time: {}".format(end_time-start_time))

loss = ut.read_image(ut.OUTPUT_DIR+"loss.img", 1, 30882, dtype='float32')
plt.plot(loss[0][:30000])