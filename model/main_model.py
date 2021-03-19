import os
import numpy as np
import tensorflow as tf
import model.guided_filter
import model.network

def cartoonize(image):
    # Define path of folders
    model_path = 'model/saved_models'

    # set cpu mode
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
    network_out = model.network.unet_generator(input_photo)
    final_out = model.guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.train.Saver(var_list=gene_vars)
    
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    # Now we apply the model
    image = model.guided_filter.resize_crop(image)
    batch_image = image.astype(np.float32)/127.5 - 1
    batch_image = np.expand_dims(batch_image, axis=0)
    output = sess.run(final_out, feed_dict={input_photo: batch_image})
    output = (np.squeeze(output)+1)*127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    sess.close()
    return output


if __name__ == '__main__':
    model_path = 'saved_models/vgg19_no_fc.npy'
    load_folder = 'test_images'
    save_folder = 'cartoonized_images'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    cartoonize(load_folder, save_folder, model_path)
    