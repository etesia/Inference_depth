import numpy as np
import tensorflow as tf
import model
import cv2
import matplotlib.pyplot as plt

train_model = model.Model(img_height=None, img_width=None, batch_size=None, is_training=False)

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, 'cityscapes_kitti_learned_intrinsics/model-1000977')


image = cv2.imread('./0000000189.png')[:,:,::-1]
inputs = cv2.resize(image, (416, 128))[np.newaxis,:]
depth = train_model.inference_depth(inputs, sess)
depth = depth[0,:,:,:]
depth = cv2.resize(depth, (image.shape[1], image.shape[0]))


plt.figure(figsize=(20, 10))
plt.imshow(image)
plt.axis('off')
plt.savefig("image.png", bbox_inches='tight', pad_inched=0)
plt.show()

plt.figure(figsize=(20, 10))
plt.imshow(1/depth[:, :], cmap='plasma')
plt.axis('off')
plt.savefig("res.png", bbox_inches='tight', pad_inched=0)
plt.show()

