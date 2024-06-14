import tempfile
import tensorflow as tf


# _temp_dir = tempfile.mkdtemp()
# path_to_flower_dataset = tf.keras.utils.get_file(
#     "flower_photos",
#     "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
#     untar=True,
#     cache_dir=_temp_dir,
# )

import os, glob
import numpy as np
import cv2  # The dependencies need to be installed manually, pip install opencv-python

# root = path_to_flower_dataset
root = './datasets/flower_photos'

classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
img_paths = []  # Used to save all picture paths
labels = []  # Used to save the picture category tags,(0,1,2,3,4)
for i, label in enumerate(classes):
    cls_img_paths = glob.glob(os.path.join(root, label, "*.jpg"))
    img_paths.extend(cls_img_paths)
    labels.extend([i] * len(cls_img_paths))

# image->numpy
img_numpys = []
labels = np.array(labels)
for img_path in img_paths:
    img_numpy = cv2.imread(img_path)
    img_numpy = cv2.resize(img_numpy, (240, 240))
    img_numpy = np.reshape(img_numpy, (1, 240, 240, 3))
    # If use Pytorch backend dimension should be exchanged
    # img_numpy = np.transpose(img_numpy, (0,3,1,2))
    img_numpys.append(img_numpy)

images = np.concatenate(img_numpys, axis=0)
print(images.shape)
print(labels.shape)

# Distribute images and labels to two nodes, allocating 50% of the data to each node.
per = 0.5
alice_images = images[: int(per * images.shape[0]), :, :, :]
alice_label = labels[: int(per * images.shape[0])]
bob_images = images[int(per * images.shape[0]) :, :, :, :]
bob_label = labels[int(per * images.shape[0]) :]
print(f"alice images shape = {alice_images.shape}, alice labels shape = {alice_label.shape}")
print(f"bob images shape = {bob_images.shape}, bob labels shape = {bob_label.shape}")

# Save the data as npz files separately, and then send them to the two machines.
np.savez("flower_alice.npz", image=alice_images, label=alice_label)
np.savez("flower_bob.npz", image=bob_images, label=bob_label)
