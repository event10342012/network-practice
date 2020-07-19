import os
import cv2


root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_dir = os.path.join(root, 'data')
data_name = 'chest_xray'

train_dir = os.path.join(data_dir, data_name, 'train')
val_dir = os.path.join(data_dir, data_name, 'val')
test_dir = os.path.join(data_dir, data_name, 'test')

img_path = os.path.join(train_dir, 'NORMAL', 'IM-0115-0001.jpeg')
img = cv2.imread(img_path)

