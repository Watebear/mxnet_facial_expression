import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon import data as gdata
import cv2
from net import resnet18


img = cv2.imread('test_imgs/2.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (48, 48))
print(img.shape)
img = nd.array(img).expand_dims(axis=2).expand_dims(axis=0)
print(img.shape)

transform_test = gdata.vision.transforms.Compose([
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.5],
                                      [0.5])])

img = transform_test(img)
net = resnet18(7)
ctx = mx.gpu(0)
net.load_parameters('trained_models/resnet18-epoch10-loss0.008498244381253217.params')

pred = net(img)[0]
idx = nd.argmax(pred, axis=0)
print(nd.argmax(pred, axis=0))
