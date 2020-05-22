import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, ndarray as nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import time
from net import resnet18
import os


# 1. build data pipline
batch_size = 64

train_dir = 'datasets/Fer2013/Training'
valid_dir = 'datasets/Fer2013/PrivateTest'
test_dir = 'datasets/Fer2013/PublicTest'

train_ds = gdata.vision.ImageFolderDataset(train_dir, flag=0)
valid_ds = gdata.vision.ImageFolderDataset(valid_dir, flag=0)
test_ds = gdata.vision.ImageFolderDataset(test_dir, flag=0)

transform_train = gdata.vision.transforms.Compose([
    gdata.vision.transforms.RandomFlipLeftRight(),
    gdata.vision.transforms.ToTensor(),
    # 对图像的每个通道做标准化
    gdata.vision.transforms.Normalize([0.5],
                                      [0.5])])

transform_test = gdata.vision.transforms.Compose([
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.5],
                                      [0.5])])

train_iter = gdata.DataLoader(train_ds.transform_first(transform_train),
                              batch_size, shuffle=True, last_batch='keep')
valid_iter = gdata.DataLoader(valid_ds.transform_first(transform_test),
                              batch_size, shuffle=True, last_batch='keep')
test_iter = gdata.DataLoader(test_ds.transform_first(transform_test),
                             batch_size, shuffle=False, last_batch='keep')

# 2. define training
ctx = mx.gpu(0)
net = resnet18(num_classes=7)
net.initialize(ctx=ctx, init=init.Xavier())
net.hybridize()  # dynamic -> static

num_epochs = 100
init_lr = 0.01
wd = 5e-4
lr_period = 10
lr_decay = 0.5

load_params = True

trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': init_lr, 'momentum': 0.9, 'wd': wd})

loss = gloss.SoftmaxCrossEntropyLoss()

# start training
param_path = 'trained_models/resnet18-epoch15-loss0.19167047693145667.params'
if os.path.exists(param_path) and load_params:
    net.load_parameters(param_path)

best_valid_acc = 0
for epoch in range(num_epochs):
    train_losses = 0
    train_accuracy = 0
    num_samples = 0
    start_time = time.time()

    # lr decay
    if epoch > 0 and epoch % lr_period == 0:
        trainer.set_learning_rate(trainer.learning_rate * lr_decay)

    for X, y in train_iter:
        y = y.astype('float32').as_in_context(ctx)
        with autograd.record():  # tell mxnet to record
            y_hat = net(X.as_in_context(ctx))
            l = loss(y_hat, y).sum()
        l.backward() # backward
        trainer.step(batch_size) # update params for a batch
        train_losses += l.asscalar()
        train_accuracy += (y_hat.argmax(axis=1) == y).sum().asscalar()  # have to be divide by num_samples
        num_samples += y.size

    epoch_time = "time %.2f sec" % (time.time() - start_time)  # cal time for train a epoch

    # valid per epoch
    if valid_iter is not None:
        epoch_valid_acc = d2l.evaluate_accuracy(valid_iter, net, ctx)

    # printing
    epoch_loss = train_losses / num_samples
    epoch_train_acc = train_accuracy / num_samples
    print("epoch: {}, training loss: {}, training acc: {}, validation acc: {}"
          .format(epoch, epoch_loss, epoch_train_acc, epoch_valid_acc))

    # test per 5 epochs and save params
    if epoch % 5 == 0 and test_iter is not None:
        epoch_test_loss = d2l.evaluate_accuracy(test_iter, net, ctx)
        print("test at epoch {}, test acc: {}".format(epoch, epoch_test_loss))
        net.save_parameters('trained_models/resnet18-epoch{}-loss{}.params'.format(epoch, epoch_loss))

    # save best params
    if epoch_valid_acc > best_valid_acc and epoch_valid_acc > 0.5:
        print('save to best!')
        best_valid_acc = epoch_valid_acc
        net.save_parameters('trained_models/resnet18_best_valid.params')
