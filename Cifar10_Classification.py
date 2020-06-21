import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, nd
from mxnet import gluon, init
from mxnet.gluon import data as gdata, loss as gloss, nn
from mxnet.gluon import utils as gutils
import time
import matplotlib.pyplot as plt
import random
import numpy as np

# 图像增广函数 对训练集做随机翻转处理 对测试集不做处理
# Update 增加更多的操作
flig_aug = gdata.vision.transforms.Compose(
    [
        gdata.vision.transforms.Resize(40),  # 把图片大小从32x32放大到40x40
        gdata.vision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),  # 随机裁剪图片
        gdata.vision.transforms.RandomFlipLeftRight(),
        # gdata.vision.transforms.RandomFlipTopBottom(),
        gdata.vision.transforms.ToTensor(),  # 把图像转换成训练所需要的格式
        # gdata.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],  # 对通道进行标准化
        #                                   [0.2023, 0.1994, 0.2010])
    ]
)
no_aug = gdata.vision.transforms.Compose(
    [
        gdata.vision.transforms.ToTensor(),
        # gdata.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
        #                                   [0.2023, 0.1994, 0.2010])
    ]
)

# 辅助函数
def try_all_gpus():
    ctxes = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctxes.append(ctx)
    except mx.base.MXNetError:
        pass
    if not ctxes:
        ctxes = [mx.cpu()]
    return ctxes

def get_batch(data, ctx):  # 把数据批量放进GPU
    features, labels = data
    # ctx = [ctx]  # 是否需要
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return gutils.split_and_load(features, ctx), gutils.split_and_load(labels, ctx), features.shape[0]


def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):  # 计算模型准确率
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        features, labels, _ = get_batch(batch, ctx)
        for x, y in zip(features, labels):
            y = y.astype('float32')
            acc_sum += (net(x).argmax(axis=1) == y).sum().copyto(mx.cpu()) # cpu?
            n += y.size
        acc_sum.wait_to_read()  # 异步计算 等待
    return acc_sum.asscalar() / n


# 模型
class Residual(nn.HybridBlock):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def hybrid_forward(self, F, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


def get_resnet18(num_classes):
    net = nn.HybridSequential()
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))

    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.HybridSequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(Residual(num_channels))
        return blk

    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net

def Train(load_Params=False):
    ctx = try_all_gpus()
    net = get_resnet18(10)
    net.initialize(init=init.Xavier(), ctx=ctx)

    if load_Params is False:
        batch_size = 128
        lr = 0.001
        num_epochs = 1  # 降低到3以便演示
        train_iter = gdata.DataLoader(gdata.vision.CIFAR10(train=True).transform_first(flig_aug),
                                      batch_size=batch_size, shuffle=True)
        loss = gloss.SoftmaxCrossEntropyLoss()  # 交叉熵损失函数
        trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
        train(train_iter, net, loss, trainer, ctx, num_epochs, batch_size)
    else:
        filename = 'CIFAR-10.params'
        net.load_parameters(filename)

    return net

# 训练
def train(train_iter, net, loss, trainer, ctx, num_epochs, batch_size):
    print('Training on', ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum = 0.0, 0.0
        n = 0
        start = time.time()

        for batch in train_iter:
            xlist, ylist, size = get_batch(batch, ctx)
            loss_list = []
            with autograd.record():
                y_hats = [net(x) for x in xlist]
                loss_list = [loss(y_hat, y) for y_hat, y in zip(y_hats, ylist)]
            for l in loss_list:
                l.backward()
            trainer.step(batch_size)
            train_loss_sum += sum([ll.sum().asscalar() for ll in loss_list])
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar() for y_hat, y in zip(y_hats, ylist)])
            n += batch_size

        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
              % (epoch + 1, train_loss_sum / n, train_acc_sum / n,
                 time.time() - start))


def Test(net):
    ctx = try_all_gpus()
    test_iter = gdata.DataLoader(gdata.vision.CIFAR10(train=False).transform_first(no_aug),
                                 batch_size=128, shuffle=False)
    test_acc = evaluate_accuracy(test_iter, net, ctx)
    print('test acc %.3f' % test_acc)

    ctx2 = mx.gpu();
    for show_x, show_y in test_iter:
        show_x = show_x.as_in_context(ctx2)
        show_y = show_y.as_in_context(ctx2)
        break

    text_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    size = 8
    valid_index = random.sample(range(0, 128), size)
    true_y = [text_labels[int(i)] for i in show_y.asnumpy()[valid_index]]
    pred_y = [text_labels[int(i)] for i in net(show_x).argmax(axis=1).asnumpy()[valid_index]]

    titles = ['True: ' + true + '\n' + 'Pred: ' + pred for true, pred in zip(true_y, pred_y)]
    image = []

    for i in valid_index:
        image.append(show_x[i])

    show_cifar10(image, titles[0:size])
    plt.show()


def show_cifar10(images, labels):
    _, figs = plt.subplots(1, len(images), figsize=(15, 6))
    i = 0
    for f, lbl in zip(figs, labels):
        f.imshow(np.transpose(images[i].asnumpy(), [1, 2, 0]))  # 因为数据读入的时候为了训练而改变了维度 需要改回32x32x3
        i += 1
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)


def GetImage():
    test_iter_shuffle = gdata.DataLoader(gdata.vision.CIFAR10(train=False).transform_first(no_aug),
                                         batch_size=128, shuffle=True)
    ctx = mx.gpu()
    for show_x, show_y in test_iter_shuffle:
        show_x = show_x.as_in_context(ctx)
        show_y = show_y.as_in_context(ctx)
        break
    valid_index = random.sample(range(0, 128), 1)
    image = show_x[valid_index[0]]
    result = show_y[valid_index[0]].asscalar()

    return image, result


def Predict(net, X):
    y_hat = net(X.reshape(1, 3, 32, 32))

    return y_hat.argmax(axis=1).asnumpy()[0]

