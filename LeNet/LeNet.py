import tensorflow as tf
import numpy as np
import d2l_tensorflow  as d2l

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='sigmoid'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])


class TrainCallback(tf.keras.callbacks.Callback):  # @save
    """一个以可视化的训练进展的回调。"""

    def __init__(self, net, train_iter, test_iter, num_epochs, device_name):
        self.timer = d2l.Timer()
        self.animator = d2l.Animator(
            xlabel='epoch', xlim=[1, num_epochs], legend=[
                'train loss', 'train acc', 'test acc'])
        self.net = net
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.num_epochs = num_epochs
        self.device_name = device_name

    def on_epoch_begin(self, epoch, logs=None):
        self.timer.start()

    def on_epoch_end(self, epoch, logs):
        self.timer.stop()
        test_acc = self.net.evaluate(
            self.test_iter, verbose=0, return_dict=True)['accuracy']
        metrics = (logs['loss'], logs['accuracy'], test_acc)
        self.animator.add(epoch + 1, metrics)
        if epoch == self.num_epochs - 1:
            batch_size = next(iter(self.train_iter))[0].shape[0]
            num_examples = batch_size * tf.data.experimental.cardinality(
                self.train_iter).numpy()
            print(f'loss {metrics[0]:.3f}, train acc {metrics[1]:.3f}, '
                  f'test acc {metrics[2]:.3f}')
            print(f'{num_examples / self.timer.avg():.1f} examples/sec on '
                  f'{str(self.device_name)}')

# 参数初始化
batch_size, lr, num_epochs = 256, 0.9, 10
resize = None

# 下载Fashion-MNIST数据集并将其载入内存
mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
# 对图片数据进行归一化，并且将输入数据进行升维(在最后加上一个batch_size的维度)，最后将label映射为int32类型
process = lambda X, y: (tf.expand_dims(X, axis=3) / 255, tf.cast(y, dtype='int32'))
# 对图片进行resize，如果必要的话
resize_fn = lambda X, y: (tf.image.resize_with_pad(X, resize, resize) if resize else X, y)
# 将ndarray数据转为tensor，打乱重排并按batch_size大小得到训练集与测试集的迭代器
train_iter = tf.data.Dataset.from_tensor_slices(process(*mnist_train)).batch(batch_size).shuffle(
    len(mnist_train[0])).map(resize_fn)
test_iter = tf.data.Dataset.from_tensor_slices(process(*mnist_test)).batch(batch_size).map(resize_fn)

# 训练过程
device_name = d2l.try_gpu()._device_name  # 获取训练设备
strategy = tf.distribute.OneDeviceStrategy(device_name)
with strategy.scope():
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model = net()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = TrainCallback(model, train_iter, test_iter, num_epochs, device_name)
model.fit(train_iter, epochs=num_epochs, verbose=0, callbacks=[callback])


