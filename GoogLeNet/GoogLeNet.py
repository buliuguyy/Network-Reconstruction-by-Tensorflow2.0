import tensorflow as tf
from d2l import tensorflow as d2l

# 由于Inception块有concat的操作，因此设计成class来实现
# Inception块从四条路径从不同层面抽取信息，然后在输出通道维合并
class Inception(tf.keras.Model):
    # c1--c4是每条路径的输出通道数
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        # 线路1，单1x1卷积层
        self.p1_1 = tf.keras.layers.Conv2D(c1, 1, activation='relu')
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = tf.keras.layers.Conv2D(c2[0], 1, activation='relu')
        self.p2_2 = tf.keras.layers.Conv2D(c2[1], 3, padding='same',
                                           activation='relu')
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = tf.keras.layers.Conv2D(c3[0], 1, activation='relu')
        self.p3_2 = tf.keras.layers.Conv2D(c3[1], 5, padding='same',
                                           activation='relu')
        # 线路4，3x3最大池化层后接1x1卷积层
        self.p4_1 = tf.keras.layers.MaxPool2D(3, 1, padding='same')
        self.p4_2 = tf.keras.layers.Conv2D(c4, 1, activation='relu')


    def call(self, x):
        # 和Pytorch不同, Tensorflow的activation function必须写在卷积层中, 所以这里没有额外添加ReLU激活层
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        # 在通道维度上连结输出
        return tf.keras.layers.Concatenate()([p1, p2, p3, p4])

# GoogLeNet总体分成5块架构，依次实现
# 第一块使用64个通道、 7×7 卷积层
def b1():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 7, strides=2, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])

# 第二个模块使用两个卷积层：第一个卷积层是64个通道、 1×1 卷积层；第二个卷积层使用将通道数量增加三倍的 3×3 卷积层
def b2():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 1, activation='relu'),
        tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])

# 第三个模块串联两个完整的Inception块。 第一个Inception块的输出通道数为 64+128+32+32=256 ，
#  四个路径之间的输出通道数量比为 64:128:32:32=2:4:1:1 。 第二个和第三个路径首先将输入通道的
#  数量分别减少到 96/192=1/2 和 16/192=1/12 ，然后连接第二个卷积层。第二个Inception块的输出通道数增加到 128+192+96+64=480 ，
#  四个路径之间的输出通道数量比为 128:192:96:64=4:6:3:2 。 第二条和第三条路径首先将输入通道的数量分别减少到 128/256=1/2 和 32/256=1/8
def b3():
    return tf.keras.models.Sequential([
        Inception(64, (96, 128), (16, 32), 32),
        Inception(128, (128, 192), (32, 96), 64),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])

# 第四模块更加复杂， 它串联了5个Inception块，其输出通道数分别是 192+208+48+64=512 、 160+224+64+64=512 、 
#  128+256+64+64=512 、 112+288+64+64=528 和 256+320+128+128=832 。 这些路径的通道数分配和第三模块中的类似，
#  首先是含 3×3 卷积层的第二条路径输出最多通道，其次是仅含 1×1 卷积层的第一条路径，之后是含 5×5 卷积层的
#  第三条路径和含 3×3 最大汇聚层的第四条路径。 其中第二、第三条路径都会先按比例减小通道数。 这些比例在各个Inception块中都略有不同
def b4():
    return tf.keras.Sequential([
        Inception(192, (96, 208), (16, 48), 64),
        Inception(160, (112, 224), (24, 64), 64),
        Inception(128, (128, 256), (24, 64), 64),
        Inception(112, (144, 288), (32, 64), 64),
        Inception(256, (160, 320), (32, 128), 128),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])

# 第五模块包含输出通道数为 256+320+128+128=832 和 384+384+128+128=1024 的两个Inception块。 其中每条路径通道数的分配思路
#  和第三、第四模块中的一致，只是在具体数值上有所不同。 需要注意的是，第五模块的后面紧跟输出层，该模块同NiN一样使用全局平均汇聚层，
#  将每个通道的高和宽变成1。 最后我们将输出变成二维数组，再接上一个输出个数为标签类别数的全连接层
def b5():
    return tf.keras.Sequential([
        Inception(256, (160, 320), (32, 128), 128),
        Inception(384, (192, 384), (48, 128), 128),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Flatten()
    ])

# “net”必须是一个将被传递给“d2l.train_ch6（）”的函数。
# 为了利用我们现有的CPU/GPU设备，这样模型构建/编译需要在“strategy.scope()”
def GoogLeNet():
    return tf.keras.Sequential([b1(), b2(), b3(), b4(), b5(),
                                tf.keras.layers.Dense(10)])

X = tf.random.uniform(shape=(1, 96, 96, 1))
for layer in GoogLeNet().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
