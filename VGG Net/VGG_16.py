import tensorflow as tf

# VGG卷积块, 由于VGG-16原论文中使用了1×1的卷积层, 因此这边加了一个flag(last_1_1_kernel)来判断是否加入这层
def vgg_block(num_convs, num_channels, last_1_1_kernel):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(tf.keras.layers.Conv2D(num_channels,kernel_size=3,
                                    padding='same',activation='relu'))
    if last_1_1_kernel:  # 如果最后一层为1×1的卷积层
        blk.add(tf.keras.layers.Conv2D(num_channels,kernel_size=1,activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk

conv_arch = ((1, 64, False), (1, 128, False), (2, 256, True), (2, 512, True), (2, 512, True))
def vgg_16(conv_arch):
    net = tf.keras.models.Sequential()
    # 卷积层部分
    for (num_convs, num_channels, last_1_1_kernel) in conv_arch:
        net.add(vgg_block(num_convs, num_channels, last_1_1_kernel))
    # 全连接层部分
    net.add(tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1000),
        tf.keras.layers.Softmax()  # 原论文末尾提到了使用softmax作为网络的最后一层
    ]))
    return net

net = vgg_16(conv_arch)

# 简单查看网络的信息
X = tf.random.uniform((1, 224, 224, 1))
for blk in net.layers:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t', X.shape)
    
    
    
    
    