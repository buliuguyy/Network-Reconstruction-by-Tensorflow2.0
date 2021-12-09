import tensorflow as tf

# NiN 中的mlpconv块
def mlpconv(num_channels, kernel_size, strides, padding):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(num_channels, kernel_size, strides=strides,
                               padding=padding, activation='relu'),
        # 卷积层后接两层1×1的卷积层，用于添加非线性并帮助特征提取
        tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                               activation='relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                               activation='relu')]
    )

def NiN_Net():
    return tf.keras.models.Sequential([
        # 先是三层mlpconv的叠加
        mlpconv(96, kernel_size=11, strides=4, padding='valid'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        mlpconv(256, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        mlpconv(384, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # 原论文中有提到加了dropout之后准确率会上升。。。
        tf.keras.layers.Dropout(0.5),  
        # 标签类别数是10
        mlpconv(10, kernel_size=3, strides=1, padding='same'),
        # 全局平均池化直接转换feature map为categories
        tf.keras.layers.GlobalAveragePooling2D(),  
        tf.keras.layers.Reshape((1, 1, 10)),
        # 将四维的输出转成二维的输出，其形状为(批量大小,10)
        tf.keras.layers.Flatten(),
        ]
    )

# 简单查看网络结构
X = tf.random.uniform((1, 224, 224, 1))
for layer in NiN_Net().layers:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)       
        
        
        
        