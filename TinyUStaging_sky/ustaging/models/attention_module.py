from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, \
    Concatenate, Conv2D, Add, Activation, Lambda, Dropout, SeparableConv2D
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid
import tensorflow as tf

from mpunet.logging import ScreenLogger
from tensorflow.python.keras import regularizers


def attach_attention_module(net, attention_module,
                            ratio_se=8,
                            ratio_cbam=8, data_per_prediction=3840, kernel_size=5,
                            isChanAttn=True, isSeparable=True,
                            isAdd=False, logger=None):
    if logger is None:
        logger = ScreenLogger()
    logger(f"ljy -- attach_attention_module : {attention_module}")

    if attention_module in ['se_block', 'SE', 'se']:  # SE_block
        net = se_block(net, ratio_se)
    elif attention_module in ['cbam_block', 'CBAM', 'cbam']:  # CBAM_block
        net = cbam_block(net, data_per_prediction, kernel_size,
                         isChanAttn, ratio_cbam,
                         isSeparable=isSeparable,
                         isAdd=isAdd, logger=logger)
    else:
        raise Exception("'{}' is not supported attention module!".format(attention_module))

    return net


def se_block(input_feature, ratio=8):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]  # 

    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    assert se_feature.shape[1:] == (1, 1, channel)
    se_feature = Dense(channel // ratio,
                       activation='elu',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature.shape[1:] == (1, 1, channel // ratio)
    se_feature = Dense(channel,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature.shape[1:] == (1, 1, channel)
    if K.image_data_format() == 'channels_first':
        se_feature = Permute((3, 1, 2))(se_feature)

    se_feature = multiply([input_feature, se_feature])
    return se_feature


"""
# TEST:
import numpy as np
input_feature = np.random.rand(8, 31*3840, 1, 22)  # TensorShape([8, 119040, 1, 22])
input_feature = tf.convert_to_tensor(input_feature)
"""

def cbam_block(cbam_feature,  data_per_prediction=3840, kernel_size=5,
               isChanAttn=True, ratio=4,
               isSpaceAttn=True, isSeparable=True,
               isAdd=False,
               logger=None):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    logger(f" ******* CBAM - isChanAttn：{isChanAttn} ********")
    logger(f" ******* CBAM - isSpaceAttn：{isSpaceAttn} ********")

    if not isAdd:
        if isChanAttn:
            cbam_feature = channel_attention(cbam_feature, ratio)  # TensorShape([8, 31*3840, 1, 22])
        if isSpaceAttn:
            cbam_feature = spatial_attention(cbam_feature, kernel_size=kernel_size, isSeparable=isSeparable, logger=logger)
    else:
        cbam_feature1 = cbam_feature2 = 0

        if isChanAttn:
            cbam_feature1 = channel_attention(cbam_feature, ratio)  # TensorShape([8, 31*3840, 1, 22])
        if isSpaceAttn:
            cbam_feature2 = spatial_attention(cbam_feature, kernel_size=kernel_size, isSeparable=isSeparable, logger=logger)

        cbam_feature = cbam_feature1 + cbam_feature2
    return cbam_feature   # TensorShape([8, 31*3840, 1, 22])


def channel_attention(input_feature, ratio=4):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1  # -1: input_shape = [bs=8, H=3840*23/31, W=1, C=22]
    channel = input_feature.shape[channel_axis]  # [3840*31, 1, 22]
    avg_pool = GlobalAveragePooling2D()(input_feature)  # -> GAP: [bs, C=22]
    avg_pool = Reshape((1, 1, channel))(avg_pool)  # [bs=8, H=1, W=1, C=22]

    shared_layer_one = Dense(channel // ratio,  # units=22//4=5
                             activation='elu',  # TODO: TEST RELU不好/ELU更好
                             kernel_initializer='he_normal',
                             use_bias=True,
                             # kernel_regularizer=regularizers.l2(0.001),
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,  # units=22
                             kernel_initializer='he_normal',
                             use_bias=True,
                             # kernel_regularizer=regularizers.l2(0.001),
                             bias_initializer='zeros')


    assert avg_pool.shape[1:] == (1, 1, channel)  # avg_pool.shape[bs, 1, 1, C]
    avg_pool = shared_layer_one(avg_pool)  # [8, 1,1, 22] -> [8, 1,1, 22//4=5]
    assert avg_pool.shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)  # [8, 1,1, 5] -> [8, 1,1, 22]
    assert avg_pool.shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)  # 通道注意力2 # -> GAP: [bs, C=22]
    max_pool = Reshape((1, 1, channel))(max_pool)  # [bs=8, H=1, W=1, C=22]
    assert max_pool.shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])  # [bs=8, H=1, W=1, C=22]
    cbam_feature = Activation('sigmoid')(cbam_feature)  # [bs=8, H=1, W=1, C=22]

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])  # [bs=8, H=3840*31, W=1, C=22] * [bs=8, H=1, W=1, C=22]
                                                 # -> [bs=8, H=3840*31, W=1, C=22]


def spatial_attention(input_feature, kernel_size=5, isSeparable=False, logger=None):
    # # input_feature: [bs=8, H=31*3840, W=1, C=22]
    # # reshape -> (bs=8, H=n_periods=23/31, W=data_per_prediction=3840, C=22)
    kernel_size = (kernel_size, 1)
    ori_shape = input_feature.get_shape().as_list()
    # to_shape = (ori_shape[0],
    #             int(ori_shape[1] // data_per_prediction),
    #             data_per_prediction,
    #             ori_shape[-1])
    # # input_feature = tf.reshape(input_feature, to_shape)  # [bs=8, H=31, W=3840, C=22]
    # input_feature = Reshape(to_shape)(input_feature)
    logger(f'ljy---spatial_attention -- input_feature: {ori_shape} ')
    #

    if K.image_data_format() == "channels_first":
        channel = input_feature.get_shape().as_list()[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.get_shape().as_list()[-1]  # 22
        cbam_feature = input_feature  # channel_attn: [bs=8, H=1, W=1, C=22]

    avg_pool = tf.reduce_max(cbam_feature, axis=3, keepdims=True)  # 对比Lambda更快。
    max_pool = tf.reduce_mean(cbam_feature, axis=3, keepdims=True)
    concat = tf.concat([avg_pool, max_pool], axis=3)

    # avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature) # TensorShape([8, 31*3840, 1, 1]) # 通道压缩-> 对空间信息的池化
    assert avg_pool.get_shape()[-1] == 1
    # max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)  # TensorShape([8, 31*3840, 1, 1])
    assert max_pool.get_shape()[-1] == 1
    # concat = Concatenate(axis=3)([avg_pool, max_pool])  # TensorShape([8, 31*3840, 1, 堆叠C=2])
    # Dropout(0.3)
    assert concat.get_shape()[-1] == 2
    logger(f'ljy---spatial_attention -- max_avg_pool_concat: {concat.get_shape()}')

    if isSeparable:
        # https://blog.csdn.net/c_chuxin/article/details/88581411
        cbam_feature = SeparableConv2D(filters=1,   # point-wise Conv  # SeparableConv2D
                              kernel_size=kernel_size,
                              depth_multiplier=1,
                              strides=1,
                              padding='same',
                              activation='sigmoid',
                              depthwise_initializer='he_normal',
                              pointwise_initializer='he_normal',
                              # kernel_regularizer=regularizers.l2(0.001),
                              use_bias=False)(concat)  # TensorShape([8, 31*3840, 1, 1])
    else:
        cbam_feature = Conv2D(filters=1,  # point-wise Conv  # SeparableConv2D
                              kernel_size=kernel_size,
                              strides=1,
                              padding='same',
                              activation='sigmoid',
                              kernel_initializer='he_normal',
                              # kernel_regularizer=regularizers.l2(0.001),
                              use_bias=False)(concat)  # TensorShape([8, 31*3840, 1, 1])

    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)


    res = multiply([input_feature, cbam_feature])  # [bs=8, H=31, W=3840, C=22] * [8, H=31, W=3840, C=1]
    # return Reshape(ori_shape)(res)# 错：tf.reshape(res, ori_shape)
    return res  # TODO: LJY us5--要不要改成 ADD(空间+通道)注意力？
    # TODO: -> 1.乘法运算费时 2.抑制了空间/通道有强项也有弱项的位置点-> 在浅层是否应该同时关注，
