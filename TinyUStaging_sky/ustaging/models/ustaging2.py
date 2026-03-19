"""
Implementation of UTime as described in:

Mathias Perslev, Michael Hejselbak Jensen, Sune Darkner, Poul Jørgen Jennum
and Christian Igel. U-Time: A Fully Convolutional Network for Time Series
Segmentation Applied to Sleep Staging. Advances in Neural Information
Processing Systems (NeurIPS 2019)
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, BatchNormalization, Cropping2D, \
                                    Concatenate, MaxPooling2D, Dense, \
                                    UpSampling2D, ZeroPadding2D, Lambda, Conv2D, \
                                    AveragePooling2D, DepthwiseConv2D, Dropout

from mpunet.logging import ScreenLogger
from mpunet.utils.conv_arithmetics import compute_receptive_fields
from mpunet.train.utils import init_activation
from .attention_module import *
# from tensorflow.keras.utils import plot_model
# import os

class UStaging2(Model):          
    """
    # ljy❤：attention_module - dense - VS UStaging v1 -- 减少参数 [None, 23/31*3840, 1, 【7】]
    
    OBS: Uses 2D operations internally with a 'dummy' axis, so that a batch
         of shape [bs, d, c] is processed as [bs, d, 1, c]. These operations
         are (on our systems, at least) currently significantly faster than
         their 1D counterparts in tf.keras.❤ 优化网络

    See also original U-net paper at http://arxiv.org/abs/1505.04597
    """
    def __init__(self,
                 n_classes,
                 batch_shape,
                 depth=4,
                 dilation=((1,2), (3,1), (2,3), (1,2)),  # TODO: ASPP + 避免gridding effect
                 activation="elu",
                 dense_classifier_activation="tanh",
                 kernel_size=5,
                 transition_window=1,
                 padding="same",
                 init_filters=16,
                 complexity_factor=2,
                 l2_reg=None, # 0.001,  # ljy加，原：None
                 pools=(10, 8, 6, 4),
                 data_per_prediction=None,
                 attention_module_bottom=None,
                 attention_module_dense=None,
                 ratio_se=None,
                 ratio_cbam=None,
                 logger=None,
                 build=True,
                 **kwargs):
        """
        n_classes (int):
            The number of classes to model, gives the number of filters in the
            final 1x1 conv layer.
        batch_shape (list): Giving the shape of one one batch of data,
                            potentially omitting the zeroth axis (the batch
                            size dim)
        depth (int):
            Number of conv blocks in encoding layer (number of 2x2 max pools)
            Note: each block doubles the filter count while halving the spatial
            dimensions of the features.
        dilation (int):
            TODO
        activation (string):
            Activation function for convolution layers
        dense_classifier_activation (string):
            TODO
        kernel_size (int):
            Kernel size for convolution layers
        transition_window (int):
            TODO
        padding (string):
            Padding type ('same' or 'valid')
        complexity_factor (int/float):
            Use int(N * sqrt(complexity_factor)) number of filters in each
            convolution layer instead of default N.
        l2_reg (float in [0, 1])
            L2 regularization on conv weights
        pools (int or list of ints):
            TODO
        data_per_prediction (int):
            TODO
        logger (mpunet.logging.Logger | ScreenLogger):
            mpunet.Logger object, logging to files or screen.
        build (bool):
            TODO
        """
        super(UStaging2, self).__init__()

        # Set logger or standard print wrapper
        if dilation is None:
            dilation = [[2,2]] * depth # [[1, 2], [3, 1], [2, 3], [1, 2]]

        self.logger = logger or ScreenLogger()

        # Set various attributes
        assert len(batch_shape) == 4  # ljy: [bs=8/12/16, seg_len=2*margin+1=23/31, data_per_prediction=3840, c=2]
        self.n_periods = batch_shape[1]  # seg_len/n_labels=31
        self.input_dims = batch_shape[2] # 3840
        self.n_channels = batch_shape[3] # 2
        self.n_classes = int(n_classes)
        self.cf = np.sqrt(complexity_factor)
        self.init_filters = init_filters     # 16
        self.kernel_size = int(kernel_size)  # 5
        self.transition_window = transition_window
        self.activation = activation
        self.l2_reg = l2_reg
        self.depth = depth
        self.n_crops = 0
        # attention_module
        self.attention_module_bottom = attention_module_bottom 
        self.attention_module_dense = attention_module_dense
        self.ratio_se = ratio_se
        self.ratio_cbam = ratio_cbam

        self.pools = [pools] * self.depth if not \
            isinstance(pools, (list, tuple)) else pools
        if len(self.pools) != self.depth:
            raise ValueError("Argument 'pools' must be a single integer or a "
                             "list of values of length equal to 'depth'.")

        # self.dilation = int(dilation)
        # ljy改：↓注意❤：每层有两个卷积！gridding effect必须锯齿！ [[1,2], [3,1], [2, 3], [1, 2]]
        self.dilation = [[dilation, dilation]] * self.depth if not \
            isinstance(dilation, (list, tuple)) else dilation
        if len(self.dilation) != self.depth:
            raise ValueError("Argument 'dilation' must be a single integer or a "
                             "list of values of length equal to 'depth'.")


        self.padding = padding.lower()
        if self.padding != "same":
            raise ValueError("Currently, must use 'same' padding.")

        self.dense_classifier_activation = dense_classifier_activation
        self.data_per_prediction = data_per_prediction or self.input_dims
        if not isinstance(self.data_per_prediction, (int, np.integer)):
            raise TypeError("data_per_prediction must be an integer value")
        if self.input_dims % self.data_per_prediction:
            raise ValueError("'input_dims' ({}) must be evenly divisible by "
                             "'data_per_prediction' ({})".format(self.input_dims,
                                                                 self.data_per_prediction))

        if build:
            # Build model and init base keras Model class
            super().__init__(*self.init_model())

            # Compute receptive field
            ind = [x.__class__.__name__ for x in self.layers].index("UpSampling2D")
            self.receptive_field = compute_receptive_fields(self.layers[:ind])[-1][-1]

            # Log the model definition
            self.log()

            # plot_model(UStaging, to_file=os.path.join('/export/ljy', "UStaging.png"), show_shapes=True)

        else:
            self.receptive_field = [None]

    @staticmethod
    def create_encoder(in_,
                       depth,
                       pools,
                       filters,
                       kernel_size,
                       activation,
                       dilation,
                       padding,
                       complexity_factor,
                       regularizer=None,
                       name="encoder",
                       name_prefix=""):
        name = "{}{}".format(name_prefix, name)
        residual_connections = []
        for i in range(depth):
            l_name = name + "_L%i" % i
            conv = Conv2D(int(filters*complexity_factor), (kernel_size, 1),
                          activation=activation, padding=padding,
                          kernel_regularizer=regularizer,
                          bias_regularizer=regularizer,
                          dilation_rate=dilation[i][0],
                          name=l_name + "_conv1")(in_)  # name=f'encoder_L{i}_conv1'
            bn = BatchNormalization(name=l_name + "_BN1")(conv)  # name=f'encoder_L{i}_BN1'
            conv = Conv2D(int(filters*complexity_factor), (kernel_size, 1),
                          activation=activation, padding=padding,
                          kernel_regularizer=regularizer,
                          bias_regularizer=regularizer,
                          dilation_rate=dilation[i][1],
                          name=l_name + "_conv2")(bn)
            bn = BatchNormalization(name=l_name + "_BN2")(conv)
            in_ = MaxPooling2D(pool_size=(pools[i], 1),
                               name=l_name + "_pool")(bn)

            # add bn layer to list for residual connections❤ ljy
            residual_connections.append(bn)  # ljy: 每层的BN2
            filters = int(filters * 2)  # (16 -> 32) - 64 - 128 - 256

        # Bottom
        name = "{}bottom".format(name_prefix)
        conv = Conv2D(int(filters*complexity_factor), (kernel_size, 1),
                      activation=activation, padding=padding,
                      kernel_regularizer=regularizer,
                      bias_regularizer=regularizer,
                      dilation_rate=1,
                      name=name + "_conv1")(in_)
        bn = BatchNormalization(name=name + "_BN1")(conv)
        conv = Conv2D(int(filters*complexity_factor), (kernel_size, 1),
                      activation=activation, padding=padding,
                      kernel_regularizer=regularizer,
                      bias_regularizer=regularizer,
                      dilation_rate=1,
                      name=name + "_conv2")(bn)
        encoded = BatchNormalization(name=name + "_BN2")(conv)

        return encoded, residual_connections, filters  # 最后，L4的filters=256

    def create_upsample(self,
                        in_,
                        res_conns,
                        depth,
                        pools,
                        filters,
                        kernel_size,  # 5
                        activation,
                        dilation,  # NOT USED
                        padding,
                        complexity_factor,
                        regularizer=None,
                        name="upsample",
                        attention_module_bottom=None,
                        name_prefix=""):
        name = "{}{}".format(name_prefix, name)
        residual_connections = res_conns[::-1]

        # attention_module （At bottom）
        self.logger("ljy--- [Decoder(Upsample)] - cls.get_shape().as_list(): ",
                    in_.get_shape().as_list())  # [None, 23/31*3840, 1, 7]


        if attention_module_bottom is not None:
            self.logger(f'Attention(1) - Bottom: {attention_module_bottom}')

            if isinstance(self.ratio_se, list):
                ratio_se=self.ratio_se[0]
            else:
                ratio_se = self.ratio_se

            if isinstance(self.ratio_cbam, list):
                ratio_cbam=self.ratio_cbam[0]
            else:
                ratio_cbam = self.ratio_cbam

            in_ = attach_attention_module(in_,
                                          attention_module_bottom,
                                          ratio_cbam=ratio_cbam,
                                          ratio_se=ratio_se,
                                          data_per_prediction=self.data_per_prediction,
                                          # kernel_size=7,
                                          logger=self.logger)


        for i in range(depth):
            filters = int(filters/2)  # (256 -> 128) - 64 - 32 - 16
            l_name = name + "_L%i" % i

            # Up-sampling block
            fs = pools[::-1][i]  # 逆置 [4, 6, 8, 10]
            up = UpSampling2D(size=(fs, 1) ,# interpolation='bilinear',
                              name=l_name + "_up")(in_)
            conv = Conv2D(int(filters*complexity_factor), (fs, 1),
                          activation=activation,
                          padding=padding,
                          kernel_regularizer=regularizer,
                          bias_regularizer=regularizer,
                          name=l_name + "_conv1")(up)
            bn = BatchNormalization(name=l_name + "_BN1")(conv)

            # Crop and concatenate
            cropped_res = self.crop_nodes_to_match(residual_connections[i], bn)  # ljy❤: Encoder的尺寸>=Decoder，∴编码器在前
            # cropped_res = residual_connections[i]
            merge = Concatenate(axis=-1,  # ljy：沿着C轴->针对每个channel去融合
                                name=l_name + "_concat")([cropped_res, bn])
            conv = Conv2D(int(filters*complexity_factor), (kernel_size, 1),
                          activation=activation, padding=padding,
                          kernel_regularizer=regularizer,
                          bias_regularizer=regularizer,
                          name=l_name + "_conv2")(merge)
            bn = BatchNormalization(name=l_name + "_BN2")(conv)
            conv = Conv2D(int(filters*complexity_factor), (kernel_size, 1),
                          activation=activation, padding=padding,
                          kernel_regularizer=regularizer,
                          bias_regularizer=regularizer,
                          name=l_name + "_conv3")(bn)
            in_ = BatchNormalization(name=l_name + "_BN3")(conv)
        return in_

    def create_dense_modeling(self,
                              in_,
                              in_reshaped,  # ljy❤: [-1, 23*3840, 1, 2] -> 拉成1D卷积❤❤❤
                              filters,  # ljy❤ =n_classes=5
                              dense_classifier_activation,
                              regularizer,
                              complexity_factor,
                              name_prefix="",
                              attention_module_dense=None,
                              **kwargs):
        self.logger("ljy--- [create_dense_modeling(1*1 Conv)] - in_.get_shape().as_list(): ", in_.get_shape().as_list())    # [None, 23/31*3840, 1, 【22】]

        cls = Conv2D(filters=int(filters * complexity_factor),
                     kernel_size=(1, 1),
                     kernel_regularizer=regularizer,
                     bias_regularizer=regularizer,
                     activation=dense_classifier_activation,
                     name="{}dense_classifier_out".format(name_prefix))(in_)

        # attention_module - VS UStaging v1 -- 减少参数 [None, 23/31*3840, 1, 【7】]
        if attention_module_dense is not None:
            self.logger(f'Attention (2) - Dense: {attention_module_dense}')

            if isinstance(self.ratio_se, list):
                ratio_se = self.ratio_se[1]
            else:
                ratio_se = self.ratio_se

            if isinstance(self.ratio_cbam, list):
                ratio_cbam = self.ratio_cbam[1]
            else:
                ratio_cbam = self.ratio_cbam

            cls = attach_attention_module(cls,
                                          attention_module_dense,
                                          ratio_cbam=ratio_cbam,
                                          ratio_se=ratio_se,
                                          data_per_prediction=self.data_per_prediction,
                                          # kernel_size=7,
                                          logger=self.logger)


        self.logger("ljy--- [dense(1*1 Conv)] - cls.get_shape().as_list(): ",
                    cls.get_shape().as_list())  # [None, 23/31*3840, 1, 7]

        s = (self.n_periods * self.input_dims) - cls.get_shape().as_list()[1]  # 为了对齐23*3840
        out = self.crop_nodes_to_match(  # reshape -> [-1, 23*3840 (seg轴H-裁减), 5？ (时间/filters轴W-不裁剪), 2] -> 拉成1D卷积❤❤❤
            node1=ZeroPadding2D(padding=[[s // 2, s // 2 + s % 2], [0, 0]])(cls),
            node2=in_reshaped
        )
        self.logger("ljy--- [create_dense_modeling(1*1 Conv) (cropped)] - ❤ out.get_shape().as_list(): ", out.get_shape().as_list())    # [None, 23/31*3840, 1, 7]
        self.logger("ljy--- [create_dense_modeling(1*1 Conv)] - in_reshaped.get_shape().as_list(): ", in_reshaped.get_shape().as_list())  # [None, 23/31*3840, 1, 2]
        return out  # [-1, 31/23*3840, 1, 7]

    @staticmethod
    def create_seq_modeling(in_,  # ljy❤: [-1, 31/23*3840, 1, 7] -> 每类一个FeatureMap❤❤❤
                            input_dims,  # 3840=30*128
                            data_per_period,  # 3840=30*128(可自定义-- 128Hz * ? sec/label)
                            n_periods,   # seg_len/n_labels=31
                            n_classes,   # 5
                            transition_window,
                            activation,
                            regularizer=None,
                            name_prefix=""):
        cls = AveragePooling2D((data_per_period, 1),  # ljly❤❤ [-1, n_periods(28/31), 1, 7] -> 每类一个FeatureMap❤(转为Label长度)
                               name="{}average_pool".format(name_prefix))(in_)
        out = Conv2D(filters=n_classes,   # 5 -> 每类一个FeatureMap❤
                     kernel_size=(transition_window, 1),
                     activation=activation,
                     kernel_regularizer=regularizer,
                     bias_regularizer=regularizer,
                     padding="same",
                     name="{}sequence_conv_out_1".format(name_prefix))(cls)
        out = Conv2D(filters=n_classes,
                     kernel_size=(transition_window, 1),
                     activation="softmax",  # ljy❤：置信分数 -> softmax 概率输出
                     kernel_regularizer=regularizer,
                     bias_regularizer=regularizer,
                     padding="same",
                     name="{}sequence_conv_out_2".format(name_prefix))(out)
        s = [-1, n_periods, input_dims//data_per_period, n_classes]  # ljy❤ [-1, 31/23, 3840 // (128*?), 5]
        if s[2] == 1:
            s.pop(2)  # Squeeze the dim
        out = Lambda(lambda x: tf.reshape(x, s),
                     name="{}sequence_classification_reshaped".format(name_prefix))(out)
        return out

    def init_model(self, inputs=None, name_prefix=""):
        """
        Build the UNet model with the specified input image shape.
        """
        if inputs is None:
            inputs = Input(shape=[self.n_periods,  # ljy❤某一个batch元素：[23, 3840, 2]
                                  self.input_dims,
                                  self.n_channels])
        reshaped = [-1, self.n_periods*self.input_dims, 1, self.n_channels]  # ljy❤: [-1, 23*3840, 1, 2] -> 拉成1D卷积❤❤❤
        in_reshaped = Lambda(lambda x: tf.reshape(x, reshaped))(inputs)

        # Apply regularization if not None or 0
        regularizer = regularizers.l2(self.l2_reg) if self.l2_reg else None

        # Get activation func from tf or tfa
        activation = init_activation(activation_string=self.activation)

        settings = {
            "depth": self.depth,
            "pools": self.pools,
            "filters": self.init_filters,  # 初始值为16（L4最后为256）
            "kernel_size": self.kernel_size,
            "activation": activation,
            "dilation": self.dilation,  # TODO: [[1,2], [3,1], [2,3], [1,2]...] -> 避免gridding effect
            "padding": self.padding,
            "regularizer": regularizer,
            "name_prefix": name_prefix,
            "complexity_factor": self.cf
        }

        """
        Encoding path
        """
        enc, residual_cons, filters = self.create_encoder(in_=in_reshaped,
                                                          **settings)

        """
        Decoding path
        """
        settings["filters"] = filters
        up = self.create_upsample(enc, residual_cons, attention_module_bottom=self.attention_module_bottom, **settings)

        """
        Dense class modeling layers
        """
        cls = self.create_dense_modeling(in_=up,
                                         in_reshaped=in_reshaped,  # ljy❤: [-1, 23*3840, 1, 2] -> 拉成1D卷积❤❤❤
                                         filters=self.n_classes,  # 5
                                         dense_classifier_activation=self.dense_classifier_activation,
                                         regularizer=regularizer,
                                         complexity_factor=self.cf,
                                         attention_module_dense=self.attention_module_dense,
                                         name_prefix=name_prefix)

        """
        Sequence modeling
        """
        out = self.create_seq_modeling(in_=cls,  # ljy❤: [-1, 23*3840, 5???, 2] -> 每类一个featureMap❤❤❤
                                       input_dims=self.input_dims,  # 3840
                                       data_per_period=self.data_per_prediction,  # 3840
                                       n_periods=self.n_periods,  # seg_len/n_labels=31
                                       n_classes=self.n_classes,  # 5
                                       transition_window=self.transition_window,
                                       activation=activation,
                                       regularizer=regularizer,
                                       name_prefix=name_prefix)

        return [inputs], [out]  # ljy❤ out：[-1, 31/23, 3840 // (128*?), 5]

    def crop_nodes_to_match(self, node1, node2):
        """
        If necessary, applies Cropping2D layer to node1 to match shape of node2
        """
        s1 = np.array(node1.get_shape().as_list())[1:-2]
        s2 = np.array(node2.get_shape().as_list())[1:-2]

        if np.any(s1 != s2):
            self.n_crops += 1
            c = (s1 - s2).astype(np.int)
            cr = np.array([c // 2, c // 2]).flatten()
            cr[self.n_crops % 2] += c % 2
            cropped_node1 = Cropping2D([list(cr), [0, 0]])(node1)  # ljy❤- 沿seg轴(H)：裁减, 沿时间轴(W)=1：不裁剪
        else:
            cropped_node1 = node1
        return cropped_node1

    def log(self):
        self.logger("{} Model Summary\n"
                    "-------------------".format(__class__.__name__))
        self.logger("N periods:         {}".format(self.n_periods))
        self.logger("Input dims:        {}".format(self.input_dims))
        self.logger("N channels:        {}".format(self.n_channels))
        self.logger("N classes:         {}".format(self.n_classes))
        self.logger("Kernel size:       {}".format(self.kernel_size))
        self.logger("Dilation rate:     {}".format(self.dilation))
        self.logger("CF factor:         {:.3f}".format(self.cf**2))
        self.logger("Init filters:      {}".format(self.init_filters))
        self.logger("Depth:             {}".format(self.depth))
        self.logger("Poolings:          {}".format(self.pools))
        self.logger("Transition window  {}".format(self.transition_window))
        self.logger("Dense activation   {}".format(self.dense_classifier_activation))
        self.logger("l2 reg:            {}".format(self.l2_reg))
        self.logger("Padding:           {}".format(self.padding))
        self.logger("Conv activation:   {}".format(self.activation))
        self.logger("Receptive field:   {}".format(self.receptive_field[0]))
        self.logger("Seq length.:       {}".format(self.n_periods*self.input_dims))
        self.logger("N params:          {}".format(self.count_params()))
        self.logger("Input:             {}".format(self.input))
        self.logger("Output:            {}".format(self.output))


