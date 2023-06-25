from models import *
from keras.utils.vis_utils import plot_model


class TGNet(BaseModel):
    def build_model(self, input_shape, args):
        nf = args.nf
        h, w = input_shape[0:2]
        print('input_shape = ',input_shape)
        start_input = Input(shape=input_shape)
        end_input = Input(shape=(h, w, 8))
        temporal_input = Input(shape=(57,))
        roi_input = Input(shape=(57,))
        # coord_input = Input(shape=(h, w, 2))
        # input_tensors = [start_input, end_input, temporal_input, coord_input] Base Input
        # input_tensors = [start_input, end_input, temporal_input] 1차 Customized Input
        input_tensors = [start_input, end_input, temporal_input,roi_input] #2차 Customized Input

        # net_coord = Conv2D(args.coord_net, kernel_size=(1, 1), padding='same')(coord_input)
        # net_coord = Activation('relu')(net_coord)

        ### Temporal guided embedding
        net_temp = Dense(args.temp, activation='relu')(temporal_input)
        self.net_temp = Dense(args.temp, activation='relu')(net_temp)
        net_temp = RepeatVector(h * w)(self.net_temp)
        net_temp = Reshape((h, args.temp,w))(net_temp)

        ### roi guided embedding
        net_roi = Dense(args.temp, activation='relu')(roi_input)
        self.net_roi = Dense(args.temp, activation='relu')(net_roi)
        net_roi = RepeatVector(h * w)(self.net_roi)
        net_roi = Reshape((h, args.temp,w))(net_roi)

        ### U-net layers
        net1 = concatenate([start_input, net_temp], axis=-1)
        net1 = concatenate([net1, net_roi], axis=-1) ## appended for roi data
        net1 = gn_block(net1, nf, dropout=self.args.drop_p, regularizer=args.reg)
        net11 = AveragePooling2D(pool_size=(2, 2))(net1)
        net2 = gn_block(net11, nf * 2, dropout=self.args.drop_p, regularizer=args.reg)
        net3 = gn_block(net2, nf * 2, dropout=self.args.drop_p, regularizer=args.reg)
        net33 = concatenate([net2, net3])
        net4 = gn_block(net33, nf * 2, dropout=self.args.drop_p, regularizer=args.reg)
        net4 = concatenate([net2, net3, net4])

        net5 = deconv_block(net4, nf * 4, (2, 2), (2, 2), dropout=self.args.drop_p, regularizer=args.reg)
        net5 = concatenate([net5, net1])
        net6 = deconv_block(net5, nf * 4, (3, 3), (1, 1), 'same', dropout=self.args.drop_p, regularizer=args.reg)
        ### Drop-off Embedding
        net_end = gn_block(end_input, nf * 2, dropout=self.args.drop_p, regularizer=args.reg)
        net_end = gn_block(net_end, nf * 2, dropout=self.args.drop_p, regularizer=args.reg)

        ### Position-wise Regression
        # net7 = concatenate([net6, start_input, net_end, end_input, net_temp, net_coord], axis=-1)
        net7 = concatenate([net6, start_input, net_end, end_input, net_temp], axis=-1)
        net7 = gn_block(net7, nf * 4, kernel_size=(1, 1), dropout=self.args.drop_p, regularizer=args.reg)

        output = Conv2D(1, kernel_size=(1, 1), padding='same', kernel_regularizer=regularizers.l2(args.reg))(net7)
        self.output = Activation('relu')(output)

        model = Model(inputs=input_tensors, outputs=self.output)

        print("[*] Model Created: ", self.model_name)
        print(model.summary())
        # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        return model
