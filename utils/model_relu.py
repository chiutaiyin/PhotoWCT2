import tensorflow as tf
from tensorflow.keras import layers, Model
import csv
import numpy as np

csvfile = open('./utils/vgg_layers.csv', newline='')
vgg_layers = csv.DictReader(csvfile)

class VggEncoder(tf.keras.Model):
    def __init__(self, nblocks=3):
        super(VggEncoder, self).__init__()
        output_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'][:nblocks+1]
        btnecks = [[] for _ in range(nblocks+1)]
        self.kernels, self.biases = [{} for _ in range(len(output_layers))], [{} for _ in range(len(output_layers))]

        for idx, vgg_layer in enumerate(vgg_layers):
            name = vgg_layer['name'] 
            typename = vgg_layer['_typename']

            if idx == 0:
                name = 'preprocess'  # VGG 1st layer preprocesses with a 1x1 conv to multiply by 255 and subtract BGR mean as bias
                i = 0

            if typename == 'nn.SpatialReflectionPadding':
                btnecks[i].append(tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')))
            elif typename == 'nn.SpatialConvolution':
                filters = int(vgg_layer['filters'])
                kernel_size = (int(vgg_layer['kernel_size']), int(vgg_layer['kernel_size']))
                btnecks[i].append(tf.keras.layers.Conv2D(filters, kernel_size, padding='valid', name=name, trainable=False))
            elif typename == 'nn.ReLU':
                btnecks[i].append(tf.keras.layers.ReLU(name=name))                
            elif typename == 'nn.SpatialMaxPooling':
                btnecks[i].append(tf.keras.layers.AveragePooling2D(name=name))
            else:
                raise NotImplementedError(typename)
                
            if name in output_layers:
                i += 1

            if name == output_layers[-1]:
                print("Reached target layer: {}".format(name))
                break        
                
        self.btnecks = []
        for i, layers in enumerate(btnecks):
            shape = [3, 64, 128, 256][i]
            input_tensor = tf.keras.layers.Input(shape=[None, None, shape])
            x = input_tensor  

            for layer in layers:
                if type(layer) is tf.keras.layers.AveragePooling2D:
                    skip = x
                    x = layer(x)
                    skip = skip - tf.image.resize(x, tf.shape(skip)[1:3])
                else:
                    x = layer(x)
                
            outputs = [x] if i == 0 else [x, skip]
            self.btnecks.append(tf.keras.Model(inputs=input_tensor, outputs=outputs))
                
    def call(self, block, input_tensor):
        return self.btnecks[block](input_tensor)
       

def create_btneck(bt_layers, n_channels):
    input_layer = layers.Input(shape=(None, None, n_channels[1]))
    skip_layer = layers.Input(shape=(None, None, n_channels[0]))

    x = bt_layers[0](input_layer)
    x = bt_layers[1](x)
    x = bt_layers[2](x)
    x = x + skip_layer
    for layer in bt_layers[3:]:
        x = layer(x) 

    return tf.keras.Model(inputs=[input_layer, skip_layer], outputs=x)
    
    
class VggDecoder(tf.keras.Model):
    def __init__(self):
        super(VggDecoder, self).__init__()
        self.btnecks = []
        n1 = 64 
        self.btnecks.append(tf.keras.Sequential([layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
                                            layers.Conv2D(3, (3,3), padding='valid')]))
        
        n2 = 128 
        self.btnecks.append(create_btneck([layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
                                            layers.Conv2D(n1, (3,3), padding='valid', activation='relu'),
                                            layers.UpSampling2D(interpolation='bilinear'),
                                            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
                                            layers.Conv2D(n1, (3,3), padding='valid', activation='relu')], [n1, n2]))
        
        n3 = 256 
        self.btnecks.append(create_btneck([layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
                                            layers.Conv2D(n2, (3,3), padding='valid', activation='relu'),
                                            layers.UpSampling2D(interpolation='bilinear'),
                                            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
                                            layers.Conv2D(n2, (3,3), padding='valid', activation='relu')], [n2, n3]))
        
        n4 = 512 
        self.btnecks.append(create_btneck([layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
                                            layers.Conv2D(n3, (3,3), padding='valid', activation='relu'),
                                            layers.UpSampling2D(interpolation='bilinear'),
                                            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
                                            layers.Conv2D(n3, (3,3), padding='valid', activation='relu'),
                                            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
                                            layers.Conv2D(n3, (3,3), padding='valid', activation='relu'),
                                            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
                                            layers.Conv2D(n3, (3,3), padding='valid', activation='relu')], [n3, n4]))

        
    def call(self, block, input_tensor, skip):
        if block != 0:         
#             skip = tf.image.resize_with_crop_or_pad(skip, tf.shape(input_tensor)[1] * 2, tf.shape(input_tensor)[2] * 2)
            input_tensor = [input_tensor, skip]
        bt_out = self.btnecks[block](input_tensor)
        return bt_out