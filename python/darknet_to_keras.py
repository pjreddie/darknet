#! /usr/bin/env python
import argparse
import configparser
import io
import os
from collections import defaultdict

import numpy as np
from keras.applications.mobilenet import DepthwiseConv2D
from keras.layers import (Conv2D, GlobalAveragePooling2D, Input, Lambda,
                          MaxPooling2D, Activation, BatchNormalization, concatenate, LeakyReLU)
from keras.models import Model


parser = argparse.ArgumentParser(description='Darknet to Keras converter.')
parser.add_argument('config_path', help='Path to Darknet cfg file.')
parser.add_argument('weights_path', help='Path to Darknet weights file.')
parser.add_argument('output_path', help='Path to output Keras model file.')


def space_to_depth_x2(x):
    """Thin wrapper for Tensorflow space_to_depth with block_size=2."""
    # Import currently required to make Lambda work.
    # See: https://github.com/fchollet/keras/issues/5088#issuecomment-273851273
    import tensorflow as tf
    return tf.space_to_depth(x, block_size=2)


def space_to_depth_x2_output_shape(input_shape):
    """Determine space_to_depth output shape for block_size=2.

    Note: For Lambda with TensorFlow backend, output shape may not be needed.
    """
    return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 *
            input_shape[3]) if input_shape[1] else (input_shape[0], None, None,
                                                    4 * input_shape[3])


def unique_config_sections(config_file):
    """Convert all config sections to have unique names.

    Adds unique suffixes to config sections for compability with configparser.
    """
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream


def padding_type_from_pad_number(size, pad):
    if pad == 0:
        padding = 'valid'
    elif size % pad == 1 or pad == 1:
        padding = 'same'
    else:
        raise ValueError("Only 'valid' and 'same' padding are supported!")

    return padding


def activation_layer_from_name(name):
    """
        Translates DarkNet activation string to Keras Activation Layer
        See: https://github.com/pjreddie/darknet/blob/master/src/activations.c
    """
    if name == 'leaky':
        return LeakyReLU(alpha=0.1)

    if name == 'logistic':
        keras_name = 'sigmoid'
    elif name == 'relu':
        keras_name = 'relu'
    elif name == 'elu':
        keras_name = 'elu'
    elif name == 'tanh':
        keras_name = 'tanh'
    elif name == 'linear':
        keras_name = 'linear'
    else:
        raise ValueError('Unknown activation: {}'.format(name))

    return Activation(keras_name)


# Block with actual BatchNorm Layer
def add_conv_block_bn(prev_layer, weights_file, filters, size, stride, pad, groups, activation, batch_normalize):
    if size % 2 == 0:
        raise ValueError('Filter size must be odd number!')

    if not (groups == 1 or groups == filters):
        raise ValueError('Currently only Conv2D and DepthwiseConv2D are supported!')

    weights_read = 0
    prev_layer_channels = prev_layer._keras_shape[-1]

    use_bias = not batch_normalize
    is_depthwise_conv = groups == filters
    keras_padding = padding_type_from_pad_number(size, pad)

    # TensorFlow weights order: (height, width, in_dim, out_dim)
    weights_shape = (size, size, prev_layer_channels, filters // groups)

    # DarkNet weights are serialized Caffe-style: (out_dim, in_dim, height, width)
    darknet_w_shape = (filters // groups, prev_layer_channels, size, size)
    kernel_weights_count = np.product(weights_shape)

    # Weights in DarkNet are stored as [biases, [bn_scales, bn_mean, bn_variance], kernels]
    # See "save_convolutional_weights" here: https://github.com/pjreddie/darknet/blob/master/src/parser.c
    bias_buffer = weights_file.read(filters * 4)
    conv_bias = np.ndarray(shape=(filters, ), dtype='float32', buffer=bias_buffer)
    weights_read += filters

    if batch_normalize:
        bn_epsilon = 1e-5

        # Note: DarkNet doesn't have "beta" in convolutions (biases are used instead)
        bn_buffer = weights_file.read(filters * 12)
        bn_weights = np.ndarray(shape=(3, filters), dtype='float32', buffer=bn_buffer)
        weights_read += 3 * filters

        gamma, running_mean, running_var = bn_weights[0], bn_weights[1], bn_weights[2]

        batch_norm_weights = [
            gamma, conv_bias, running_mean, running_var
        ]

    kernels_buffer = weights_file.read(kernel_weights_count * 4)
    conv_kernels = np.ndarray(shape=darknet_w_shape, dtype='float32', buffer=kernels_buffer)
    conv_kernels = np.transpose(conv_kernels, [2, 3, 1, 0])

    assert conv_kernels.shape == weights_shape
    weights_read += kernel_weights_count

    layer_weights = [conv_kernels]
    if use_bias:
        layer_weights.append(conv_bias)

    # Create Conv2D or DepthwiseConv2D layer
    layer_params = {
        'kernel_size': (size, size),
        'strides': (stride, stride),
        'use_bias': use_bias,
        'weights': layer_weights,
        'activation': None,
        'padding': keras_padding
    }

    if is_depthwise_conv:
        conv_block = DepthwiseConv2D(**layer_params)(prev_layer)
    else:
        conv_block = Conv2D(filters, **layer_params)(prev_layer)

    # Add BatchNormalization layer if necessary
    if batch_normalize:
        conv_block = BatchNormalization(weights=batch_norm_weights, epsilon=bn_epsilon)(conv_block)

    # Finally, add activation layer
    act_layer = activation_layer_from_name(activation)
    conv_block = act_layer(conv_block)

    return conv_block, weights_read


def _main(args):
    config_path = os.path.expanduser(args.config_path)
    weights_path = os.path.expanduser(args.weights_path)
    assert config_path.endswith('.cfg'), '{} is not a .cfg file'.format(
        config_path)
    assert (
            weights_path.endswith('.weights') or
            weights_path.endswith('.backup')), '{} is not a .weights file'.format(weights_path)

    output_path = os.path.expanduser(args.output_path)
    assert output_path.endswith('.h5'), 'output path {} is not a .h5 file'.format(output_path)
    output_root = os.path.splitext(output_path)[0]

    # Load weights and config.
    print('Loading weights.')
    weights_file = open(weights_path, 'rb')
    weights_header = np.ndarray(shape=(4, ), dtype='int32', buffer=weights_file.read(20))
    print('Weights Header: ', weights_header)

    print('Parsing Darknet config.')
    unique_config_file = unique_config_sections(config_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)

    print('Creating Keras model.')
    image_height = int(cfg_parser['net_0']['height'])
    image_width = int(cfg_parser['net_0']['width'])

    prev_layer = Input(shape=(image_height, image_width, 3))
    all_layers = [prev_layer]

    count = 0
    for section in cfg_parser.sections():
        print('Parsing section {}'.format(section))

        if section.startswith('convolutional'):
            filters = int(cfg_parser[section]['filters'])
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            pad = int(cfg_parser[section]['pad'])
            groups = int(cfg_parser[section].get('groups', 1))
            activation = cfg_parser[section]['activation']
            batch_normalize = 'batch_normalize' in cfg_parser[section]

            conv_block, weights_read = add_conv_block_bn(
                prev_layer, weights_file, filters, size, stride, pad, groups, activation, batch_normalize
            )

            all_layers.append(conv_block)
            prev_layer = all_layers[-1]
            count += weights_read

        elif section.startswith('maxpool'):
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])

            max_pool = MaxPooling2D(
                padding='same',
                pool_size=(size, size),
                strides=(stride, stride))(prev_layer)

            all_layers.append(max_pool)
            prev_layer = all_layers[-1]

        elif section.startswith('avgpool'):
            if len(cfg_parser.items(section)) != 0:
                raise ValueError('{} with params unsupported.'.format(section))

            avg_pool = GlobalAveragePooling2D()(prev_layer)

            all_layers.append(avg_pool)
            prev_layer = all_layers[-1]

        elif section.startswith('route'):
            ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
            layers = [all_layers[i] for i in ids]

            if len(layers) > 1:
                print('Concatenating route layers:', layers)
                concatenate_layer = concatenate(layers)
                all_layers.append(concatenate_layer)
                prev_layer = concatenate_layer
            else:
                skip_layer = layers[0]  # only one layer to route
                all_layers.append(skip_layer)
                prev_layer = skip_layer

        elif section.startswith('reorg'):
            block_size = int(cfg_parser[section]['stride'])
            assert block_size == 2, 'Only reorg with stride 2 supported.'

            reorg_layer = Lambda(
                space_to_depth_x2,
                output_shape=space_to_depth_x2_output_shape,
                name='space_to_depth_x2')(prev_layer)

            all_layers.append(reorg_layer)
            prev_layer = all_layers[-1]

        elif section.startswith('region'):
            with open('{}_anchors.txt'.format(output_root), 'w') as f:
                print(cfg_parser[section]['anchors'], file=f)

        elif (section.startswith('net') or section.startswith('cost') or
              section.startswith('softmax')):
            pass  # Configs not currently handled during model definition.

        else:
            raise ValueError(
                'Unsupported section header type: {}'.format(section))

    # Create and save model.
    model = Model(inputs=all_layers[0], outputs=all_layers[-1])
    print(model.summary())

    model.save('{}'.format(output_path))
    print('Saved Keras model to {}'.format(output_path))

    # Check to see if all weights have been read.
    remaining_weights = len(weights_file.read()) / 4
    weights_file.close()

    print('Read {} of {} from Darknet weights.'.format(count, count + remaining_weights))

    if remaining_weights > 0:
        print('Warning: {} unused weights'.format(remaining_weights))


if __name__ == '__main__':
    _main(parser.parse_args())
