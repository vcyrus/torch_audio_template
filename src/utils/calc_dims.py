from torch.nn import MaxPool2d, Conv2d
import math

def calculate_out_dims(input_dims, layers, n_conv_channels):
    ''' Calculates the dimensions of 
        input dims: (batch_size, nc, w, h)
        layers: sequence of layers
    '''
    _, n_channels, h, w = input_dims
    for l in layers:
        if isinstance(l, Conv2d):
            k_h, k_w = l.kernel_size
            p_h, p_w = l.padding
            s_h, s_w = l.stride
        if isinstance(l, MaxPool2d):
            if type(l.kernel_size) is not int:
                k_h, k_w = l.kernel_size
                s_h, s_w = l.stride 
            else:
                k_h = l.kernel_size
                k_w = k_h
                s_h = l.stride
                s_w = s_h
            p_h = 0
            p_w = 0
        h = math.floor((h + 2*p_h - k_h) / s_h + 1)
        w = math.floor((w + 2*p_w - k_w) / s_w + 1)

    return (n_conv_channels, h, w)
