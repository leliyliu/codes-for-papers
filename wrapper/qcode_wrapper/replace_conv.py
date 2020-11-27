r"""
    Replace `conv` with `convq`;
    Replace `Linear` with `LinearQ`
"""

from models._modules import Conv2dQv2, LinearQv2, ActQv2
# from models._modules import Conv2dBP, LinearBP
import models._modules as my_nn 
import torch.nn as nn
import ipdb


def quantize_scale_and_bias(model, bias_bits=8, scale_bits=8):
    for module_name, module in model.named_modules():
        if isinstance(module, ActQv2):
            if bias_bits > 0:
                module.set_out_scale(True)
            module.set_scale_bits(nbits=scale_bits)
        elif isinstance(module, LinearQv2) or isinstance(module, Conv2dQv2):
            module.set_scale_bits(nbits=scale_bits)
            module.set_bias_bits(nbits=bias_bits)
    return model


def add_radix_position(model):
    for module_name, module in model.named_modules():
        ipdb.set_trace()
        if isinstance(module, nn.Linear):
            m = LinearBP()
            pass
        if isinstance(module, nn.Conv2d):
            m = Conv2dBP()
            pass

is_first = True

def replace_conv(conv_ori, conv_name, act_name, args, **kwargs):
    global is_first
    if isinstance(conv_ori, nn.Conv2d):
        if is_first:
            is_first = False
            return conv_ori
        if conv_name == 'Conv2dDPQ':
            # if is_first:
            #     is_first = False
            #     return conv_ori
            # else:
            m = conv_ori
            has_bias = m.bias is not None
            qmode = 2 if args.q_mode == 'kernel_wise' else 1
            my_m = my_nn.__dict__[conv_name](m.in_channels, m.out_channels, m.kernel_size, m.stride,
                                            m.padding, m.dilation, groups=m.groups, bias=has_bias, 
                                            mode = qmode, **kwargs)
            conv_st_dict = m.state_dict()
            W = conv_st_dict['weight']
            my_m.weight.data.copy_(W)
            if has_bias:
                bias = conv_st_dict['bias']
                my_m.bias.data.copy_(bias)
            my_m.to(m.weight.device)
            return my_m
        else:
            m = conv_ori 
            has_bias = m.bias is not None
            qmode = 2 if args.q_mode == 'kernel_wise' else 1
            my_m = my_nn.__dict__[conv_name](m.in_channels, m.out_channels, m.kernel_size, m.stride,
                                             m.padding, m.dilation, groups=m.groups, bias=has_bias, mode=qmode)
            conv_st_dict = m.state_dict()
            W = conv_st_dict['weight']
            my_m.weight.data.copy_(W)
            if has_bias:
                bias = conv_st_dict['bias']
                my_m.bias.data.copy_(bias)
            my_m.to(m.weight.device)
            return my_m


    else:
        return conv_ori


def replace_conv_recursively(model, conv_name, args, **kwargs):
    # todo: support otherwise conv and linear replacement
    act_name = conv_name.replace('Conv2d', 'Act')

    for module_name in model._modules:
        if len(model._modules[module_name]._modules) > 0: # 这里实际上对应的是一个module 下面的小的module 
            replace_conv_recursively(model._modules[module_name], conv_name, args, **kwargs)
        model._modules[module_name] = replace_conv(model._modules[module_name], conv_name, act_name, args, **kwargs)


    return model
