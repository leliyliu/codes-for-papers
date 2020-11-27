import models._modules as my_nn
import torch.nn as nn
import ipdb

is_first = True


def replace_sq(block, **kwargs):
    global is_first
    if isinstance(block, nn.Conv2d):
        if is_first:
            is_first = False
            return block
        else:
            m = block
            has_bias = m.bias is not None
            my_m = my_nn.Conv2dSQ(m.in_channels, m.out_channels, m.kernel_size, m.stride,
                                  m.padding, m.dilation, groups=m.groups, bias=has_bias,
                                  **kwargs)
            conv_st_dict = m.state_dict()
            W = conv_st_dict['weight']
            my_m.weight.data.copy_(W)
            if has_bias:
                bias = conv_st_dict['bias']
                my_m.bias.data.copy_(bias)
            my_m.to(m.weight.device)
            return my_m
    else:
        return block


def replace_sq_recursively(model, **kwargs):
    for module_name in model._modules:
        model._modules[module_name] = replace_sq(model._modules[module_name], **kwargs)
        if len(model._modules[module_name]._modules) > 0:
            replace_sq_recursively(model._modules[module_name], **kwargs)

    return model
