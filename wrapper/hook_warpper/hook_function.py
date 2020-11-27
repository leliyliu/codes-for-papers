import numpy as np
import torch.nn as nn

import models._modules as my_nn
import ipdb

__all__ = ['debug_graph_hooks', 'save_inner_hooks']


def save_inner_hooks(model):
    for name, module in model.named_modules():
        if isinstance(module, my_nn._ActQ) or isinstance(module, my_nn._LinearQ) \
                or isinstance(module, my_nn._Conv2dQ) \
                or isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) \
                or isinstance(module, my_nn.LinearBP) or isinstance(module, my_nn.Conv2dBP) \
                or isinstance(module, my_nn.ActShareQ) or isinstance(module, my_nn.Conv2dShareQ):
            # TODO: ReLU(inplace=false) MaxPool ????
            module.name = name
            module.register_forward_hook(save_inner_data)


# or isinstance(module, nn.ReLU) \
#  or isinstance(module, nn.MaxPool2d)

def save_inner_data(self, input, output, num_share=3):
    print('{}: {}'.format(self.name, self))
    if len(output) == num_share:
        for i in range(num_share):
            print('saving {}-out-{} shape: {}'.format(self.name, i, output[i].size()))
            np.save('{}_out{}'.format(self.name, i), output[i].detach().cpu().numpy())
    else:
        out = output
        print('saving {}_out shape: {}'.format(self.name, out.size()))
        nu = out.detach().cpu().numpy()
        # np_save = nu.reshape(-1, nu.shape[-1])
        # np.savetxt('{}_out.txt'.format(self.name), np_save, delimiter=' ', fmt='%.8f')
        np.save('{}_out'.format(self.name), nu)

    in_data = input
    in_data = in_data[0]
    # while not isinstance(in_data, torch.Tensor):
    #     in_data = in_data[0]
    if len(in_data) == num_share:
        for i in range(num_share):
            print('saving {}-in-{} shape: {}'.format(self.name, i, in_data[i].size()))
            nu1 = in_data[i].detach().cpu().numpy()
            np.save('{}_in{}'.format(self.name, i), nu1)
            # np_save = nu.reshape(-1, nu.shape[-1])
    else:
        print('saving {}_in shape: {}'.format(self.name, in_data.size()))
        nu = in_data.detach().cpu().numpy()
        # np_save = nu.reshape(-1, nu.shape[-1])
        # np.savetxt('{}_out.txt'.format(self.name), np_save, delimiter=' ', fmt='%.8f')
        np.save('{}_in'.format(self.name), nu)
    # np.savetxt('{}_in.txt'.format(self.name), np_save, delimiter=' ', fmt='%.8f')


def debug_graph(self, input, output):
    print('{}: type:{} input:{} ==> output:{} (max: {})'.format(self.name, type_str(self), [i.size() for i in input],
                                                                output.size(), output.max()))


def debug_graph_hooks(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or \
                isinstance(module, nn.MaxPool2d) or isinstance(module, my_nn.Concat) or \
                isinstance(module, nn.ZeroPad2d) or isinstance(module, my_nn.UpSample):
            module.name = name
            module.register_forward_hook(debug_graph)


def type_str(module):
    if isinstance(module, nn.Conv2d):
        return 'Conv2d'
    if isinstance(module, nn.MaxPool2d):
        return 'MaxPool2d'
    if isinstance(module, nn.Linear):
        return 'Linear'
    if isinstance(module, my_nn.Concat):
        return 'Concat'
    if isinstance(module, nn.ZeroPad2d):
        return 'ZeroPad2d'
    if isinstance(module, my_nn.UpSample):
        return 'Upsample'
    return 'Emtpy'
