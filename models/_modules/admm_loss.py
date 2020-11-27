from torch.nn.modules.loss import _Loss
from torch import Tensor
import math

__all__ = ['AdmmLoss']


class AdmmLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, rho, size_average=None,
                 reduce=None, reduction: str = 'mean') -> None:
        super(AdmmLoss, self).__init__(size_average, reduce, reduction)
        self.rho = rho
        self.rho_init = rho

    def forward(self, model, Z, U) -> Tensor:
        return admm_loss(model, Z, U, self.rho)

    def adjust_rho(self, convergence, accuracy, epoch):
        assert 0 <= epoch <= 1, 'Please Use Normalized epoch'
        if epoch < 0.3:
            self.rho = self.rho_init
        elif epoch > 0.7:
            self.rho = self.rho_init * 10
        else:
            self.rho = self.rho_init + (epoch - 0.3) * 9 * self.rho_init / 0.4

    def adjust_rho_v2(self, convergence, accuracy, epoch):
        # todo: how to adjust rho for better convergence with higher accuracy
        expected_convergence = math.sin(epoch * 2 * math.pi) / 4 + 0.25 if epoch < 0.718 else 1e-2
        if convergence < expected_convergence:  # good
            self.rho = self.rho * accuracy
        else:  # bad
            self.rho = self.rho * (convergence / expected_convergence)
        assert NotImplementedError


def admm_loss(model, Z, U, rho):
    idx = 0
    loss = None
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) != 1:  # len(BN.shape) = 1
            u = U[idx].to(param.device)
            z = Z[idx].to(param.device)
            if loss is None:
                loss = rho / 2 * (param - z + u).norm()
            else:
                loss += rho / 2 * (param - z + u).norm()
            idx += 1
    return loss
