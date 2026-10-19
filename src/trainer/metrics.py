import torch
import torchmetrics


class LossMomentum(torchmetrics.Metric):
    """
    Computes momentum of loss with beta_t = 1 - 1/t

    m_t = (1 - 1/t) m_{t-1} + (1/t) loss_t
    """

    def __init__(self):
        super().__init__()

        self.add_state("momentum", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("t", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, loss: torch.Tensor):
        """
        loss: scalar tensor
        """
        loss = loss.detach()

        self.t += 1
        beta_t = 1.0 - 1.0 / self.t.float()

        self.momentum = beta_t * self.momentum + (1.0 - beta_t) * loss

    def compute(self):
        return self.momentum


class VRLossMomentum(torchmetrics.Metric):
    """
    m_t =
        (1 - beta_t) * l_t
        + beta_t * (l_t - l_prev)
        + beta_t * m_{t-1}

    where:
        l_prev = l(x_{t-1}, z_t)
        l_t    = l(x_t, z_t)
        beta_t = 1 - 1/t
    """

    def __init__(self):
        super().__init__()

        self.add_state("m", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("t", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("initialized", default=torch.tensor(0), dist_reduce_fx="max")

    def update(self, loss_prev: torch.Tensor, loss_curr: torch.Tensor):

        loss_prev = loss_prev.detach()
        loss_curr = loss_curr.detach()

        self.t += 1
        beta_t = 1.0 - 1.0 / self.t.float()

        if self.initialized == 0:
            self.m = loss_curr
            self.initialized = torch.tensor(1, device=loss_curr.device)
            return

        m_new = (
            (1.0 - beta_t) * loss_curr
            + beta_t * (loss_curr - loss_prev)
            + beta_t * self.m
        )

        self.m = m_new

    def compute(self):
        return self.m


