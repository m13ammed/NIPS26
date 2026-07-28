import torch
import torch.nn.functional as F
from einops import rearrange


class L2Loss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(L2Loss, self).__init__()

        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class DerivLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True, shapelist=None):
        super(DerivLoss, self).__init__()

        assert d > 0 and p > 0
        self.shapelist = shapelist
        self.de_x = L2Loss(d=d, p=p, size_average=size_average, reduction=reduction)
        self.de_y = L2Loss(d=d, p=p, size_average=size_average, reduction=reduction)

    def central_diff(self, x, h1, h2, s1, s2):
        # assuming PBC
        # x: (batch, n, feats), h is the step size, assuming n = h*w
        x = rearrange(x, 'b (h w) c -> b h w c', h=s1, w=s2)
        x = F.pad(x,
                  (0, 0, 1, 1, 1, 1), mode='constant', value=0.)  # [b c t h+2 w+2]
        grad_x = (x[:, 1:-1, 2:, :] - x[:, 1:-1, :-2, :]) / (2 * h1)  # f(x+h) - f(x-h) / 2h
        grad_y = (x[:, 2:, 1:-1, :] - x[:, :-2, 1:-1, :]) / (2 * h2)  # f(x+h) - f(x-h) / 2h

        return grad_x, grad_y

    def __call__(self, out, y):
        out = rearrange(out, 'b (h w) c -> b c h w', h=self.shapelist[0], w=self.shapelist[1])
        out = out[..., 1:-1, 1:-1].contiguous()
        out = F.pad(out, (1, 1, 1, 1), "constant", 0)
        out = rearrange(out, 'b c h w -> b (h w) c')
        gt_grad_x, gt_grad_y = self.central_diff(y, 1.0 / float(self.shapelist[0]),
                                                 1.0 / float(self.shapelist[1]), self.shapelist[0], self.shapelist[1])
        pred_grad_x, pred_grad_y = self.central_diff(out, 1.0 / float(self.shapelist[0]),
                                                     1.0 / float(self.shapelist[1]), self.shapelist[0],
                                                     self.shapelist[1])
        deriv_loss = self.de_x(pred_grad_x, gt_grad_x) + self.de_y(pred_grad_y, gt_grad_y)
        return deriv_loss
