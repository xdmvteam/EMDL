import torch
import torch.nn as nn
import torch.nn.functional as F


# KL Divergence calculator. alpha shape(batch_size, num_classes)
def KL(alpha):
    ones = torch.ones([1, alpha.shape[-1]], dtype=torch.float32, device=alpha.device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (alpha - ones).mul(torch.digamma(alpha) - torch.digamma(sum_alpha)).sum(dim=1, keepdim=True)
    kl = first_term + second_term
    return kl.reshape(-1)


def loss_log(alpha, labels, kl_penalty):
    y = F.one_hot(labels.long(), alpha.shape[-1])
    log_likelihood = torch.sum(y * (torch.log(alpha.sum(dim=-1, keepdim=True)) - torch.log(alpha)), dim=-1)
    loss = log_likelihood + kl_penalty * KL((alpha - 1) * (1 - y) + 1)
    return loss  # shape=(batch_size,)


def loss_digamma(alpha, labels, kl_penalty):
    y = F.one_hot(labels.long(), alpha.shape[-1])
    log_likelihood = torch.sum(y * (torch.digamma(alpha.sum(dim=-1, keepdim=True)) - torch.digamma(alpha)), dim=-1)
    loss = log_likelihood + kl_penalty * KL((alpha - 1) * (1 - y) + 1)
    return loss


def loss_mse(alpha, labels, kl_penalty):
    y = F.one_hot(labels.long(), alpha.shape[-1])
    sum_alpha = torch.sum(alpha, dim=-1, keepdim=True)
    err = (y - alpha / sum_alpha) ** 2
    var = alpha * (sum_alpha - alpha) / (sum_alpha ** 2 * (sum_alpha + 1))
    loss = torch.sum(err + var, dim=-1)
    loss = loss + kl_penalty * KL((alpha - 1) * (1 - y) + 1)
    return loss


class DegradeLoss(nn.Module):
    def __init__(self, num_views, num_classes):
        super(DegradeLoss, self).__init__()
        self.U = nn.Parameter(torch.eye(num_classes).repeat(num_views, 1, 1))

    def forward(self, view_e, fusion_e):
        certainty = dict()
        sum_certainty = 1e-9
        for v, e in view_e.items():
            certainty[v] = torch.sum(e, dim=-1) / torch.sum(e + 1, dim=-1)
            sum_certainty += certainty[v]
        normed_u = self.U / self.U.sum(dim=-2, keepdim=True)  # Normalize
        loss = torch.zeros(fusion_e.shape[0], device=fusion_e.device)
        for v, e in view_e.items():
            view_loss = torch.sum((e - fusion_e @ normed_u[v]) ** 2, dim=-1)
            coe = certainty[v] / sum_certainty
            loss += coe * view_loss
        return loss  # shape=(batch_size,)


class InferNet(nn.Module):
    def __init__(self, sample_shape, num_classes, dropout=0.5):
        super().__init__()
        if len(sample_shape) == 1:
            self.conv = nn.Sequential()
            fc_in = sample_shape[0]
        else:  # 3
            dims = [sample_shape[0], 20, 50]
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=dims[0], out_channels=dims[1], kernel_size=5, stride=1, padding=2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=dims[1], out_channels=dims[2], kernel_size=5, stride=1, padding=2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
            )
            fc_in = sample_shape[1] // 4 * sample_shape[2] // 4 * dims[2]

        fc_dims = [fc_in, min(fc_in, 500), num_classes]
        self.fc = nn.Sequential(
            nn.Linear(fc_dims[0], fc_dims[1]),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dims[1], fc_dims[2]),
            nn.ReLU(),
        )

    def forward(self, x):
        out_conv = self.conv(x).view(x.shape[0], -1)
        evidence = self.fc(out_conv)
        return evidence


class EMDL(nn.Module):
    def __init__(self, num_train, num_valid, sample_shapes: list, num_classes, annealing=50, degrade=0.1):
        assert len(sample_shapes[0]) in [1, 3], '`sample_shape` is 1 for vector or 3 for image.'
        super().__init__()
        self.num_views = len(sample_shapes)
        self.num_classes = num_classes
        self.annealing = annealing
        self.degrade = degrade

        self.inferences = nn.ModuleList([InferNet(shape, num_classes) for shape in sample_shapes])
        self.evidences = nn.ModuleList([
            nn.Embedding(num_train, num_classes, _weight=torch.randn(num_train, num_classes).abs()),
            nn.Embedding(num_valid, num_classes, _weight=torch.randn(num_valid, num_classes).abs())
        ])
        self.degrade_loss = DegradeLoss(self.num_views, num_classes)

    def device(self):
        return self.evidences[0].weight.device

    def forward(self, sample_id, x, target=None, epoch=0):
        view_e = dict()
        for v in x.keys():
            view_e[v] = self.inferences[v](x[v].to(self.device()))
        fusion_e = F.softplus(self.evidences[0 if target is not None else 1](sample_id.to(self.device())))

        ret = {'view_e': view_e, 'fusion_e': fusion_e}
        if target is not None:  # `target` is not None denotes current mode is on training
            loss_d = self.degrade_loss(view_e, fusion_e)
            loss_c = loss_log(fusion_e + 1, target.to(self.device()), kl_penalty=min(1., epoch / self.annealing))
            loss = loss_c + self.degrade * loss_d
            ret.update({
                'loss': loss,
                'loss_c': loss_c,
                'loss_d': loss_d
            })
        return ret

    def ds_combine(self, view_e):
        fusion_e = torch.zeros_like(view_e[0])  # shape=(batch_size,num_classes)
        for e in view_e.values():
            fusion_e = fusion_e + e + fusion_e * e / self.num_classes
        return fusion_e
