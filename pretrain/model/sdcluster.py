import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
from model.backbone.swin_transformer import SwinTransformer


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class ProjHead(nn.Module):
    def __init__(self, in_dim, use_bn=False, nlayers=3, hidden_dim=4096, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Conv2d(in_dim, bottleneck_dim, 1)
        else:
            layers = [nn.Conv2d(in_dim, hidden_dim, 1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(hidden_dim))
            else:
                pass
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Conv2d(hidden_dim, hidden_dim, 1))
                if use_bn:
                    layers.append(nn.BatchNorm2d(hidden_dim))
                else:
                    pass
                layers.append(nn.GELU())
            layers.append(nn.Conv2d(hidden_dim, bottleneck_dim, 1))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        return x


class ProjHeadLinear(nn.Module):
    def __init__(self, in_dim, use_bn=False, nlayers=3, hidden_dim=4096, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            else:
                pass
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                else:
                    pass
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        return x


class ClusterAssignment(nn.Module):
    def __init__(self, num_prototype, dim_prototype, temp=0.07, eps=1e-6):
        super().__init__()
        self.num_prototype = num_prototype
        self.dim_prototype = dim_prototype
        self.temp = temp
        self.eps = eps

        self.prototype_embed = nn.Embedding(num_prototype, dim_prototype)

    def forward(self, x):
        x_prev = x
        prototypes = self.prototype_embed(torch.arange(0, self.num_prototype, device=x.device)).unsqueeze(0).repeat(
            x.size(0), 1, 1)
        dots = torch.einsum('bkd,bdhw->bkhw', F.normalize(prototypes, dim=2), F.normalize(x, dim=1))
        attn = (dots / self.temp).softmax(dim=1) + self.eps
        prototypes = torch.einsum('bdhw,bkhw->bkd', x_prev, attn / attn.sum(dim=(2, 3), keepdim=True))
        return prototypes, dots


class SDCLuster(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dim_hidden = args['model']['dim_hidden']
        self.dim_out = args['model']['dim_out']
        self.teacher_momentum = args['model']['teacher_momentum']

        self.num_channels = 768
        self.encoder_q = SwinTransformer(in_chans=3, embed_dim=args['model']['swin']['embed_dim'],
                                         depths=args['model']['swin']['depths'],
                                         num_heads=args['model']['swin']['num_heads'])
        self.encoder_k = SwinTransformer(in_chans=3, embed_dim=args['model']['swin']['embed_dim'],
                                         depths=args['model']['swin']['depths'],
                                         num_heads=args['model']['swin']['num_heads'])

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.group_loss_weight = args['model']['group_loss_weight']
        self.student_temp = args['model']['student_temp']
        self.teacher_temp = args['model']['teacher_temp']

        self.projector_q = ProjHead(self.num_channels, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)
        self.projector_k = ProjHead(self.num_channels, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)

        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.num_prototype = args['model']['num_prototypes']

        self.grouping_q = ClusterAssignment(self.num_prototype, self.dim_out, self.teacher_temp)
        self.grouping_k = ClusterAssignment(self.num_prototype, self.dim_out, self.teacher_temp)
        self.predictor_slot = ProjHeadLinear(self.dim_out, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)

        for param_q, param_k in zip(self.grouping_q.parameters(), self.grouping_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.K = int(args['train']['num_instances'] * 1. / args['train']['batch_size'] *
                     args['train']['epochs'])
        self.k = int(args['train']['num_instances'] * 1. / args['train']['batch_size'] *
                     (args['train']['start_epoch'] - 1))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        momentum = 1. - (1. - self.teacher_momentum) * (math.cos(math.pi * self.k / self.K) + 1.) * 0.5
        self.k += 1
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        for param_q, param_k in zip(self.grouping_q.parameters(), self.grouping_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)

    def invaug(self, x, coords, flags, output_size):
        N, C, H, W = x.shape

        batch_idxs = torch.arange(N, device=coords.device).view(N, 1)
        coords_rescaled = coords.clone()
        coords_rescaled[:, 0] = coords_rescaled[:, 0] * W  # x1
        coords_rescaled[:, 2] = coords_rescaled[:, 2] * W  # x2
        coords_rescaled[:, 1] = coords_rescaled[:, 1] * H  # y1
        coords_rescaled[:, 3] = coords_rescaled[:, 3] * H  # y2
        coords_with_idxs = torch.cat([batch_idxs, coords_rescaled], dim=1)

        x_aligned = torchvision.ops.roi_align(x, coords_with_idxs, output_size, aligned=True)
        x_flipped = torch.stack([feat.flip(-1) if flag else feat for feat, flag in zip(x_aligned, flags)])
        return x_flipped

    def sinkhorn_knopp(self, teacher_output, teacher_temp, nmb_iters):
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # teacher_output = teacher_output.float()
        Q = teacher_output.permute(1, 0, 2, 3).flatten(1)
        Q = torch.exp(Q / teacher_temp)  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(nmb_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        Q *= B  # the columns must sum to 1 so that Q is an assignment

        teacher_shape = teacher_output.permute(1, 0, 2, 3).shape

        Q = Q.unflatten(1, teacher_shape[1:]).permute(1, 0, 2, 3)
        return Q

    def self_distill(self, q, k):
        q = F.log_softmax(q / self.student_temp, dim=-1)
        return torch.sum(-k * q, dim=-1).mean()

    def ctr_loss_filtered(self, q, k, score_q, score_k, tau=0.2):
        q = q.flatten(0, 1)
        k = F.normalize(k.flatten(0, 1), dim=1)

        mask_q = (torch.zeros_like(score_q).scatter_(1, score_q.argmax(1, keepdim=True), 1).sum(-1).sum(
            -1) > 0).long().detach()
        mask_k = (torch.zeros_like(score_k).scatter_(1, score_k.argmax(1, keepdim=True), 1).sum(-1).sum(
            -1) > 0).long().detach()
        mask_intersection = (mask_q * mask_k).view(-1)
        idxs_q = mask_intersection.nonzero().squeeze(-1)

        mask_k = concat_all_gather(mask_k.view(-1))
        # mask_k = mask_k.view(-1)
        idxs_k = mask_k.nonzero().squeeze(-1)

        N = k.shape[0]
        logits = torch.einsum('nc,mc->nm',
                              [F.normalize(self.predictor_slot(q[idxs_q]), dim=1), concat_all_gather(k)[idxs_k]]) / tau
        labels = mask_k.cumsum(0)[idxs_q + N * torch.distributed.get_rank()] - 1
        return F.cross_entropy(logits, labels) * (2 * tau)

    def forward(self, input):
        crops, gc_bboxes, otc_bboxes, flags = input

        loss = 0
        iter = 0

        feature_list_q = []
        feature_list_k = []
        for i in range(len(crops)):
            feature_list_q.append(self.encoder_q(crops[i])[-1])
            with torch.no_grad():  # no gradient to keys
                feature_list_k.append(self.encoder_k(crops[i])[-1])

        for i in range(self.args['transform']['nmb_samples'][0]):
            for j in range(self.args['transform']['nmb_samples'][0] + self.args['transform']['nmb_samples'][1]):
                if i >= j:
                    continue

                x1, x2 = self.projector_q(feature_list_q[i]), self.projector_q(feature_list_q[j])
                with torch.no_grad():  # no gradient to keys
                    self._momentum_update_key_encoder()  # update the key encoder
                    y1, y2 = self.projector_k(feature_list_k[i]), self.projector_k(feature_list_k[j])

                (q1, score_q1), (q2, score_q2) = self.grouping_q(x1), self.grouping_q(x2)
                coords_1 = gc_bboxes[:, i, j, :]
                coords_2 = otc_bboxes[:, j, i, :]
                q1_aligned, q2_aligned = (self.invaug(score_q1, coords_1, flags[i], (14, 14)),
                                          self.invaug(score_q2, coords_2, flags[j], (14, 14)))
                with torch.no_grad():
                    (k1, score_k1), (k2, score_k2) = self.grouping_k(y1), self.grouping_k(y2)
                    score_k1 = self.sinkhorn_knopp(score_k1, teacher_temp=0.07, nmb_iters=3)
                    score_k2 = self.sinkhorn_knopp(score_k2, teacher_temp=0.07, nmb_iters=3)

                    coords_1 = gc_bboxes[:, i, j, :]
                    coords_2 = otc_bboxes[:, j, i, :]
                    k1_aligned, k2_aligned = (self.invaug(score_k1, coords_1, flags[i], (14, 14)),
                                              self.invaug(score_k2, coords_2, flags[j], (14, 14)))

                loss += self.group_loss_weight * self.self_distill(q1_aligned.permute(0, 2, 3, 1).flatten(0, 2),
                                                                   k2_aligned.permute(0, 2, 3, 1).flatten(0, 2)) \
                        + self.group_loss_weight * self.self_distill(q2_aligned.permute(0, 2, 3, 1).flatten(0, 2),
                                                                     k1_aligned.permute(0, 2, 3, 1).flatten(0, 2))
                loss += (1. - self.group_loss_weight) * self.ctr_loss_filtered(q1, k2, score_q1, score_k2) \
                        + (1. - self.group_loss_weight) * self.ctr_loss_filtered(q2, k1, score_q2, score_k1)
                iter += 1

        loss = loss / iter
        return loss
