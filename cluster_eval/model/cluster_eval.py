import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.swin_transformer import SwinTransformer


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


class ClusterAssignment(nn.Module):
    def __init__(self, num_prototype, dim_prototype):
        super().__init__()
        self.num_prototypes = num_prototype

        self.prototype_embed = nn.Embedding(num_prototype, dim_prototype)

    def forward(self, x):
        prototypes = self.prototype_embed(torch.arange(0, self.num_prototypes, device=x.device)).unsqueeze(0).repeat(
            x.size(0), 1, 1)
        dots = torch.einsum('bkd,bdhw->bkhw', F.normalize(prototypes, dim=2), F.normalize(x, dim=1))
        return dots


class ClusterEval(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dim_hidden = args['model']['dim_hidden']
        self.dim_out = args['model']['dim_out']
        self.num_prototypes = args['model']['num_prototypes']
        self.num_channels = 768

        self.encoder = SwinTransformer(in_chans=3, embed_dim=args['embed_dim'], depths=args['depths'],
                                       num_heads=args['num_heads'])

        self.projector = ProjHead(self.num_channels, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)

        self.assign = ClusterAssignment(self.num_prototypes, self.dim_out)

    def forward(self, x):
        with torch.no_grad():
            probs = self.assign(self.projector(self.encoder(x)[-1]))
            return probs
