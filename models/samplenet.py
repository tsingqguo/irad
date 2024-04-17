import torch
import torch.nn.functional as F
import models
from models.mlp import MLP
from utils import make_coord


class SampleNetMLP(torch.nn.Module):
    def __init__(self, liif_path):
        super(SampleNetMLP, self).__init__()
        model_spec = torch.load(liif_path)['model']
        liif_model = models.make(model_spec, load_sd=True)
        self.encoder = liif_model.encoder
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
            # param.requires_grad = True

        imnet_in_dim = self.encoder.out_dim
        imnet_in_dim *= 9
        imnet_in_dim += 2  # attach coord
        imnet_in_dim += 2  # attach cell
        self.imnet = MLP(in_dim=imnet_in_dim, out_dim=2, hidden_list=[256, 256, 256, 256])

    def forward(self, x, coord, cell):
        feat = self.encoder(x)
        # feat_unfold
        feat = F.unfold(feat, 3, padding=1).view(
            feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
        # local_ensemble
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                # del coord_, q_feat, q_coord
                # gc.collect()
                # torch.cuda.empty_cache()

                # cell_decode:
                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([inp, rel_cell], dim=-1)

                # del rel_cell
                # gc.collect()
                # torch.cuda.empty_cache()

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)

        # local_ensemble
        t = areas[0];
        areas[0] = areas[3];
        areas[3] = t
        t = areas[1];
        areas[1] = areas[2];
        areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        x = ret
        # x = ret.repeat((1, 1, 2))
        abs_x = abs(x).mean()
        return x, abs_x
