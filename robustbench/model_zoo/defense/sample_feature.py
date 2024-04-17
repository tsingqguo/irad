from models.samplenet import SampleNetMLP
import models as inr_models
from utils import make_coord
import torch


class SampleFeatureMLP(object):
    def __init__(self, device, sample_path, feature_path, height=299, width=299):
        self.device = device
        self.inr_model = inr_models.make(torch.load(feature_path)['model'], load_sd=True).to(self.device)
        self.inr_model.eval()

        self.sample_model = SampleNetMLP(feature_path).to(device)
        self.sample_model.load_state_dict(torch.load(sample_path)['model'])

        self.sample_model.eval()
        self.height = height
        self.width = width

        self.coord = make_coord((self.height, self.width)).to(self.device)
        self.cell = torch.ones_like(self.coord)
        self.cell[:, 0] *= 2 / self.height
        self.cell[:, 1] *= 2 / self.width

    def forward(self, x):
        w = self.width
        h = self.height
        with torch.no_grad():
            coord_offset, _ = self.sample_model(x, self.coord.repeat(x.shape[0], 1, 1), self.cell.repeat(x.shape[0], 1, 1))

            outputs_adv = self.inr_model((x - 0.5) / 0.5,
                                     self.coord.repeat(x.shape[0], 1, 1) + coord_offset,
                                     self.cell.repeat(x.shape[0], 1, 1))

            outputs_adv = outputs_adv * 0.5 + 0.5
            outputs_adv = outputs_adv.clamp(0, 1).view(outputs_adv.shape[0], w, h, 3).permute(0, 3, 1, 2).contiguous()

        return outputs_adv

