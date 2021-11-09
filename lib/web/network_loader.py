from csbdeep.utils import normalize
from detectors.fcrn import model
from detectors.splinedist.config import Config
from detectors.splinedist.models.model2d import SplineDist2D
from lib.image.drawing import get_interpolated_points
import torch
import numpy as np


class NetworkLoader:
    def __init__(self, net_arch):
        self.net_arch = net_arch
        self.init_network()

    def init_network(self):
        if self.net_arch == "fcrn":
            self.init_fcrn_network()
        elif self.net_arch == "splinedist":
            self.init_splinedist_network()
        self.network.load_state_dict(torch.load(self.model_path))

    def predict_instances(self, image):
        if image is None:
            return {"count": 0, "outlines": []}
        if self.net_arch == "fcrn":
            image = torch.from_numpy(
                (1 / 255) * np.expand_dims(np.moveaxis(image, 2, 0), 0)
            )
            return {"count": int(torch.sum(self.network(image)).item() / 100)}
        elif self.net_arch == "splinedist":
            image = normalize(image, 1, 99.8, axis=(0, 1))
            image = image.astype(np.float32)
            results = self.network.predict_instances(image)[1]
            sampled_points = get_interpolated_points(results['coord'])
            return {"count": len(results["points"]), "outlines": sampled_points}

    def init_fcrn_network(self):
        self.model_path = (
            "models/egg_FCRN_A_150epochs_Yang-Lab-Dell2_2021-01-06"
            + " 17-04-53.765866.pth"
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        network = model.FCRN_A(input_filters=3, N=2).to(device)
        network.train(False)
        self.network = torch.nn.DataParallel(network)

    def init_splinedist_network(self):
        config_path = "configs/unet_backbone_rand_zoom.json"
        self.model_path = (
            "models/splinedist_unet_full_400epochs_"
            + "NZXT-U_2021-08-12 08-39-05.733572.pth"
        )
        config = Config(config_path, n_channel_in=3)
        self.network = SplineDist2D(config)
        self.network.cuda()
        self.network.train(False)
