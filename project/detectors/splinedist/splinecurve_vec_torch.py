import torch

from project.detectors.splinedist.spline_generator import SplineCurve


class SplineCurveVectorizedTorch(SplineCurve):
    def sampleSequential(self, phi):
        contour_points = torch.matmul(phi, self.coefs)
        return contour_points
