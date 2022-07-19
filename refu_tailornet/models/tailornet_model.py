import torch
import numpy as np
import os

import global_var
from trainer.lf_trainer import get_best_runner as lf_runner
from trainer.hf_trainer import get_best_runner as hf_runner
from trainer.ss2g_trainer import get_best_runner as ss2g_runner
from dataset.canonical_pose_dataset import ShapeStyleCanonPose

from trainer.lf_trainer import get_best_model as lf_model
from trainer.hf_trainer import get_best_model as hf_model
from trainer.ss2g_trainer import get_best_model as ss2g_model


class TailorNetTestModel(object):
    """Main TailorNet model class.
    This class provides API for TailorNet prediction given trained models of low
    frequency predictor, pivot high frequency predictors and ss2g predictor.
    """
    def __init__(self, lf_logdir, hf_logdir, ss2g_logdir, garment_class, gender):
        self.gender = gender
        self.garment_class = garment_class
        self.lf_logdir = lf_logdir
        self.hf_logdir = hf_logdir
        self.ss2g_logdir = ss2g_logdir
        print("USING LF LOG DIR: ", lf_logdir)
        print("USING HF LOG DIR: ", hf_logdir)
        print("USING SS2G LOG DIR: ", ss2g_logdir)

        pivots_ds = ShapeStyleCanonPose(garment_class=garment_class, gender=gender,
                                        shape_style_list_path='pivots.txt')

        self.train_betas = pivots_ds.betas.cuda()
        self.train_gammas = pivots_ds.gammas.cuda()
        self.basis = pivots_ds.unpose_v.cuda()
        self.train_pivots = pivots_ds.ss_list

        self.hf_runners = [
            hf_runner("{}/{}_{}".format(hf_logdir, shape_idx, style_idx))
            for shape_idx, style_idx in self.train_pivots
        ]
        self.lf_runner = lf_runner(lf_logdir)
        self.ss2g_runner = ss2g_runner(ss2g_logdir)

    def forward(self, thetas, betas, gammas, ret_separate=False):
        inp_type = type(thetas)
        inp_device = None if inp_type == np.ndarray else thetas.device
        bs = thetas.shape[0]

        if isinstance(thetas, np.ndarray):
            thetas = torch.from_numpy(thetas.astype(np.float32))
            betas = torch.from_numpy(betas.astype(np.float32))
            gammas = torch.from_numpy(gammas.astype(np.float32))

        with torch.no_grad():
            pred_disp_hf_pivot = torch.stack([
                rr.forward(thetas.cuda(), betas.cuda(),
                           gammas.cuda()).view(bs, -1, 3)
                for rr in self.hf_runners
            ]).transpose(0, 1)

        pred_disp_hf = self.interp4(thetas, betas, gammas, pred_disp_hf_pivot, sigma=0.01)
        pred_disp_lf = self.lf_runner.forward(thetas, betas, gammas).view(bs, -1, 3)

        if inp_type == np.ndarray:
            pred_disp_hf = pred_disp_hf.cpu().numpy()
            pred_disp_lf = pred_disp_lf.cpu().numpy()
        else:
            pred_disp_hf = pred_disp_hf.to(inp_device)
            pred_disp_lf = pred_disp_lf.to(inp_device)
        if ret_separate:
            return pred_disp_lf, pred_disp_hf
        else:
            return pred_disp_lf + pred_disp_hf

    def interp4(self, thetas, betas, gammas, pred_disp_pivot, sigma=0.5):
        """RBF interpolation with distance by SS2G."""
        # disp for given shape-style in canon pose
        bs = pred_disp_pivot.shape[0]
        rest_verts = self.ss2g_runner.forward(betas=betas, gammas=gammas).view(bs, -1, 3)
        # distance of given shape-style from pivots in terms of displacement
        # difference in canon pose
        dist = rest_verts.unsqueeze(1) - self.basis.unsqueeze(0)
        dist = (dist ** 2).sum(-1).mean(-1) * 1000.

        # compute normalized RBF distance
        weight = torch.exp(-dist/sigma)
        weight = weight / weight.sum(1, keepdim=True)

        # interpolate using weights
        pred_disp = (pred_disp_pivot * weight.unsqueeze(-1).unsqueeze(-1)).sum(1)

        return pred_disp


class TailorNetModel(torch.nn.Module):
    def __init__(self, lf_logdir, hf_logdir, ss2g_logdir, garment_class, gender):
        super(TailorNetModel, self).__init__()
        self.gender = gender
        self.garment_class = garment_class

        lf_logdir = os.path.join(lf_logdir, "{}_{}".format(garment_class, gender))
        hf_logdir = os.path.join(hf_logdir, "{}_{}".format(garment_class, gender))
        ss2g_logdir = os.path.join(ss2g_logdir, "{}_{}".format(garment_class, gender))

        self.lf_logdir = lf_logdir
        self.hf_logdir = hf_logdir
        self.ss2g_logdir = ss2g_logdir
        print("USING LF LOG DIR: ", lf_logdir)
        print("USING HF LOG DIR: ", hf_logdir)
        print("USING SS2G LOG DIR: ", ss2g_logdir)

        pivots_ds = ShapeStyleCanonPose(garment_class=garment_class, gender=gender,
                                        shape_style_list_path='pivots.txt')

        self.train_betas = pivots_ds.betas.cuda()
        self.train_gammas = pivots_ds.gammas.cuda()
        self.basis = pivots_ds.unpose_v.cuda()
        self.train_pivots = pivots_ds.ss_list

        self.hf_models = torch.nn.ModuleList([
            hf_model("{}/{}_{}".format(hf_logdir, shape_idx, style_idx))
            for shape_idx, style_idx in self.train_pivots
        ])
        self.lf_model = lf_model(lf_logdir)
        self.ss2g_model = ss2g_model(ss2g_logdir)

    def forward(self, thetas, betas, gammas):
        # inp_type = type(thetas)
        # inp_device = None if inp_type == np.ndarray else thetas.device
        bs = thetas.shape[0]

        # if isinstance(thetas, np.ndarray):
        #     thetas = torch.from_numpy(thetas.astype(np.float32))
        #     betas = torch.from_numpy(betas.astype(np.float32))
        #     gammas = torch.from_numpy(gammas.astype(np.float32))

        # with torch.no_grad():
        pred_disp_hf_pivot = torch.stack([
            model(thetas.cuda(), betas.cuda(),
                        gammas.cuda()).view(bs, -1, 3)
            for model in self.hf_models
        ]).transpose(0, 1)

        # def interp4(thetas, betas, gammas, pred_disp_pivot, sigma=0.5):
        #     """RBF interpolation with distance by SS2G."""
        #     # disp for given shape-style in canon pose
        #     bs = pred_disp_pivot.shape[0]
        #     rest_verts = self.ss2g_model(betas=betas, gammas=gammas).view(bs, -1, 3)
        #     # distance of given shape-style from pivots in terms of displacement
        #     # difference in canon pose



        #     dist = rest_verts.unsqueeze(1) - self.basis.unsqueeze(0)
        #     dist = (dist ** 2).sum(-1).mean(-1) * 1000.

        #     # compute normalized RBF distance
        #     weight = torch.exp(-dist/sigma)
        #     weight = weight / weight.sum(1, keepdim=True)

        #     # interpolate using weights
        #     pred_disp = (pred_disp_pivot * weight.unsqueeze(-1).unsqueeze(-1)).sum(1)

        #     return pred_disp

        # pred_disp_hf = interp4(thetas, betas, gammas, pred_disp_hf_pivot, sigma=0.01)
        pred_disp_hf = self.interp4(thetas, betas, gammas, pred_disp_hf_pivot, sigma=0.01)
        pred_disp_lf = self.lf_model.forward(thetas, betas, gammas).view(bs, -1, 3)

        return pred_disp_lf + pred_disp_hf

    def interp4(self, thetas, betas, gammas, pred_disp_pivot, sigma=0.5):
        """RBF interpolation with distance by SS2G."""
        # disp for given shape-style in canon pose
        bs = pred_disp_pivot.shape[0]
        rest_verts = self.ss2g_model(betas=betas, gammas=gammas).view(bs, -1, 3)
        # distance of given shape-style from pivots in terms of displacement
        # difference in canon pose
        dist = rest_verts.unsqueeze(1).to(self.basis.device) - self.basis.unsqueeze(0)
        dist = (dist ** 2).sum(-1).mean(-1) * 1000.

        dist = dist.to(pred_disp_pivot.device)

        # compute normalized RBF distance
        weight = torch.exp(-dist/sigma)
        weight = weight / weight.sum(1, keepdim=True)

        # interpolate using weights
        pred_disp = (pred_disp_pivot * weight.unsqueeze(-1).unsqueeze(-1)).sum(1)

        return pred_disp


def get_best_runner(garment_class='t-shirt', gender='female', lf_logdir=None, hf_logdir=None, ss2g_logdir=None):
    """Helper function to get TailorNet runner."""
    if lf_logdir is None:
        lf_logdir = os.path.join(global_var.MODEL_WEIGHTS_PATH, "{}_{}_weights/tn_orig_lf".format(garment_class, gender))
    if hf_logdir is None:
        hf_logdir = os.path.join(global_var.MODEL_WEIGHTS_PATH, "{}_{}_weights/tn_orig_hf".format(garment_class, gender))
    if ss2g_logdir is None:
        ss2g_logdir = os.path.join(global_var.MODEL_WEIGHTS_PATH, "{}_{}_weights/tn_orig_ss2g".format(garment_class, gender))

    lf_logdir = os.path.join(lf_logdir, "{}_{}".format(garment_class, gender))
    hf_logdir = os.path.join(hf_logdir, "{}_{}".format(garment_class, gender))
    ss2g_logdir = os.path.join(ss2g_logdir, "{}_{}".format(garment_class, gender))
    runner = TailorNetTestModel(lf_logdir, hf_logdir, ss2g_logdir, garment_class, gender)
    return runner


if __name__ == '__main__':
    # gender = 'male'
    # garment_class = 't-shirt'
    # runner = get_best_runner(garment_class, gender)
    pass