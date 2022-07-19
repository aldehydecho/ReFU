import sys
import os

import numpy as np
import pickle
from numpy import save

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import global_var
from tnutils.io import write_obj


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def evaluate():
    """Evaluate TailorNet (or any model for that matter) on test set."""
    from dataset.static_pose_shape_final import MultiStyleShape
    import torch
    from torch.utils.data import DataLoader
    from tnutils.eval import AverageMeter
    from models import ops

    gender = 'female'
    garment_class = 'skirt'

    dataset = MultiStyleShape(garment_class=garment_class, gender=gender, split='test')
    dataloader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False, drop_last=False)
    print(len(dataset))

    val_dist = AverageMeter()
    from models.tailornet_model import get_best_runner as tn_runner
    runner = tn_runner(garment_class, gender)
    # from trainer.base_trainer import get_best_runner as baseline_runner
    # runner = baseline_runner("/BS/cpatel/work/data/learn_anim/{}_{}_weights/tn_orig_baseline/{}_{}".format(garment_class, gender, garment_class, gender))

    device = torch.device('cuda:0')
    with torch.no_grad():
        for i, inputs in enumerate(dataloader):
            gt_verts, thetas, betas, gammas, _ = inputs

            thetas, betas, gammas = ops.mask_inputs(thetas, betas, gammas, garment_class)
            gt_verts = gt_verts.to(device)
            thetas = thetas.to(device)
            betas = betas.to(device)
            gammas = gammas.to(device)
            pred_verts = runner.forward(thetas=thetas, betas=betas, gammas=gammas).view(gt_verts.shape)

            dist = ops.verts_dist(gt_verts, pred_verts) * 1000.
            val_dist.update(dist.item(), gt_verts.shape[0])
            print(i, len(dataloader))
    print(val_dist.avg)


def evaluate_save():
    """Evaluate TailorNet (or any model for that matter) on test set."""
    from dataset.static_pose_shape_final import MultiStyleShape
    import torch
    from torch.utils.data import DataLoader
    from tnutils.eval import AverageMeter
    from models import ops
    from models.smpl4garment import SMPL4Garment
    
    import os

    gender = 'female'
    garment_class = 'skirt'
    smpl = SMPL4Garment(gender)
    vis_freq = 512
    log_dir = "/BS/cpatel/work/code_test2/try"

    dataset = MultiStyleShape(garment_class=garment_class, gender=gender, split='test')
    dataloader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False, drop_last=False)
    print(len(dataset))

    val_dist = AverageMeter()
    from models.tailornet_model import get_best_runner as tn_runner
    runner = tn_runner(garment_class, gender)
    # from trainer.base_trainer import get_best_runner as baseline_runner
    # runner = baseline_runner("/BS/cpatel/work/data/learn_anim/{}_{}_weights/tn_orig_baseline/{}_{}".format(garment_class, gender, garment_class, gender))

    device = torch.device('cuda:0')
    with torch.no_grad():
        for i, inputs in enumerate(dataloader):
            gt_verts, thetas, betas, gammas, idxs = inputs

            thetas, betas, gammas = ops.mask_inputs(thetas, betas, gammas, garment_class)
            gt_verts = gt_verts.to(device)
            thetas = thetas.to(device)
            betas = betas.to(device)
            gammas = gammas.to(device)
            pred_verts = runner.forward(thetas=thetas, betas=betas, gammas=gammas).view(gt_verts.shape)

            for lidx, idx in enumerate(idxs):
                if idx % vis_freq != 0:
                    continue
                theta = thetas[lidx].cpu().numpy()
                beta = betas[lidx].cpu().numpy()
                pred_vert = pred_verts[lidx].cpu().numpy()
                gt_vert = gt_verts[lidx].cpu().numpy()

                body_m, pred_m = smpl.run(theta=theta, garment_d=pred_vert,
                                               beta=beta,
                                               garment_class=garment_class)
                _, gt_m = smpl.run(theta=theta, garment_d=gt_vert,
                                        beta=beta,
                                        garment_class=garment_class)

                save_dir = log_dir
                pred_m.write_ply(
                    os.path.join(save_dir, "pred_{}.ply".format(idx)))
                gt_m.write_ply(os.path.join(save_dir, "gt_{}.ply".format(idx)))
                body_m.write_ply(
                    os.path.join(save_dir, "body_{}.ply".format(idx)))

    print(val_dist.avg)

def evaluate_save_obj():
    """Evaluate TailorNet (or any model for that matter) on test set."""
    from dataset.static_pose_shape_final import MultiStyleShape
    import torch
    from torch.utils.data import DataLoader
    from tnutils.eval import AverageMeter
    from models import ops
    from models.smpl4garment import SMPL4Garment
    import os
    import os.path as osp

    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    gender = 'male'
    garment_class = 'shirt'
    smpl = SMPL4Garment(gender)
    # vis_freq = 512
    save_dir = osp.join("/home/code-base/user_space/TailorNet_eval_results_new", garment_class+'_'+gender+'_newtailornet')


    # from models.torch_smpl4garment import TorchSMPL4Garment
    # smpl_torch = TorchSMPL4Garment(gender)
    # smpl_torch.cuda()

    body_save_dir = osp.join(save_dir, 'body')
    garment_save_dir = osp.join(save_dir, 'garment')

    predict_garment_save_dir = osp.join(garment_save_dir, 'predict')
    gt_garment_save_dir = osp.join(garment_save_dir, 'gt')

    os.makedirs(body_save_dir, exist_ok=True)
    os.makedirs(predict_garment_save_dir, exist_ok=True)
    os.makedirs(gt_garment_save_dir, exist_ok=True)

    dataset = MultiStyleShape(garment_class=garment_class, gender=gender, split='test', collision_info=True)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False, drop_last=False)
    print(len(dataset))

    # val_dist = AverageMeter()
    from models.tailornet_model import get_best_runner as tn_runner
    # runner = tn_runner(garment_class, gender)
    runner = tn_runner(garment_class, gender, \
                        lf_logdir='/home/code-base/user_space/TailorNet/log/tn_lf_100', \
                        hf_logdir='/home/code-base/user_space/TailorNet/log/tn_hf_800', \
                        ss2g_logdir='/home/code-base/user_space/TailorNet/log/tn_ss2g_400')
    # from trainer.base_trainer import get_best_runner as baseline_runner
    # runner = baseline_runner("/scratch-space/TailorNet/log/tn_baseline/shirt_male")

    device = torch.device('cuda:0')
    with torch.no_grad():
    # for _ in range(0, 1):
        for i, inputs in enumerate(dataloader):
            gt_verts, thetas, betas, gammas, idxs = inputs

            # thetas, betas, gammas = ops.mask_inputs(thetas, betas, gammas, garment_class)
            # thetas.requires_grad = True
            # betas.requires_grad = True
            # gammas.requires_grad = True
            gt_verts = gt_verts.to(device)
            thetas = thetas.to(device)
            betas = betas.to(device)
            gammas = gammas.to(device)
            pred_verts = runner.forward(thetas=thetas, betas=betas, gammas=gammas).view(gt_verts.shape)

            for lidx, idx in enumerate(idxs):
                # if idx % vis_freq != 0:
                #     continue
                # body_m_torch, pred_m_torch = smpl_torch.forward(thetas[lidx].unsqueeze(0), beta = betas[lidx].unsqueeze(0), garment_d = pred_verts[lidx].unsqueeze(0),
                #                                                 garment_class = garment_class)

                # print(body_m_torch, pred_m_torch)

                theta = thetas[lidx].cpu().numpy()
                beta = betas[lidx].cpu().numpy()
                pred_vert = pred_verts[lidx].cpu().numpy()
                gt_vert = gt_verts[lidx].cpu().numpy()

                body_m, pred_m = smpl.run(theta=theta, garment_d=pred_vert,
                                               beta=beta,
                                               garment_class=garment_class)

                
            
                _, gt_m = smpl.run(theta=theta, garment_d=gt_vert,
                                        beta=beta,
                                        garment_class=garment_class)

                # pred_m.write_ply(
                #     os.path.join(save_dir, "pred_{}.ply".format(idx)))
                # gt_m.write_ply(os.path.join(save_dir, "gt_{}.ply".format(idx)))
                # body_m.write_ply(
                #     os.path.join(save_dir, "body_{}.ply".format(idx)))

                pred_m.write_obj(
                    os.path.join(predict_garment_save_dir, "{}.obj".format(idx)))
                gt_m.write_obj(os.path.join(gt_garment_save_dir, "{}.obj".format(idx)))
                body_m.write_obj(
                    os.path.join(body_save_dir, "{}.obj".format(idx)))

    # print(val_dist.avg)

def error_evaluate():
    """Evaluate TailorNet (or any model for that matter) on test set."""
    from dataset.static_pose_shape_final import MultiStyleShape
    import torch
    from torch.utils.data import DataLoader
    from tnutils.eval import AverageMeter
    from models import ops
    from models.smpl4garment import SMPL4Garment
    from models.torch_smpl4garment import TorchSMPL4Garment
    import os
    import os.path as osp

    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    gender = 'male'
    garment_class = 'shirt'
    # smpl = SMPL4Garment(gender)
    smpl_torch = TorchSMPL4Garment(gender)
    smpl_torch.cuda()
    # vis_freq = 512

    dataset = MultiStyleShape(garment_class=garment_class, gender=gender, split='test', collision_info=True)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False, drop_last=False)
    print(len(dataset))

    # val_dist = AverageMeter()
    from models.tailornet_model import get_best_runner as tn_runner
    # runner = tn_runner(garment_class, gender)
    runner = tn_runner(garment_class, gender, \
                        lf_logdir='/home/code-base/user_space/TailorNet/log/tn_lf_100', \
                        hf_logdir='/home/code-base/user_space/TailorNet/log/tn_hf_800', \
                        ss2g_logdir='/home/code-base/user_space/TailorNet/log/tn_ss2g_400')
    # from trainer.base_trainer import get_best_runner as baseline_runner
    # runner = baseline_runner("/scratch-space/TailorNet/log/tn_baseline/shirt_male")

    val_generation_loss = AverageMeter()



    device = torch.device('cuda:0')
    with torch.no_grad():
    # for _ in range(0, 1):
        for i, inputs in enumerate(dataloader):
            gt_vert_displacements, thetas, betas, gammas, idxs = inputs

            point_num = gt_vert_displacements.shape[1]

            thetas, betas, gammas = ops.mask_inputs(thetas, betas, gammas, garment_class)

            gt_vert_displacements = gt_vert_displacements.to(device)
            thetas = thetas.to(device)
            betas = betas.to(device)
            gammas = gammas.to(device)

            pred_vert_displacements = runner.forward(thetas=thetas, betas=betas, gammas=gammas).view(gt_vert_displacements.shape)

            body_verts, gt_verts = smpl_torch.forward(thetas, beta = betas, garment_d = gt_vert_displacements, garment_class = garment_class)
            body_verts, pred_verts = smpl_torch.forward(thetas, beta = betas, garment_d = pred_vert_displacements, garment_class = garment_class)

            generation_loss = torch.nn.functional.mse_loss(pred_verts, gt_verts)*point_num*3

            val_generation_loss.update(generation_loss.item(), n = pred_verts.shape[0])

    
    print('average error {:.8f}'.format(val_generation_loss.avg))

def torch_smpl_output(network_name, on_train=False):
    """Evaluate TailorNet (or any model for that matter) on test set."""
    from dataset.static_pose_shape_final import MultiStyleShape
    import torch
    from torch.utils.data import DataLoader
    from tnutils.eval import AverageMeter
    from models import ops
    from models.smpl4garment import SMPL4Garment
    from models.torch_smpl4garment import TorchSMPL4Garment
    import os
    import os.path as osp
    from models.soft_collision_loss import Soft_Collision_Loss

    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    gender = 'female'
    garment_class = 'skirt'
    # smpl = SMPL4Garment(gender)
    smpl_torch = TorchSMPL4Garment(gender)
    smpl_torch.cuda()
    # vis_freq = 512

    if on_train == True:
        dataset = MultiStyleShape(garment_class=garment_class, gender=gender, split='train', collision_info=True)
    else:
        dataset = MultiStyleShape(garment_class=garment_class, gender=gender, split='test', collision_info=True)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False, drop_last=False)
    print(len(dataset))

    save_dir = osp.join("/mnt/session_space/TailorNet_eval_results", garment_class+'_'+gender, network_name)

    body_save_dir = osp.join(save_dir, 'body')
    garment_save_dir = osp.join(save_dir, 'garment')

    predict_garment_save_dir = osp.join(garment_save_dir, 'predict')
    gt_garment_save_dir = osp.join(garment_save_dir, 'gt')

    os.makedirs(body_save_dir, exist_ok=True)
    os.makedirs(predict_garment_save_dir, exist_ok=True)
    os.makedirs(gt_garment_save_dir, exist_ok=True)

    # val_dist = AverageMeter()
    from models.tailornet_model import get_best_runner as tn_runner
    # runner = tn_runner(garment_class, gender)
    runner = tn_runner(garment_class, gender, \
                        lf_logdir='/home/code-base/user_space/TailorNet/log/tn_lf_100', \
                        hf_logdir='/home/code-base/user_space/TailorNet/log/tn_hf_800', \
                        ss2g_logdir='/home/code-base/user_space/TailorNet/log/tn_ss2g_400')
    # from trainer.base_trainer import get_best_runner as baseline_runner
    # runner = baseline_runner("/scratch-space/TailorNet/log/tn_baseline/shirt_male")

    # garment specific things
    with open(os.path.join(global_var.DATA_DIR, global_var.GAR_INFO_FILE), 'rb') as f:
        class_info = pickle.load(f)
    body_f_np = smpl_torch.faces.astype(np.long)
    garment_f_np = class_info[garment_class]['f']

    val_generation_loss = AverageMeter()
    val_collision_loss = AverageMeter()
    val_collision_percentage = AverageMeter()

    soft_collision_loss_model = Soft_Collision_Loss(body_f_np)




    device = torch.device('cuda:0')
    with torch.no_grad():
        for i, inputs in enumerate(dataloader):
            gt_vert_displacements, thetas, betas, gammas, idxs, _, _, _, _ = inputs

            point_num = gt_vert_displacements.shape[1]

            # thetas, betas, gammas = ops.mask_inputs(thetas, betas, gammas, garment_class)

            gt_vert_displacements = gt_vert_displacements.to(device)
            thetas = thetas.to(device)
            betas = betas.to(device)
            gammas = gammas.to(device)

            pred_vert_displacements = runner.forward(thetas=thetas, betas=betas, gammas=gammas).view(gt_vert_displacements.shape)

            body_verts, gt_verts = smpl_torch.forward(thetas, beta = betas, garment_d = gt_vert_displacements, garment_class = garment_class)
            body_verts, pred_verts = smpl_torch.forward(thetas, beta = betas, garment_d = pred_vert_displacements, garment_class = garment_class)

            generation_loss = ops.verts_dist(pred_verts, gt_verts) * 1000.

            val_generation_loss.update(generation_loss.item(), n = pred_verts.shape[0])

            collision_loss, collision_percentage = soft_collision_loss_model(pred_verts, body_verts, output_collision_percentage=True)

            collision_loss = collision_loss.mean()
            collision_percentage = collision_percentage.mean()

            val_collision_loss.update(collision_loss.item(), n = pred_verts.shape[0])
            val_collision_percentage.update(collision_percentage.item(), n = pred_verts.shape[0])

            # for lidx, idx in enumerate(idxs):

            #     write_obj(body_verts[lidx].cpu().numpy(), body_f_np, osp.join(body_save_dir, str(idx.item())+'.obj'))
            #     # write_obj(body_verts[lidx].cpu().numpy(), body_f_np, osp.join(body_save_dir, shape_idx[lidx]+'_'+style_idx[lidx]+'_'+str(int(seq_items[lidx]))+'_'+str(idx.item())+'.obj'))

            #     write_obj(pred_verts[lidx].cpu().numpy(), garment_f_np, osp.join(predict_garment_save_dir, str(idx.item())+'.obj'))

            #     # write_obj(gt_verts[lidx].cpu().numpy(), garment_f_np, osp.join(gt_garment_save_dir, shape_idx[lidx]+'_'+style_idx[lidx]+'_'+str(int(seq_items[lidx]))+'_'+str(idx.item())+'.obj'))

            #     write_obj(gt_verts[lidx].cpu().numpy(), garment_f_np, osp.join(gt_garment_save_dir, str(idx.item())+'.obj'))


    
    print('average error {:.8f}'.format(val_generation_loss.avg))
    print('average collision loss {:.8f}'.format(val_collision_loss.avg))
    print('average collision percentage {:.8f}'.format(val_collision_percentage.avg))


def torch_smpl_evaulation(on_train=False):
    """Evaluate TailorNet (or any model for that matter) on test set."""
    from dataset.static_pose_shape_final import MultiStyleShape
    import torch
    from torch.utils.data import DataLoader
    from tnutils.eval import AverageMeter
    from models import ops
    from models.smpl4garment import SMPL4Garment
    from models.torch_smpl4garment import TorchSMPL4Garment
    import os
    import os.path as osp
    from models.soft_collision_loss import Soft_Collision_Loss

    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    gender = 'male'
    garment_class = 'short-pant'
    smpl_torch = TorchSMPL4Garment(gender)
    smpl_torch.cuda()

    if on_train == True:
        dataset = MultiStyleShape(garment_class=garment_class, gender=gender, split='train', collision_info=True)
    else:
        dataset = MultiStyleShape(garment_class=garment_class, gender=gender, split='test', collision_info=True)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False, drop_last=False)
    print(len(dataset))

    # val_dist = AverageMeter()
    from models.tailornet_model import get_best_runner as tn_runner
    # runner = tn_runner(garment_class, gender)
    runner = tn_runner(garment_class, gender, \
                        lf_logdir='/home/code-base/user_space/TailorNet/log/tn_lf_100', \
                        hf_logdir='/home/code-base/user_space/TailorNet/log/tn_hf_800', \
                        ss2g_logdir='/home/code-base/user_space/TailorNet/log/tn_ss2g_400')
    # from trainer.base_trainer import get_best_runner as baseline_runner
    # runner = baseline_runner("/scratch-space/TailorNet/log/tn_baseline/shirt_male")

    # garment specific things
    with open(os.path.join(global_var.DATA_DIR, global_var.GAR_INFO_FILE), 'rb') as f:
        class_info = pickle.load(f)
    body_f_np = smpl_torch.faces.astype(np.long)
    garment_f_np = class_info[garment_class]['f']

    val_generation_loss = AverageMeter()
    val_collision_loss = AverageMeter()
    val_collision_percentage = AverageMeter()

    soft_collision_loss_model = Soft_Collision_Loss(body_f_np)




    device = torch.device('cuda:0')
    with torch.no_grad():
        for i, inputs in enumerate(dataloader):
            gt_vert_displacements, thetas, betas, gammas, idxs, _, _, _, _ = inputs

            point_num = gt_vert_displacements.shape[1]

            gt_vert_displacements = gt_vert_displacements.to(device)
            thetas = thetas.to(device)
            betas = betas.to(device)
            gammas = gammas.to(device)

            pred_vert_displacements = runner.forward(thetas=thetas, betas=betas, gammas=gammas).view(gt_vert_displacements.shape)

            body_verts, gt_verts = smpl_torch.forward(thetas, beta = betas, garment_d = gt_vert_displacements, garment_class = garment_class)
            body_verts, pred_verts = smpl_torch.forward(thetas, beta = betas, garment_d = pred_vert_displacements, garment_class = garment_class)

            generation_loss = ops.verts_dist(pred_verts, gt_verts) * 1000.

            val_generation_loss.update(generation_loss.item(), n = pred_verts.shape[0])

            collision_loss, collision_percentage = soft_collision_loss_model(pred_verts, body_verts, output_collision_percentage=True)

            collision_loss = collision_loss.mean()
            collision_percentage = collision_percentage.mean()

            val_collision_loss.update(collision_loss.item(), n = pred_verts.shape[0])
            val_collision_percentage.update(collision_percentage.item(), n = pred_verts.shape[0])


    print(point_num)
    print('average error {:.8f}'.format(val_generation_loss.avg))
    print('average collision loss {:.8f}'.format(val_collision_loss.avg))
    print('average collision percentage {:.8f}'.format(val_collision_percentage.avg))


if __name__ == '__main__':
    # evaluate()
    # new_evaluate_save()
    # error_evaluate()
    # torch_smpl_output('baselinetailornet_torch')
    torch_smpl_evaulation()
    # evaluate_save_obj()
