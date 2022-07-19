from numpy import log, save

import sys
import os
import tensorboardX
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
import pickle
from datetime import datetime

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

igr_smpl_project_dir = '../sdf/code'

sys.path.append(igr_smpl_project_dir)

from pyhocon import ConfigFactory

import igrutils.general as igrutils


from models import networks
from models import ops
from dataset.static_pose_shape_final import OneStyleShapeHF
import global_var
from trainer import base_trainer

from models.smpl4garment import SMPL4Garment
from models.torch_smpl4garment import TorchSMPL4Garment
from models.tailornet_model import TailorNetModel
from dataset.static_pose_shape_final import MultiStyleShape
from tnutils.eval import AverageMeter
from tnutils.logger import TailorNetLogger
from tnutils import sio
import global_var

from models.sdf_collision_response_model import SDF_Collsion_Response_Hybrid

from models.networks import FullyConnected_SDF_Hybrid_Weight


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

class Col_Trainer():
    def __init__(self, params):
        self.params = params

        self.num_of_gpus = torch.cuda.device_count()

        self.gender = params['gender']
        self.garment_class = params['garment_class']

        self.bs = params['batch_size']
        self.note = params['note']

        # smpl for garment
        self.smpl_torch = TorchSMPL4Garment(self.gender)
        self.smpl_torch.cuda()

        # garment specific things
        with open(os.path.join(global_var.DATA_DIR, global_var.GAR_INFO_FILE), 'rb') as f:
            class_info = pickle.load(f)
        self.body_f_np = self.smpl_torch.faces.astype(np.long)
        self.garment_f_np = class_info[self.garment_class]['f']
        self.garment_f_torch = torch.tensor(self.garment_f_np.astype(np.long)).long().cuda()
        self.vert_indices = np.array(
            class_info[self.garment_class]['vert_indices'])

        self.garment_vertex_num = self.garment_f_np.max() + 1

        self.sdf_conf = ConfigFactory.parse_file(self.params['sdf_conf_file'])

        self.sdf_network = igrutils.get_class(self.sdf_conf.get_string('train.network_class'))(d_in=self.sdf_conf.get_int('train.latent_size')+self.sdf_conf.get_int('train.d_in'), **self.sdf_conf.get_config('network.inputs'))

        sdf_state_dict = torch.load(self.params['sdf_checkpoint'])

        self.sdf_network.load_state_dict({k.replace('module.', ''): v for k, v in sdf_state_dict["model_state_dict"].items()})

        self.sdf_network.cuda()

        self.sdf_hybrid_weight_model = FullyConnected_SDF_Hybrid_Weight(input_size=72+10+4, middle_output_size=self.garment_vertex_num * 10, middle_output_shape = [self.garment_vertex_num, 10], hidden_size=self.params["hybrid_mlp_hidden_size"], num_layers=self.params["hybrid_mlp_num_layers"], drop_prob = self.params["hybrid_mlp_drop_out"])

        self.sdf_hybrid_weight_model.cuda()

        self.sdf_collision_response_model = SDF_Collsion_Response_Hybrid(self.sdf_network, self.sdf_hybrid_weight_model, self.garment_class)

        self.sdf_collision_response_model.cuda()

        if self.num_of_gpus > 0:
            self.sdf_collision_response_model = torch.nn.DataParallel(self.sdf_collision_response_model)


        

        # log and backup
        timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())

        log_name = os.path.join(params['log_name'],
                                '{}_{}'.format(self.garment_class, self.gender))
        if params['shape_style'] != '':
            log_name = os.path.join(log_name, params['shape_style'], timestamp)
        else:
            log_name = os.path.join(log_name, timestamp)
        self.log_dir = sio.prepare_log_dir(log_name)
        sio.save_params(self.log_dir, params, save_name='params')

        self.iter_nums = 0 if 'iter_nums' not in params else params['iter_nums']

        

        if self.num_of_gpus > 0:
            self.smpl_torch = torch.nn.DataParallel(self.smpl_torch)


        

        # get dataset and dataloader
        self.train_dataset, self.train_loader = self.load_dataset('train', collision_info=True)
        self.test_dataset, self.test_loader = self.load_dataset('test', collision_info=True)
        print("Train dataset size", len(self.train_dataset))
        print("Test dataset size", len(self.test_dataset))

        # model and optimizer

        self.model = TailorNetModel(self.params['lf_logdir'], self.params['hf_logdir'], self.params['ss2g_logdir'], self.garment_class, self.gender)

        self.model.cuda()

        if self.num_of_gpus > 0:
            self.model = torch.nn.DataParallel(self.model)

        self.optimizer = torch.optim.Adam(
            [
            {'params':self.model.parameters()}, {'params':self.sdf_hybrid_weight_model.parameters()}] , lr=params['lr'], weight_decay=params['weight_decay'])


        #  continue training from checkpoint if provided
        if params['checkpoint']:
            ckpt_path = params['checkpoint']
            print('loading ckpt from {}'.format(ckpt_path))
            state_dict = torch.load(os.path.join(ckpt_path, 'lin.pth.tar'))
            self.model.load_state_dict(state_dict)
            state_dict = torch.load(os.path.join(ckpt_path, 'optimizer.pth.tar'))
            self.optimizer.load_state_dict(state_dict)

        self.best_error = np.inf
        self.best_epoch = -1


        self.best_collision_error = np.inf
        self.best_collision_epoch = -1
        
        self.best_generation_error = np.inf
        self.best_generation_epoch = -1

        # logger
        self.logger = tensorboardX.SummaryWriter(os.path.join(self.log_dir))
        self.csv_logger = self.get_logger()

    def load_dataset(self, split, collision_info = True):
        params = self.params
        dataset = MultiStyleShape(self.garment_class, split=split, gender=self.gender,
                                  collision_info=collision_info)
        shuffle = True if split == 'train' else False

        drop_last = False
        dataloader = DataLoader(dataset, batch_size=self.bs, num_workers=params['num_workers'], shuffle=shuffle,
                                drop_last=drop_last)
        return dataset, dataloader

    def get_logger(self):
        return TailorNetLogger()

    def evaluate_one_batch(self, inputs, eval = False):
        gt_vert_displacements, thetas, betas, gammas, _, _, _, _, _ = inputs
        
        point_num = gt_vert_displacements.shape[1]
        
        gt_vert_displacements = gt_vert_displacements.cuda()

        thetas = thetas.cuda()
        betas = betas.cuda()
        gammas = gammas.cuda()

        if eval == True:
            with torch.no_grad():
                pred_vert_displacements = self.model(thetas, betas, gammas)
                
                _, pred_garment_verts = self.smpl_torch(thetas, beta = betas, garment_d = pred_vert_displacements, garment_class = self.garment_class)
                _, gt_garment_verts = self.smpl_torch(thetas, beta = betas, garment_d = gt_vert_displacements, garment_class = self.garment_class)

        else:
            pred_vert_displacements = self.model(thetas, betas, gammas)
            
            _, pred_garment_verts = self.smpl_torch(thetas, beta = betas, garment_d = pred_vert_displacements, garment_class = self.garment_class)
            _, gt_garment_verts = self.smpl_torch(thetas, beta = betas, garment_d = gt_vert_displacements, garment_class = self.garment_class)

        modified_pred_garment_verts, modified_pred_garment_sdf_values = self.sdf_collision_response_model(thetas, betas, gammas, pred_garment_verts, eval=eval)

        generation_loss = torch.nn.functional.mse_loss(modified_pred_garment_verts, gt_garment_verts)*point_num*3

        collision_loss = torch.nn.functional.relu(-modified_pred_garment_sdf_values).sum(dim=(1,2)).mean()

        total_loss = self.params['generation_loss_weight'] * generation_loss + self.params['collision_loss_weight'] * collision_loss

        return modified_pred_garment_verts, generation_loss, collision_loss, total_loss

    def train(self, epoch):

        train_generation_loss = AverageMeter()
        train_collision_loss = AverageMeter()
        train_total_loss = AverageMeter()

        self.model.train()
        self.smpl_torch.train()
        self.sdf_collision_response_model.train()

        for i, inputs in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            modified_pred_garment_verts, generation_loss, collision_loss, total_loss = self.evaluate_one_batch(inputs)

            sample_num = modified_pred_garment_verts.shape[0]

            train_generation_loss.update(generation_loss.item(), n = sample_num)
            train_collision_loss.update(collision_loss.item(), n = sample_num)
            train_total_loss.update(total_loss.item(), n = sample_num)

            self.logger.add_scalar("train/generation_loss", generation_loss.item(), self.iter_nums)
            self.logger.add_scalar("train/collision_loss", collision_loss.item(), self.iter_nums)
            self.logger.add_scalar("train/total_loss", total_loss.item(), self.iter_nums)

            print("[epoch {} | train iter {}]\t generation loss: {:.8f}\t collision loss: {:.8f}\t total loss: {:.8f}".format(epoch, i, generation_loss.item(), collision_loss.item(), total_loss.item()))

            total_loss.backward()

            self.optimizer.step()

            self.iter_nums+=1

        self.logger.add_scalar("train_epoch/generation_loss", train_generation_loss.avg, epoch)
        self.logger.add_scalar("train_epoch/collision_loss", train_collision_loss.avg, epoch)
        self.logger.add_scalar("train_epoch/total_loss", train_total_loss.avg, epoch)

        return train_total_loss.avg, train_generation_loss.avg, train_collision_loss.avg

    
    def validate(self, epoch):
        val_generation_loss = AverageMeter()
        val_collision_loss = AverageMeter()
        val_total_loss = AverageMeter()

        self.model.eval()
        self.smpl_torch.eval()
        self.sdf_collision_response_model.eval()

        for i, inputs in enumerate(self.test_loader):

            modified_pred_garment_verts, generation_loss, collision_loss, total_loss = self.evaluate_one_batch(inputs, eval = True)

            for param in self.model.parameters():
                param.grad = None
            
            for param in self.smpl_torch.parameters():
                param.grad = None

            for param in self.sdf_collision_response_model.parameters():
                param.grad = None

            sample_num = modified_pred_garment_verts.shape[0]

            val_generation_loss.update(generation_loss.item(), n = sample_num)
            val_collision_loss.update(collision_loss.item(), n = sample_num)
            val_total_loss.update(total_loss.item(), n = sample_num)

            self.logger.add_scalar("val/generation_loss", generation_loss.item(), self.iter_nums)
            self.logger.add_scalar("val/collision_loss", collision_loss.item(), self.iter_nums)
            self.logger.add_scalar("val/total_loss", total_loss.item(), self.iter_nums)

            print("[epoch {} | val iter {}]\t generation loss: {:.8f}\t collision loss: {:.8f}\t total loss: {:.8f}".format(epoch, i, generation_loss.item(), collision_loss.item(), total_loss.item()))

            total_loss.backward()

        self.logger.add_scalar("val_epoch/generation_loss", val_generation_loss.avg, epoch)
        self.logger.add_scalar("val_epoch/collision_loss", val_collision_loss.avg, epoch)
        self.logger.add_scalar("val_epoch/total_loss", val_total_loss.avg, epoch)

        if val_total_loss.avg < self.best_error:
            self.best_error = val_total_loss.avg
            self.best_epoch = epoch
            self.save_model(epoch, name='best')

        if val_collision_loss.avg < self.best_collision_error:
            self.best_collision_error = val_collision_loss.avg
            self.best_collision_epoch = epoch
            self.save_model(epoch, name='best_collision')

        if val_generation_loss.avg < self.best_generation_error:
            self.best_generation_error = val_generation_loss.avg
            self.best_generation_epoch = epoch
            self.save_model(epoch, name="best_generation")

        return val_total_loss.avg, val_generation_loss.avg, val_collision_loss.avg




    def write_log(self):
        """Log training info once training is done."""
        if self.best_epoch >= 0:
            self.csv_logger.add_item(
                best_error=self.best_error, best_epoch=self.best_epoch, **self.params)

    def save_model(self, epoch, name = None):
        if name is None:
            torch.save(self.model.state_dict(), os.path.join(self.log_dir, str(epoch)+'_model.pth'))
            torch.save(self.sdf_hybrid_weight_model.state_dict(), os.path.join(self.log_dir, str(epoch)+'_hybrid_weight_model.pth'))
            torch.save(self.optimizer.state_dict(), os.path.join(self.log_dir, str(epoch)+'_optimizer.pth'))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.log_dir, name + '_'+str(epoch)+'_model.pth'))
            torch.save(self.sdf_hybrid_weight_model.state_dict(), os.path.join(self.log_dir, name + '_'+str(epoch)+'_hybrid_weight_model.pth'))
            torch.save(self.optimizer.state_dict(), os.path.join(self.log_dir, name + '_' + str(epoch)+'_optimizer.pth'))








        






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_config', default='')

    parser.add_argument('--gpu', type=str, default='0,1', help='GPU to use [default: GPU ignore]')

    parser.add_argument('--garment_class', default="shirt")
    parser.add_argument('--gender', default="male")
    parser.add_argument('--shape_style', default="")

    # some training hyper parameters
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--max_epoch', default=40, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--checkpoint', default="")

    parser.add_argument('--lf_logdir', default='')
    parser.add_argument('--hf_logdir', default='')
    parser.add_argument('--ss2g_logdir', default='')

    parser.add_argument('--hybrid_mlp_num_layers', default=3)
    parser.add_argument('--hybrid_mlp_hidden_size', default=1024)
    parser.add_argument('--hybrid_mlp_drop_out', type=float, default=0.2)

    parser.add_argument('--generation_loss_weight', default=1.5, type=float)
    parser.add_argument('--collision_loss_weight', default=0.5, type=float)

    # name under which experiment will be logged
    parser.add_argument('--log_name', default="refu trainer")

    # small experiment description
    parser.add_argument('--note', default="SDF collision resolving finetune")

    parser.add_argument('--sdf_conf_file', type=str, default='')
    parser.add_argument('--sdf_checkpoint', type=str, default='')

    args = parser.parse_args()
    params = args.__dict__

    os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(params['gpu'])

    # load params from local config if provided
    if os.path.exists(params['local_config']):
        print("loading config from {}".format(params['local_config']))
        with open(params['local_config']) as f:
            lc = json.load(f)
        for k, v in lc.items():
            params[k] = v

    trainer = Col_Trainer(params)

    for epoch in range(params['start_epoch'], params['max_epoch']):
        
        train_total_loss, train_generation_loss, train_collision_loss = trainer.train(epoch)
        val_total_loss, val_generation_loss, val_collision_loss = trainer.validate(epoch)
        
        log_table = {
            "epoch":epoch,
            "train_total_loss" : train_total_loss,
            "train_genertaion_loss" : train_generation_loss,
            "train_collision_loss" : train_collision_loss,
            "val_total_loss" : val_total_loss,
            "val_genertaion_loss" : val_generation_loss,
            "val_collision_loss": val_collision_loss,
            "bestval" : trainer.best_error,
            "bestepoch": trainer.best_epoch
            }

        with open(os.path.join(trainer.log_dir, "log.txt"), 'a') as f: #open and append
            f.write(json.dumps(log_table) + '\n')
        
        
        

        if epoch % 5 == 0:
            trainer.save_model(epoch)



    trainer.write_log()

