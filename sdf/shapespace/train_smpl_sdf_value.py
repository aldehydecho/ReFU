import os
import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)
from datetime import datetime
from pyhocon import ConfigFactory
from time import time
import argparse
import json
import torch
import igrutils.general as utils
from model.sample import Sampler
from model.network import gradient
from igrutils.plots import plot_surface, plot_cuts
from datasets.smpldataset import SMPLDataSet
import shutil
import global_var_fun

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

class ShapeSpaceRunner:

    def __init__(self,**kwargs):

        # config setting

        self.home_dir = os.path.abspath(os.pardir)

        if type(kwargs['conf']) == str:
            self.conf_filename = './shapespace/' + kwargs['conf']
            self.conf = ConfigFactory.parse_file(self.conf_filename)
        else:
            self.conf = kwargs['conf']

        self.expname = kwargs['expname']

        # GPU settings

        self.GPU_INDEX = kwargs['gpu_index']

        if not self.GPU_INDEX == 'ignore':
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        self.num_of_gpus = torch.cuda.device_count()

        # settings for loading an existing experiment

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join(self.home_dir, 'exps', self.expname)):
                timestamps = os.listdir(os.path.join(self.home_dir, 'exps', self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        self.exps_folder_name = 'exps'

        utils.mkdir_ifnotexists(utils.concat_home_dir(os.path.join(self.home_dir, self.exps_folder_name)))

        self.expdir = utils.concat_home_dir(os.path.join(self.home_dir, self.exps_folder_name, self.expname))
        utils.mkdir_ifnotexists(self.expdir)

        if is_continue:
            self.timestamp = timestamp
        else:
            self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())

        self.cur_exp_dir = self.timestamp
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.cur_exp_dir))

        shutil.copy('./shapespace/' + kwargs['conf'], os.path.join(self.expdir, self.cur_exp_dir))

        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.cur_exp_dir, 'codebackup'))

        global_var_fun.backup_file(global_var_fun.ROOT_DIR, os.path.join(self.expdir, self.cur_exp_dir, 'codebackup'))

        with open(os.path.join(self.expdir, self.cur_exp_dir, 'expargs.json'), "w") as outfile: 
            json.dump(kwargs, outfile)

        self.checkpoints_path = os.path.join(self.expdir, self.cur_exp_dir, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)


        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path,self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))

        self.nepochs = kwargs['nepochs']

        self.batch_size = kwargs['batch_size']

        if self.num_of_gpus > 0:
            self.batch_size *= self.num_of_gpus

        self.parallel = self.num_of_gpus > 1


        self.d_in = self.conf.get_int('train.d_in')

        self.latent_size = self.conf.get_int('train.latent_size')

        self.sdf_lambda = self.conf.get_float('network.loss.sdf_lambda')
        self.grad_lambda = self.conf.get_float('network.loss.lambda')
        self.normals_lambda = self.conf.get_float('network.loss.normals_lambda')

        self.with_normals = self.normals_lambda > 0


        self.train_dataset = SMPLDataSet(self.conf.get_string('train.dataset_path'), self.conf.get_string('train.train_file_list'), points_batch = kwargs['points_batch'], with_normals=self.with_normals)

        self.test_dataset = SMPLDataSet(self.conf.get_string('train.dataset_path'), self.conf.get_string('train.test_file_list'), points_batch = kwargs['points_batch'], with_normals=self.with_normals)

        self.num_scenes = len(self.train_dataset)

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=True,
                                                      num_workers=kwargs['threads'], drop_last=True, pin_memory=True)
        self.eval_dataloader = torch.utils.data.DataLoader(self.test_dataset,
                                                           batch_size=1,
                                                           shuffle=True,
                                                           num_workers=0, drop_last=True)

        self.network = utils.get_class(self.conf.get_string('train.network_class'))(d_in=(self.d_in+self.latent_size), **self.conf.get_config('network.inputs'))

        if self.parallel:
            self.network = torch.nn.DataParallel(self.network)

        if torch.cuda.is_available():
            self.network.cuda()

        self.lr_schedules = self.get_learning_rate_schedules(self.conf.get_list('train.learning_rate_schedule'))
        self.weight_decay = self.conf.get_float('train.weight_decay')

        # optimizer and latent settings

        self.startepoch = 0


        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.network.parameters(),
                    "lr": self.lr_schedules[0].get_learning_rate(0),
                    "weight_decay": self.weight_decay
                },
            ])


        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')


            saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.network.load_state_dict(saved_model_state["model_state_dict"])

            data = torch.load(os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])
            self.startepoch = saved_model_state['epoch'] + 1


    def run(self):

        print("running")

        train_mnfld_loss = AverageMeter()
        train_sdf_loss = AverageMeter()
        train_grad_loss = AverageMeter()
        train_normals_loss = AverageMeter()

        for epoch in range(self.startepoch, self.nepochs + 1):

            if epoch % self.conf.get_int('train.checkpoint_frequency') == 0:
                self.save_checkpoints(epoch)
                # self.plot_validation_shapes(epoch)

            # change back to train mode
            self.network.train()
            self.adjust_learning_rate(epoch)

            # start epoch
            before_epoch = time()
            for data_index,(beta, theta, mnfld_pnts, normals, sdf_pnts, sdf_values, _) in enumerate(self.train_dataloader):

                mnfld_pnts = mnfld_pnts.cuda()
                beta = beta.cuda()
                theta = theta.cuda()

                if self.with_normals:
                    normals = normals.cuda()

                sdf_pnts = sdf_pnts.cuda()
                sdf_values = sdf_values.cuda()

                sdf_values = sdf_values.reshape(-1, 1)

                # nonmnfld_pnts = self.sampler.get_points(mnfld_pnts)

                mnfld_pnts = self.add_latent(beta, theta, mnfld_pnts)
                sdf_pnts = self.add_latent(beta, theta, sdf_pnts)
                # nonmnfld_pnts = self.add_latent(beta, theta, nonmnfld_pnts)

                # forward pass

                mnfld_pnts.requires_grad_()
                sdf_pnts.requires_grad_()

                mnfld_pred = self.network(mnfld_pnts)
                sdf_pred = self.network(sdf_pnts)

                mnfld_grad = gradient(mnfld_pnts, mnfld_pred)
                sdf_grad = gradient(sdf_pnts, sdf_pred)

                # manifold loss

                mnfld_loss = (mnfld_pred.abs()).mean()

                sdf_loss = torch.abs(sdf_values - sdf_pred).mean()

                # eikonal loss

                grad_loss = ((sdf_grad.norm(2, dim=-1) - 1) ** 2).mean()

                loss = mnfld_loss + self.sdf_lambda * sdf_loss + self.grad_lambda * grad_loss

                # normals loss
                if self.with_normals:
                    normals = normals.view(-1, 3)
                    normals_loss = ((mnfld_grad - normals).abs()).norm(2, dim=1).mean()
                    loss = loss + self.normals_lambda * normals_loss
                else:
                    normals_loss = torch.zeros(1)

                batch_size = beta.shape[0]

                train_mnfld_loss.update(mnfld_loss.item(), n=batch_size)
                train_sdf_loss.update(sdf_loss.item(), n=batch_size)
                train_grad_loss.update(grad_loss.item(), n=batch_size)
                train_normals_loss.update(normals_loss.item(), n=batch_size)

                # back propagation

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                # print status
                if data_index % self.conf.get_int('train.status_frequency') == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}\tManifold loss: {:.6f}\tSdf loss: {:.6f}'
                          '\tGrad loss: {:.6f}\tNormals Loss: {:.6f}'.format(
                        epoch, data_index * self.batch_size, len(self.train_dataset), 100. * data_index / len(self.train_dataloader),
                               loss.item(), mnfld_loss.item(), sdf_loss.item(), grad_loss.item(), normals_loss.item()))


            log_table = {
                "epoch":epoch,
                "train_mnfld_loss" : train_mnfld_loss.avg,
                "train_sdf_loss" : train_sdf_loss.avg,
                "train_grad_loss" : train_grad_loss.avg,
                "train_normals_loss" : train_normals_loss.avg
                }

            with open(os.path.join(self.expdir, self.cur_exp_dir, "log.txt"), 'a') as f: #open and append
                f.write(json.dumps(log_table) + '\n')
            
            train_mnfld_loss.reset()
            train_sdf_loss.reset()
            train_grad_loss.reset()    
            train_normals_loss.reset()

            after_epoch = time()
            print('epoch time {0}'.format(str(after_epoch-before_epoch)))

    def plot_validation_shapes(self, epoch, with_cuts=False):
        # plot network validation shapes
        with torch.no_grad():

            print('plot validation epoch: ', epoch)

            self.network.eval()
            beta, theta, pnts, _,_,_, names = next(iter(self.eval_dataloader))
            pnts = utils.to_cuda(pnts)
            beta = utils.to_cuda(beta)
            theta = utils.to_cuda(theta)

            pnts = self.add_latent(beta, theta, pnts)
            latent = torch.cat([beta[0], theta[0]], 0)

            shapename = names[0]

            plot_surface(with_points=True,
                         points=pnts,
                         decoder=self.network,
                         latent=latent,
                         path=self.plots_dir,
                         epoch=epoch,
                         shapename=shapename,
                         **self.conf.get_config('plot'))

            if with_cuts:
                plot_cuts(points=pnts,
                          decoder=self.network,
                          latent=latent,
                          path=self.plots_dir,
                          epoch=epoch,
                          near_zero=False)

    def get_learning_rate_schedules(self,schedule_specs):

        schedules = []

        for schedule_specs in schedule_specs:

            if schedule_specs["Type"] == "Step":
                schedules.append(
                    utils.StepLearningRateSchedule(
                        schedule_specs["Initial"],
                        schedule_specs["Interval"],
                        schedule_specs["Factor"],
                    )
                )

            else:
                raise Exception(
                    'no known learning rate schedule of type "{}"'.format(
                        schedule_specs["Type"]
                    )
                )

        return schedules

    def add_latent(self, beta, theta, points):
        batch_size, num_of_points, dim = points.shape
        points = points.reshape(batch_size * num_of_points, dim)
        latent_inputs = torch.zeros(0).cuda()

        for ind in range(0, batch_size):
            latent_ind = torch.cat([beta[ind], theta[ind]], 0)
            latent_repeat = latent_ind.expand(num_of_points, -1)
            latent_inputs = torch.cat([latent_inputs, latent_repeat], 0)
        points = torch.cat([latent_inputs, points], 1)
        return points

    def adjust_learning_rate(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.lr_schedules[i].get_learning_rate(epoch)

    def save_checkpoints(self,epoch):

        torch.save(
            {"epoch": epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--points_batch', type=int, default=4000, help='point batch size')
    parser.add_argument('--nepoch', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='smpl_setup.conf')
    parser.add_argument('--expname', type=str, default='smpl_shapespace_sdf_value_with_cloth_shirt_male')
    parser.add_argument('--gpu', type=str, default='4,5,6,7', help='GPU to use [default: GPU ignore]')
    parser.add_argument('--threads', type=int, default=8, help='num of threads for data loader')
    parser.add_argument('--is_continue', default=True, action="store_true", help='continue')
    parser.add_argument('--timestamp', default='2021_10_26_20_18_39', type=str)
    parser.add_argument('--checkpoint', default='latest', type=str)
    

    args = parser.parse_args()

    trainrunner = ShapeSpaceRunner(
            conf=args.conf,
            batch_size=args.batch_size,
            points_batch=args.points_batch,
            nepochs=args.nepoch,
            expname=args.expname,
            gpu_index=args.gpu,
            threads=args.threads,
            is_continue=args.is_continue,
            timestamp=args.timestamp,
            checkpoint=args.checkpoint
    )

    trainrunner.run()
