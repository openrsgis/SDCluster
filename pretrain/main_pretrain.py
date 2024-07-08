import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import yaml
import torch
from model.sdcluster import SDCLuster
from timm.scheduler.cosine_lr import CosineLRScheduler
from data.dataset_pretrain import DataSet_iSAID_Pretrain
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.utils as vutils
import cv2
import torch.nn.functional as F
import numpy as np
import torch.cuda.amp as amp

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def make_img_gird(img_input, col):
    '''
    :param img_tensor: numpy array, B × H × W × 3
    :param col: the column of the gird
    :return:
    '''
    out_list = []
    k = 0
    while True:
        row_list = []
        for i in range(col):
            row_list.append(img_input[k])
            k += 1
            if k == img_input.shape[0]:
                break
        out_list.append(np.concatenate(row_list, axis=1))
        if k == img_input.shape[0]:
            break
    out = np.concatenate(out_list, axis=0)
    return out


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


class Trainer():
    def __init__(self, args, rank):
        self.args = args
        self.rank = rank

        # model
        self.net = SDCLuster(args=args).to(rank)

        self.net = DDP(self.net, device_ids=[rank], find_unused_parameters=True)

        # Define Optimizer
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=args['train']['base_lr'],
                                           betas=args['train']['optimizer']['betas'],
                                           eps=args['train']['optimizer']['eps'],
                                           weight_decay=args['train']['weight_decay'])

        self.lr_scheduler = None
        self.scaler = amp.GradScaler()

    def update(self, inputs):
        self.optimizer.zero_grad()
        with amp.autocast():
            loss = self.net(inputs)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss

    def save(self):
        if self.rank == 0:
            torch.save(self.net.module.state_dict(), r'checkpoint/net.pt')
            print('save')

    def sample(self, views, iter):
        meanimg = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1).cuda()
        stdimg = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1).cuda()

        for i in range(len(views)):
            if self.rank == 1:
                vutils.save_image(views[i] * stdimg + meanimg, f'output_pretrain/{str(iter)}_img_{str(i)}.png',
                                  normalize=True)

            _, probs = self.net.module.grouping_k(
                self.net.module.projector_k(self.net.module.encoder_k(views[i])[-1]))
            out = probs.argmax(dim=1, keepdim=True)
            out = out.float()
            out = F.interpolate(out, (224, 224))
            out = out.permute(0, 2, 3, 1).repeat(1, 1, 1, 3)
            out = out.cpu().int().numpy()

            heatmap = None
            heatmap = cv2.normalize(out, heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                    dtype=cv2.CV_8U)
            heatmap = make_img_gird(heatmap, col=8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            if self.rank == 1:
                cv2.imwrite(f'output_pretrain/{str(iter)}_out_cluster_{str(i)}.png', heatmap)

    def build_scheduler(self, args, n_iter_per_epoch):
        num_steps = int(args['train']['epochs'] * n_iter_per_epoch)
        warmup_steps = int(args['train']['warmup_epochs'] * n_iter_per_epoch)

        lr_scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=num_steps,
            lr_min=args['train']['min_lr'],
            warmup_lr_init=args['train']['warmup_lr'],
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
        self.lr_scheduler = lr_scheduler


def main(rank, world_size, args, dataset):
    print('rank: ', rank)
    ddp_setup(rank=rank, world_size=world_size)

    sampler = DistributedSampler(dataset=dataset, num_replicas=world_size, rank=rank, drop_last=True)
    loader = DataLoader(dataset, batch_size=64, num_workers=16, pin_memory=True,
                        sampler=sampler)

    args['train']['num_instances'] = len(loader) * 64
    args['train']['batch_size'] = 64
    args['train']['start_epoch'] = 1

    trainer = Trainer(args, rank)
    trainer.build_scheduler(args, len(loader))

    loss_200 = 0

    iter = 1
    max_iter = len(loader) * args['train']['epochs']
    epoch = 1
    max_epoch = args['train']['epochs']

    while epoch <= max_epoch:
        loader.sampler.set_epoch(epoch - 1)

        for crops, gc_bboxes, otc_bboxes, flags in loader:
            crops = [crop.cuda(non_blocking=True) for crop in crops]
            flags = [flag.cuda(non_blocking=True) for flag in flags]
            gc_bboxes = gc_bboxes.cuda(non_blocking=True)
            otc_bboxes = otc_bboxes.cuda(non_blocking=True)

            inputs = [crops, gc_bboxes, otc_bboxes, flags]
            loss = trainer.update(inputs)

            loss_200 += loss.item()
            trainer.lr_scheduler.step_update(iter)

            if iter % 200 == 0:
                print(
                    '[%d/%d %d/%d] loss: %.4f '
                    % (iter, max_iter, epoch, max_epoch, loss_200 / 200))
                trainer.sample(crops, iter)
                loss_200 = 0
            iter += 1

        trainer.save()
        epoch += 1


if __name__ == '__main__':
    args = get_config('config/config_pretrain.yaml')
    world_size = torch.cuda.device_count()
    print('world_size: ', world_size)

    dataset = DataSet_iSAID_Pretrain(args)
    mp.spawn(main, args=(world_size, args, dataset), nprocs=world_size)
