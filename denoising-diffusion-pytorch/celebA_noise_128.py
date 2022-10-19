#from comet_ml import Experiment
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torchvision
import os
import errno
import shutil
import argparse
import time
import wandb

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        pass


parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', default=50, type=int,
                    help="The number of steps the scheduler takes to go from clean image to an isotropic gaussian. This is also the number of steps of diffusion.")
parser.add_argument('--train_steps', default=700000, type=int,
                    help='The number of iterations for training.')
parser.add_argument('--optim', default='Adam', type=str)
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--gradient_clipping', default=-1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--nesterov', action='store_true', default=False)

parser.add_argument('--dataset', default='celebA', type=str)
parser.add_argument('--batch_size',default=32 ,type=int)
parser.add_argument('--interval', default=100, type=int)

parser.add_argument('--save_folder', default='./results_cifar10', type=str)
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--train_routine', default='Final', type=str)
parser.add_argument('--sampling_routine', default='x0_step_down', type=str,
                    help='The choice of sampling routine for reversing the diffusion process.')
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--residual', action="store_true")
parser.add_argument('--loss_type', default='l1', type=str)
parser.add_argument('--wandb', action='store_false', default=True)


args = parser.parse_args()
print(args)

if args.dataset == 'MNIST':
    args.data_path='root_mnist/'
    args.img_size=28
    args.save_folder='./results_MNIST_'+args.optim
    args.channel = 1
elif args.dataset == 'celebA':
    args.data_path='root_celebA_128_train_new/'
    args.img_size=128
    args.save_folder='./results_celebA_'+args.optim
    args.channel = 3
elif args.dataset == 'CIFAR10':
    args.data_path='root_cifar10/'
    args.img_size=32
    args.save_folder='./results_CIFAR10_'+args.optim
    args.channel = 3

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels=args.channel,
    with_time_emb=not(args.remove_time_embed),
    residual=args.residual
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = args.img_size,
    channels = 3,
    timesteps = args.time_steps,   # number of steps
    loss_type = args.loss_type,    # L1 or L2
    train_routine = args.train_routine,
    sampling_routine = args.sampling_routine
).cuda()

import torch
#diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))

config = vars(args).copy()

trainer = Trainer(
    diffusion,
    args.data_path,
    image_size = args.img_size,
    train_batch_size = args.batch_size,
    train_lr = args.lr,
    train_num_steps = args.train_steps,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,                       # turn on mixed precision training with apex
    results_folder = args.save_folder,
    load_path = args.load_path,
    dataset = 'train',
    optim=args.optim,
    interval=args.interval,
    gradient_clipping=args.gradient_clipping,
    momentum=args.momentum,
    nesterov=args.nesterov
)

if args.wandb:
    wandb.init(config=config,
               entity=os.environ.get('WANDB_ENTITY', None),
               project=os.environ.get('WANDB_PROJECT', None),
               )

start = time.time()
total_train_time = 0
trainer.train()
total_train_time += time.time() - start

if args.wandb:
        wandb.run.summary['total_train_time'] = total_train_time