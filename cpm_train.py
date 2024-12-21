import argparse
import os
import time
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.autograd import Variable
import torch.nn as nn  # <-- Pour utiliser nn.MSELoss()

import cpm_model
import cpm_data
from utils import tools

parser = argparse.ArgumentParser(description='PyTorch human pose detection')
parser.add_argument('--lsp-root', metavar='DIR',
                    help='path to root of lsp dataset')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 36)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=30, type=int,
                    metavar='N', help='print batch frequency (default: 30)')
parser.add_argument('--save-freq', '-s', default=300, type=int,
                    metavar='N', help='save batch frequency (default: 300)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--lr-decay-rate', default=1e-5, type=float,
                    help='decay rate of learning rate (default: 1e-3)')
parser.add_argument('--lr-epoch-per-decay', default=150, type=int,
                    help='epoch of per decay of learning rate (default: 150)')
parser.add_argument('--ckpt-dir', metavar='DIR',
                    help='path to save checkpoints')
parser.add_argument('--summary-dir', metavar='DIR',
                    help='path to save summary')
parser.add_argument('--weighted-loss', '--wl', action='store_true', default=False,
                    help='weighted loss')
parser.add_argument('--weighted-bandwidth', '--wb', default=0.068, type=float,
                    help='bandwidth for loss weight kde')

args = parser.parse_args()
args.cuda = (args.cuda and torch.cuda.is_available())

image_w = 368
image_h = 368

def train():
    # --- Chargement du dataset ---
    train_data = cpm_data.LSPDataset(
        args.lsp_root,
        transform=transforms.Compose([
            cpm_data.Scale(image_h, image_w),
            cpm_data.RandomHSV((0.8, 1.2), (0.8, 1.2), (25, 25)),
            cpm_data.ToTensor()
        ]),
        phase_train=True,
        weighted_loss=args.weighted_loss,
        bandwidth=args.weighted_bandwidth
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True if args.cuda else False
    )
    num_alldata = len(train_data)

    # --- Initialisation du modèle ---
    model = cpm_model.CPM(train_data.num_keypoints)
    model.train()
    if args.cuda:
        model.cuda()

    # --- Préparation de l'optimizer ---
    lr = args.lr
    params = [
        {'params': [p for n, p in model.named_parameters() if 'stage1.weight' in n], 'lr': 5 * lr},
        {'params': [p for n, p in model.named_parameters() if 'stage1.bias' in n], 'lr': 10 * lr},
        {'params': [p for n, p in model.named_parameters() if 'weight' in n and 'stage1' not in n], 'lr': lr},
        {'params': [p for n, p in model.named_parameters() if 'bias' in n and 'stage1' not in n], 'lr': 2 * lr}
    ]
    optimizer = torch.optim.SGD(params, momentum=args.momentum, weight_decay=args.weight_decay)

    # --- Instanciation de MSELoss ---
    criterion = nn.MSELoss()

    count = 0
    global_step = 0
    if args.resume:
        count, global_step, args.start_epoch = load_ckpt(model, optimizer, args.resume)

    for epoch in range(int(args.start_epoch), args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr, params,
                             args.lr_decay_rate, args.lr_epoch_per_decay)

        for batch_idx, sample in enumerate(train_loader):
            image = Variable(sample['image'].cuda() if args.cuda else sample['image'])
            gt_map = Variable(sample['gt_map'].cuda() if args.cuda else sample['gt_map'])
            center_map = Variable(sample['center_map'].cuda() if args.cuda else sample['center_map'])
            weight = Variable(sample['weight'].cuda() if args.cuda else sample['weight'])  # peut être ignoré si on n'utilise plus le weighted_loss

            optimizer.zero_grad()

            # --- Prédiction du réseau ---
            outputs = model(image, center_map)  # shape [N, 6, C, H, W]

            # Ici, on décide d'utiliser UNIQUEMENT la dernière sortie (stage 6).
            # On récupère donc outputs[:, -1, ...] qui a la shape [N, C, H, W].
            pred_final = outputs[:, -1, :, :, :]

            # --- Calcul de la perte avec nn.MSELoss() ---
            loss = criterion(pred_final, gt_map)

            # En mode anomalie, on peut déboguer plus facilement
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            count += image.data.shape[0]
            global_step += 1

            # Impression d'infos (print_freq)
            if global_step % args.print_freq == 0:
                try:
                    time_inter = time.time() - end_time
                    count_inter = count - last_count
                    print_log(global_step, epoch, count, count_inter,
                              num_alldata, loss, time_inter)
                    end_time = time.time()
                    last_count = count
                except NameError:
                    end_time = time.time()
                    last_count = count

            # Sauvegarde (save_freq)
            if (global_step % args.save_freq == 0 or global_step == 1) and args.cuda:
                save_ckpt(model, optimizer, global_step, batch_idx, count, args.batch_size,
                          num_alldata, args.weighted_loss, args.weighted_bandwidth)

    print("Training completed ")

def adjust_learning_rate(optimizer, epoch, init_lr, params, lr_decay_rate, lr_epoch_per_decay):
    """Sets the learning rate to the initial LR decayed by lr_decay_rate every lr_epoch_per_decay epochs"""
    for param_group, param in zip(optimizer.param_groups, params):
        param_group['lr'] = param['lr'] * (lr_decay_rate ** (epoch // lr_epoch_per_decay))

def print_log(global_step, epoch, count, count_inter, dataset_size, loss, time_inter):
    num_data = count % dataset_size
    num_all = dataset_size
    print('Step: {:>5} Train Epoch: {:>3} [{:>4}/{:>4} ({:3.1f}%)]    Loss: {:.6f} [{:.2f}s every {:>4} data]'.format(
        global_step, epoch + 1, num_data, num_all,
        100. * num_data / num_all, loss.item(), time_inter, count_inter))

def save_ckpt(model, optimizer, global_step, batch_idx, count,
              batch_size, num_alldata, weighted, bandwidth):
    state = {
        'global_step': global_step,
        'epoch': count / num_alldata,
        'count': count,
        'batch_size': batch_size,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'weighted': weighted,
        'bandwidth': bandwidth,
        'dataset_size': num_alldata,
    }
    ckpt_model_filename = "ckpt_epoch_{:0.2f}_batch_id_{}.pth".format(count / num_alldata, batch_idx + 1)
    path = os.path.join(args.ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>35} has been successfully saved'.format(path))

def load_ckpt(model, optimizer, model_file):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        try:
            checkpoint = torch.load(model_file)
        except AssertionError:
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.weighted_loss = checkpoint['weighted']
        args.weighted_bandwidth = checkpoint['bandwidth']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))
        return checkpoint['count'], checkpoint['global_step'], checkpoint['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        os._exit(0)

if __name__ == '__main__':
    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)
    train()
