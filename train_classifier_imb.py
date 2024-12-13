import argparse
import os
import random
import shutil
import time
import warnings
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
from LabelSmoothing import LabelSmoothingLoss
from torchvision.models import resnet50
import pytorch_warmup
from utils import CutMix, mixup_data
from randaugment import rand_augment_transform
import moco_loader as moco_loader


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, image_folder, annotation_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.data = self.load_data(annotation_file)
        self.classes = self.get_classes()

    def load_data(self, annotation_file):
        with open(annotation_file, 'r') as file:
            data = file.readlines()
        data = [line.strip().split() for line in data]
        return data

    def get_classes(self):
        classes = set()
        for _, label in self.data:
            classes.add(label)
        return sorted(list(classes))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = int(label)
        return image, label
    
# 自定义增强数据集类
class AUGCustomDataset(Dataset):
    def __init__(self, image_folder, annotation_file, replacement_p,expansion_rate,syn_dir, transform,datasets,imb_f,use_randaug):
        self.image_folder = image_folder
        self.transform = transform
        self.data = self.load_data(annotation_file)
        self.classes = self.get_classes()
        self.replacement_p = replacement_p
        self.expansion_rate = expansion_rate
        self.syn_dir = syn_dir
        self.datasets = datasets
        self.num_classes,self.normalized_probabilities = self.get_imb_weights(datasets,imb_f)
        self.class_bed = [i for i in range(len(self.num_classes))]
        self.use_randaug = use_randaug
    def load_data(self, annotation_file):
        with open(annotation_file, 'r') as file:
            data = file.readlines()
        data = [line.strip().split() for line in data]
        return data

    def get_imb_weights(self, dataset,imb_f):
        f = open('datasets/imb_'+dataset+'_'+str(imb_f)+'/train.txt','r')
        lines = f.readlines()
        f.close()

        prob = {}
        for line in lines:
            label = line.split('/')[-2]
            if label not in prob:
                prob[label]=1
            else:
                prob[label]+=1
                
        probabilities = [1 / (num + 1) for num in prob.values()]
        total_prob = sum(probabilities)
        normalized_probabilities = [prob / total_prob for prob in probabilities]

        for idx,each_key in enumerate(prob.keys()):
            prob[each_key] = normalized_probabilities[idx]
            
        return list(prob.keys()),list(prob.values())

    def get_classes(self):
        classes = set()
        for _, label in self.data:
            classes.add(label)
        return sorted(list(classes))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        syn_dir = self.syn_dir
        img_name, label = self.data[idx]
        if np.random.uniform() >= self.replacement_p:
            img_path = os.path.join(self.image_folder, img_name)
        else:
            label = np.random.choice(self.class_bed, p=self.normalized_probabilities)
            category = self.num_classes[label]
            files = os.listdir(syn_dir+'/'+category)
            if len(files) == 0:
                img_path = os.path.join(self.image_folder, img_name)
            else:
                img_path = syn_dir+'/'+category+'/' + random.sample(files, 1)[0]
        image = Image.open(img_path).convert('RGB')
        if self.use_randaug:
            r = random.random()
            if r < 0.5:
                image = self.transform[0](image)
            else:
                image = self.transform[1](image)
        else:
            if self.transform is not None:
                image = self.transform(image)
        label = int(label)
        return image, label
      

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data',
                    metavar='DIR',
                    default='/home/zhangzhi/Data/exports/ImageNet2012',
                    help='path to dataset')
parser.add_argument('--datasets',
                    default='aircraft',
                    help='the path to dataset')
parser.add_argument('-a',
                    '--arch',
                    metavar='ARCH',
                    default='resnet50',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (choice: resnet50, vit_b_16, resnet34)')
parser.add_argument('-j',
                    '--workers',
                    default=12,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',
                    default=100,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch_size',
                    default=8*4,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 3200), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.005,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--imb_f',
                    default=0.01,
                    type=float,
                    help='imbalance factor')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=5e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--syn_p',
                    default=0,
                    type=float)
parser.add_argument('--mixup_probability',
                    default=0.1,
                    type=float)
parser.add_argument('-p',
                    '--print-freq',
                    default=200,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--test',
                    dest='test',
                    action='store_true',
                    help='evaluate model on test set')

parser.add_argument('--pretrained',
                    dest='pretrained',
                    action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed',
                    default=42,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('--size',
                    default=384,
                    type=int,
                    help='input size. ')
parser.add_argument('--resize',
                    default=440,
                    type=int,
                    help='input size. ')
parser.add_argument('--num_class',
                    default=100,
                    type=int,
                    help='the number of classes of this datasets. ')
parser.add_argument('--syn_dir',
                    default='',
                    type=str,
                    help='the syn dir. ')
parser.add_argument('--test_file',
                    default='',
                    type=str,
                    help='the test file. ')
parser.add_argument('--use_cutmix',
                    default=False,
                    action='store_true',
                    help='if use cutmix. ')
parser.add_argument('--use_Adam',
                    default=False,
                    action='store_true',
                    help='if use Adam. ')
parser.add_argument('--use_randaug',
                    default=False,
                    action='store_true',
                    help='if use randaug. ')


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(local_rank, args.nprocs, args)

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.lr * epoch / 5
    elif epoch > 180:
        lr = args.lr * 0.0001
    elif epoch > 160:
        lr = args.lr * 0.01
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main_worker(local_rank, nprocs, args):
    best_acc = .0
    dist.init_process_group(backend='nccl')

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        if args.arch =='vit_b_16':
            model = models.__dict__[args.arch](weights='IMAGENET1K_SWAG_E2E_V1')
            model.heads.head = nn.Linear(
            model.heads.head.in_features, args.num_class
        )   
        if args.arch =='resnet50':
            model = models.__dict__[args.arch](weights='IMAGENET1K_V1')
            model.fc = nn.Linear(
            model.fc.in_features, args.num_class
        )
        if args.arch =='resnet34':
            model = models.__dict__[args.arch](weights='IMAGENET1K_V1')
            model.fc = nn.Linear(
            model.fc.in_features, args.num_class
        )
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch =='vit_b_16':
            model = models.__dict__[args.arch](weights=None)
            model.heads.head = nn.Linear(
            model.heads.head.in_features, args.num_class
        )   
        if args.arch =='resnet50':
            model = models.__dict__[args.arch](weights=None)
            model.fc = nn.Linear(
            model.fc.in_features, args.num_class
        )
        if args.arch =='resnet34':
            model = models.__dict__[args.arch](weights=None)
            model.fc = nn.Linear(
            model.fc.in_features, args.num_class
        )
        
    

    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / nprocs)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank])

    cudnn.benchmark = True


    # 定义数据转换
    if args.use_randaug:
        print("use randaug!!")
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(translate_const=int(224 * 0.45),
                         img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        augmentation_randncls = [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
            transforms.ToTensor(),
            normalize,
        ]
        augmentation_sim = [
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        transform = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim)]

        transform_eval = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transform_eval = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    train_dataset = AUGCustomDataset(image_folder='', annotation_file='datasets/imb_'+args.datasets+'_'+str(args.imb_f)+'/train.txt', replacement_p = args.syn_p, expansion_rate = 5,syn_dir=args.syn_dir,transform=transform,datasets = args.datasets,imb_f = args.imb_f,use_randaug=args.use_randaug)
    if args.test_file=='':
        val_dataset = CustomDataset(image_folder='', annotation_file='datasets/'+args.datasets+'/test_imb.txt', transform=transform_eval)
    else:
        val_dataset = CustomDataset(image_folder='', annotation_file=args.test_file, transform=transform_eval)
    if args.use_cutmix:
        train_dataset = CutMix(train_dataset, num_class=args.num_class, prob=args.mixup_probability)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=12,
                                               pin_memory=True,
                                               sampler=train_sampler)
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=12,
                                             pin_memory=True,
                                             sampler=val_sampler)
    


    criterion = LabelSmoothingLoss(
    classes=args.num_class, smoothing=0.1).cuda(local_rank)  # label smoothing to improve performance

    if args.use_Adam:
        optimizer = torch.optim.AdamW(model.parameters(),args.lr,weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader.dataset) // (args.batch_size*nprocs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    warmup_scheduler = pytorch_warmup.LinearWarmup(optimizer, warmup_period=max(int(0.1*total_steps),1))

    
    
    if args.evaluate:
        validate(val_loader, model, criterion, local_rank, args)
        return
    

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch,local_rank,scheduler,warmup_scheduler,
              args)

        # evaluate on validation set
        acc = validate(val_loader, model, criterion, local_rank, args)
        best_acc = max(acc, best_acc)

            
    print("The final acc is: ",best_acc)


def train(train_loader, model, criterion, optimizer, epoch, local_rank,scheduler,warmup_scheduler, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)
        if args.use_cutmix:
            target = torch.argmax(target, dim=1)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        torch.distributed.barrier()
        reduced_loss = reduce_mean(loss, args.nprocs)
        reduced_acc1 = reduce_mean(acc1, args.nprocs)
        reduced_acc5 = reduce_mean(acc5, args.nprocs)

        losses.update(reduced_loss.item(), images.size(0))
        top1.update(reduced_acc1.item(), images.size(0))
        top5.update(reduced_acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            
        # # print(i,optimizer.state_dict()['param_groups'][0]['lr'])        
        with warmup_scheduler.dampening():
            scheduler.step()

def validate(val_loader, model, criterion, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_acc1 = reduce_mean(acc1, args.nprocs)
            reduced_acc5 = reduce_mean(acc5, args.nprocs)

            losses.update(reduced_loss.item(), images.size(0))
            top1.update(reduced_acc1.item(), images.size(0))
            top5.update(reduced_acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                    top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'




def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
    
    
