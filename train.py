import os
import time
import torch
import logging
import argparse
import torchvision
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
import cv2
import torchvision.transforms as transforms
from requests.utils import urlparse
import wget

import model
import model.resnet50
import model.vgg19
from utils.utils import load_config, setup_seed, plot_roi, plot_mask_cat
from utils.visualize import Visualizer
from utils.transform import UnNormalizer
from PIL import Image

def main():
    model_options = ['resnet50', 'vgg19']
    dataset_options = ['birds', 'cars', 'airs']

    parser = argparse.ArgumentParser(description='AP-CNN')
    parser.add_argument('--dataset', '-d', default='birds',
                        choices=dataset_options)
    parser.add_argument('--model', '-a', default='resnet50',
                        choices=model_options)
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument("--gpu", type=int, default=0,
                        help='gpu index (default: 0)')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='plot attention masks and ROIs')

    args = parser.parse_args()

    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    ### prepare configurations
    config_file = "configs/config_{}.yaml".format(args.dataset)
    config = load_config(config_file)
    # data config
    train_dir = config['train_dir']
    test_dir = config['test_dir']
    num_class = config['num_class']
    # model config
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    momentum = config['momentum']
    weight_decay = float(config['weight_decay'])
    num_epoch = config['num_epoch']
    resize_size = config['resize_size']
    crop_size = config['crop_size']
    # visualizer config
    vis_host = config['vis_host']
    vis_port = config['vis_port']

    ### setup exp_dir
    exp_name = "AP-CNN_{}_{}".format(args.model, args.dataset)
    time_str = time.strftime("%m-%d-%H-%M", time.localtime())
    exp_dir = os.path.join("./logs", exp_name + '_' + time_str)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    # generate log files
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.basicConfig(filename=os.path.join(exp_dir, 'train.log'), level=logging.INFO, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)-4s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info('==>exp dir:%s' % exp_dir)
    logging.info("OPENING " + exp_dir + '/results_train.csv')
    logging.info("OPENING " + exp_dir + '/results_test.csv')

    results_train_file = open(exp_dir + '/results_train.csv', 'w')
    results_train_file.write('epoch, train_acc, train_loss\n')
    results_train_file.flush()
    results_test_file = open(exp_dir + '/results_test.csv', 'w')
    results_test_file.write('epoch, test_acc, test_loss\n')
    results_test_file.flush()

    # set up Visualizer
    vis = Visualizer(env=exp_name, port=vis_port, server=vis_host)

    ### preparing data
    logging.info('==> Preparing data..')

    transform_train = transforms.Compose([
        transforms.Resize((resize_size, resize_size), Image.BILINEAR),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((resize_size, resize_size), Image.BILINEAR),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    unorm = UnNormalizer([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    logging.info('==> Successfully Preparing data..')

    ### building model
    logging.info('==> Building model..')
    # load pretrained backbone on ImageNet
    if args.model == "resnet50":
    	url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    elif args.model == "vgg19":
    	url = 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'
    model_dir = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch/models'))
    filename = os.path.basename(urlparse(url).path)
    pretrained_path = os.path.join(model_dir, filename)
    if not os.path.exists(pretrained_path):
        wget.download(url, pretrained_path)
    net = getattr(getattr(model, args.model), args.model)(num_class)
    if pretrained_path:
        logging.info('load pretrained backbone')
        net_dict = net.state_dict()
        pretrained_dict = torch.load(pretrained_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
        net_dict.update(pretrained_dict)
        net.load_state_dict(net_dict)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()
        cudnn.benchmark = True
    logging.info('==> Successfully Building model..')

    ### training scripts
    def train(epoch):
        logging.info('Epoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        idx = 0
        flag = 0
        count = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            idx = batch_idx
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            loss_ret, acc_ret, mask_cat, roi_list = net(inputs, targets)
            loss = loss_ret['loss']
            loss.backward()
            optimizer.step()
            train_loss += loss.data
            total += targets.size(0)
            correct += acc_ret['acc']
            if args.visualize and flag % 100 == 0:
                plot_mask_cat(inputs, mask_cat, unorm, vis, 'train')
                plot_roi(inputs, roi_list, unorm, vis, 'train')
                flag += 1
        train_acc = 100. * correct / total
        train_loss = train_loss / (idx + 1)
        logging.info('Iteration %d, train_acc = %.4f, train_loss = %.4f' % (epoch, train_acc, train_loss))
        results_train_file.write('%d, %.4f,%.4f\n' % (epoch, train_acc, train_loss))
        results_train_file.flush()
        return train_acc, train_loss

    ### test scripts
    def test(epoch):
        with torch.no_grad():
            net.eval()
            test_loss = 0
            correct = 0
            total = 0
            idx = 0
            flag = 0
            count = 0
            for batch_idx, (inputs, targets) in enumerate(testloader):
                idx = batch_idx
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = Variable(inputs), Variable(targets)
                loss_ret, acc_ret, mask_cat, roi_list = net(inputs, targets)
                loss = loss_ret['loss']

                test_loss += loss.data
                total += targets.size(0)
                correct += acc_ret['acc']
                if args.visualize and flag % 100 == 0:
                    plot_mask_cat(inputs, mask_cat, unorm, vis, 'test')
                    plot_roi(inputs, roi_list, unorm, vis, 'test')
                    flag += 1

        test_acc = 100. * correct / total
        test_loss = test_loss / (idx + 1)
        logging.info('Iteration %d, test_acc = %.4f, test_loss = %.4f' % (epoch, test_acc, test_loss))
        results_test_file.write('%d, %.4f,%.4f\n' % (epoch, test_acc, test_loss))
        results_test_file.flush()
        return test_acc, test_loss

    if args.dataset == 'birds':
        optimizer = optim.SGD([
                                {'params': nn.Sequential(*list(net.children())[7:]).parameters(),   'lr': learning_rate},
                                {'params': nn.Sequential(*list(net.children())[:7]).parameters(),   'lr': learning_rate/10}
                                
                            ], 
                            momentum=momentum, weight_decay=weight_decay)

        def cosine_anneal_schedule(t):
            cos_inner = np.pi * (t % (num_epoch))
            cos_inner /= (num_epoch)
            cos_out = np.cos(cos_inner) + 1
            return float( learning_rate / 2 * cos_out)
    
        max_test_acc = 0.
        for epoch in range(0, num_epoch):
            optimizer.param_groups[0]['lr'] = cosine_anneal_schedule(epoch)
            optimizer.param_groups[1]['lr'] = cosine_anneal_schedule(epoch) / 10
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
            train(epoch)
            test_acc, _ = test(epoch)
            if test_acc > max_test_acc:
                max_test_acc = test_acc
                torch.save(net.state_dict(), os.path.join(exp_dir, 'model_best.pth'))
            print('max_test_acc=',max_test_acc)

    else:
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)
        
        max_test_acc = 0.
        for epoch in range(0, num_epoch):
            scheduler.step(epoch)
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
            train(epoch)
            test_acc, _ = test(epoch)
            if test_acc > max_test_acc:
                max_test_acc = test_acc
                torch.save(net.state_dict(), os.path.join(exp_dir, 'model_best.pth'))
            print('max_test_acc=',max_test_acc)

    torch.save(net.state_dict(), os.path.join(exp_dir, 'model_final.pth'))  

if __name__=="__main__":
    main()