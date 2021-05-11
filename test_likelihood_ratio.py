# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 22:38:49 2020

@author: xiaoz
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import DCGAN_VAE_pixel as DVAE
import torch.nn.functional as F
from torch.utils.data import Dataset


def KL_div(mu,logvar,reduction = 'none'):
    mu = mu.view(mu.size(0),mu.size(1))
    logvar = logvar.view(logvar.size(0), logvar.size(1))
    
    if reduction == 'sum':
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 
    else:
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1) 
        return KL


def store_NLL(x, recon, mu, logvar, z):
    with torch.no_grad():
        sigma = torch.exp(0.5*logvar)
        b = x.size(0)
        target = Variable(x.data.view(-1) * 255).long()
        recon = recon.contiguous()
        recon = recon.view(-1,256)
        cross_entropy = F.cross_entropy(recon, target, reduction='none')
        log_p_x_z = -torch.sum(cross_entropy.view(b ,-1), 1)
      
        log_p_z = -torch.sum(z**2/2+np.log(2*np.pi)/2,1)
        z_eps = (z - mu)/sigma
        z_eps = z_eps.view(opt.repeat,-1)
        log_q_z_x = -torch.sum(z_eps**2/2 + np.log(2*np.pi)/2 + logvar/2, 1)
        
        weights = log_p_x_z+log_p_z-log_q_z_x
        
    return weights

def compute_NLL(weights):
    
    with torch.no_grad():
        NLL_loss = -(torch.log(torch.mean(torch.exp(weights - weights.max())))+weights.max()) 
        
    return NLL_loss

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='./data', help='path to dataset')

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=1, help='input image channels')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')

    parser.add_argument('--repeat', type=int, default=200)
    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    
    parser.add_argument('--state_E', default='./saved_models/fmnist/netE_pixel.pth', help='path to encoder checkpoint')
    parser.add_argument('--state_G', default='./saved_models/fmnist/netG_pixel.pth', help='path to encoder checkpoint')

    parser.add_argument('--state_E_bg', default='./saved_models/fmnist/netE_pixel_bg.pth', help='path to encoder checkpoint')
    parser.add_argument('--state_G_bg', default='./saved_models/fmnist/netG_pixel_bg.pth', help='path to encoder checkpoint')

    opt = parser.parse_args()
    
    cudnn.benchmark = True
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    dataset_fmnist = dset.FashionMNIST(root=opt.dataroot, train=False, download=True, transform=transforms.Compose([
                                transforms.Resize(opt.imageSize),
                                transforms.ToTensor()
                            ]))
    dataloader_fmnist = torch.utils.data.DataLoader(dataset_fmnist, batch_size=opt.batch_size,
                                            shuffle=True, num_workers=int(opt.workers))

    dataset_mnist = dset.MNIST(root=opt.dataroot, train=False, download=True, transform=transforms.Compose([
                                transforms.Resize(opt.imageSize),
                                transforms.ToTensor()
                            ]))

    dataloader_mnist = torch.utils.data.DataLoader(dataset_mnist, batch_size=opt.batch_size,
                                            shuffle=True, num_workers=int(opt.workers))
    
#    
#    dataset_cifar_test = dset.CIFAR10(root=opt.dataroot, download=True,train = False,
#                                transform=transforms.Compose([
#                                transforms.Resize(opt.imageSize),
#                                transforms.ToTensor()
#                            ]))
#
#    dataset_svhn = dset.SVHN(root=opt.dataroot, download=True,
#                                        transform=transforms.Compose([
#                                        transforms.Resize(opt.imageSize),
#                                        transforms.ToTensor()
#                            ]))
#    
#
#
#    dataloader_cifar = torch.utils.data.DataLoader(dataset_cifar_test, batch_size=1,
#                                            shuffle=True, num_workers=int(opt.workers))
#    
#    dataloader_svhn = torch.utils.data.DataLoader(dataset_svhn, batch_size=1,
#                                            shuffle=True, num_workers=int(opt.workers))
    
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    nc = int(opt.nc)
    
    print('Building models...')
    netG = DVAE.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu)
    state_G = torch.load(opt.state_G, map_location = device)
    netG.load_state_dict(state_G)
    netG_bg = DVAE.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu)
    state_G_bg = torch.load(opt.state_G_bg, map_location = device)
    netG_bg.load_state_dict(state_G_bg)
    
    netE = DVAE.Encoder(opt.imageSize, nz, nc, ngf, ngpu)
    state_E = torch.load(opt.state_E, map_location = device)
    netE.load_state_dict(state_E)
    netE_bg = DVAE.Encoder(opt.imageSize, nz, nc, ngf, ngpu)
    state_E_bg = torch.load(opt.state_E_bg, map_location = device)
    netE_bg.load_state_dict(state_E_bg)

    
    netG.to(device)
    netG.eval()
    netE.to(device)
    netE.eval()
    
    netG_bg.to(device)
    netG_bg.eval()
    netE_bg.to(device)
    netE_bg.eval()
    loss_fn = nn.CrossEntropyLoss(reduction = 'none')
    
    print('Building complete...')
    '''
    First run through the VAE and record the ELBOs of each image in fmnist and mnist
    '''
    NLL_test_indist = []
    NLL_test_indist_bg = []

    for i, (x, _) in enumerate(dataloader_fmnist):
        x = x.expand(opt.repeat,-1,-1,-1).contiguous()
        weights_agg  = []
        weights_agg_bg = []
        with torch.no_grad():
            for batch_number in range(10):
            
                x = x.to(device)
                b = x.size(0)
        
                [z,mu,logvar] = netE(x)
                recon = netG(z)
                mu = mu.view(mu.size(0),mu.size(1))
                logvar = logvar.view(logvar.size(0), logvar.size(1))
                z = z.view(z.size(0),z.size(1))
                weights = store_NLL(x, recon, mu, logvar, z)
                weights_agg.append(weights)
                
                [z_bg,mu_bg,logvar_bg] = netE_bg(x)
                recon_bg = netG_bg(z_bg)
                mu_bg = mu_bg.view(mu_bg.size(0),mu_bg.size(1))
                logvar_bg = logvar_bg.view(logvar_bg.size(0), logvar_bg.size(1))
                z_bg = z_bg.view(z_bg.size(0),z_bg.size(1))
                weights_bg = store_NLL(x, recon_bg, mu_bg, logvar_bg, z_bg)
               
                weights_agg_bg.append(weights_bg)
            
            weights_agg = torch.stack(weights_agg).view(-1) 
            weights_agg_bg = torch.stack(weights_agg_bg).view(-1)
           
            NLL_loss = compute_NLL(weights_agg) 
            NLL_loss_bg =  compute_NLL(weights_agg_bg) 
            NLL_test_indist.append(NLL_loss.detach().cpu().numpy())
            NLL_test_indist_bg.append(NLL_loss_bg.detach().cpu().numpy())
            diff = -NLL_loss.item() + NLL_loss_bg.item()
            print('Indist: image {} NLL {}, NLL BG {}, diff {}'.format(i, NLL_loss.item(),NLL_loss_bg.item(), diff))
            
        if i >= 499:
            break
    NLL_test_indist = np.asarray(NLL_test_indist)
    NLL_test_indist_bg = np.asarray(NLL_test_indist_bg)
    metric_indist = -NLL_test_indist + NLL_test_indist_bg
    np.save('./array/like_ratio/metric_indist.npy', metric_indist)
##    
    
    NLL_test_ood = []
    NLL_test_ood_bg = []

    for i, (x, _) in enumerate(dataloader_mnist):
        
        x = x.expand(opt.repeat,-1,-1,-1).contiguous()
        
        
        weights_agg  = []
        weights_agg_bg = []
        with torch.no_grad():
            for batch_number in range(10):
                x = x.to(device)
                b = x.size(0)
                
                [z,mu,logvar] = netE(x)
                recon = netG(z)
                mu = mu.view(mu.size(0),mu.size(1))
                logvar = logvar.view(logvar.size(0), logvar.size(1))
                z = z.view(z.size(0),z.size(1))
                weights = store_NLL(x, recon, mu, logvar, z)
                weights_agg.append(weights)
                
                [z_bg,mu_bg,logvar_bg] = netE_bg(x)
                recon_bg = netG_bg(z_bg)
                mu_bg = mu_bg.view(mu_bg.size(0),mu_bg.size(1))
                logvar_bg = logvar_bg.view(logvar_bg.size(0), logvar_bg.size(1))
                z_bg = z_bg.view(z_bg.size(0),z_bg.size(1))
                weights_bg = store_NLL(x, recon_bg, mu_bg, logvar_bg, z_bg)
                weights_agg_bg.append(weights_bg)
            
            weights_agg = torch.stack(weights_agg).view(-1) 
            weights_agg_bg = torch.stack(weights_agg_bg).view(-1)
           
            NLL_loss = compute_NLL(weights_agg) 
            NLL_loss_bg =  compute_NLL(weights_agg_bg) 
            
            NLL_test_ood.append(NLL_loss.detach().cpu().numpy())
            NLL_test_ood_bg.append(NLL_loss_bg.detach().cpu().numpy())
            diff = -NLL_loss.item() + NLL_loss_bg.item()
            print('OOD: image {} NLL {}, NLL BG {}, diff: {}'.format(i, NLL_loss.item(),NLL_loss_bg.item(), diff))
        if i >= 499:
            break
    NLL_test_ood = np.asarray(NLL_test_ood)
    NLL_test_ood_bg = np.asarray(NLL_test_ood_bg)
    metric_ood = -NLL_test_ood + NLL_test_ood_bg
    np.save('./array/like_ratio/metric_ood.npy', metric_ood)
            
    

      




    
        
    
