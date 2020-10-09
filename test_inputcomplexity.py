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
import os
import DCGAN_VAE_pixel as DVAE
import torch.nn.functional as F
import cv2



def reparameterize(mu, logvar, device):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std).to(device)
    return eps.mul(std).add_(mu)

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
        b = x.size(0)
        target = Variable(x.data.view(-1) * 255).long()
        recon = recon.contiguous()
        recon = recon.view(-1,256)
        cross_entropy = F.cross_entropy(recon, target, reduction='none')
        log_p_x_z = -torch.sum(cross_entropy.view(b ,-1), 1)
      
        log_p_z = -torch.sum(z**2/2+np.log(2*np.pi)/2,1)
        z_eps = z - mu
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
    parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    
    parser.add_argument('--state_E', default='./saved_models/fmnist/netE_pixel.pth', help='path to encoder checkpoint')
    parser.add_argument('--state_G', default='./saved_models/fmnist/netG_pixel.pth', help='path to encoder checkpoint')

    parser.add_argument('--ic_type', default='png', help='type of complexity measure, choose between png and jp2')

    opt = parser.parse_args()
    
    cudnn.benchmark = True
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
#    dataset_cifar_test = dset.CIFAR10(root=opt.dataroot, download=True,train = False,
#                                transform=transforms.Compose([
#                                transforms.Resize(opt.imageSize),
#                                transforms.ToTensor()
#                            ]))
#
#    dataset_svhn = dset.SVHN(root = opt.dataroot, download=True,
#                                        transform=transforms.Compose([
#                                        transforms.Resize(opt.imageSize),
#                                        transforms.ToTensor()
#                            ]))
#        
#
#    test_loader_cifar = torch.utils.data.DataLoader(dataset_cifar_test, batch_size=1,
#                                            shuffle=True, num_workers=int(opt.workers))
#    
#    test_loader_svhn = torch.utils.data.DataLoader(dataset_svhn, batch_size=1,
#                                            shuffle=True, num_workers=int(opt.workers))
    
    
    dataset_fmnist = dset.FashionMNIST(root=opt.dataroot, train=False, download=True, transform=transforms.Compose([
                                transforms.Resize(opt.imageSize),
                                transforms.ToTensor()
                            ]))
    test_loader_fmnist = torch.utils.data.DataLoader(dataset_fmnist, batch_size=opt.batch_size,
                                            shuffle=True, num_workers=int(opt.workers))
    
    dataset_mnist = dset.MNIST(root=opt.dataroot, train=False, download=True, transform=transforms.Compose([
                                transforms.Resize(opt.imageSize),
                                transforms.ToTensor()
                            ]))

    test_loader_mnist = torch.utils.data.DataLoader(dataset_mnist, batch_size=opt.batch_size,
                                            shuffle=True, num_workers=int(opt.workers))
    
    
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    nc = int(opt.nc)
    
    print('Building models...')
    netG = DVAE.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu)
    state_G = torch.load(opt.state_G, map_location = device)
    netG.load_state_dict(state_G)
    
    netE = DVAE.Encoder(opt.imageSize, nz, nc, ngf, ngpu)
    state_E = torch.load(opt.state_E, map_location = device)
    netE.load_state_dict(state_E)

    
    netG.to(device)
    netG.eval()
    netE.to(device)
    netE.eval()
    
    loss_fn = nn.CrossEntropyLoss(reduction = 'none')
    
    print('Building complete...')
    '''
    First run through the VAE and record the ELBOs of each image in cifar and svhn
    '''
    NLL_test_indist = []
    Complexity_test_indist = []
    difference_indist = []

    for i, (x, _) in enumerate(test_loader_fmnist):
        x = x.expand(opt.repeat,-1,-1,-1).contiguous()
        
        weights_agg  = []
        with torch.no_grad():
            for batch_number in range(5):
            
                x = x.to(device)
                b = x.size(0)
        
                [z,mu,logvar] = netE(x)
                recon = netG(z)
                mu = mu.view(mu.size(0),mu.size(1))
                logvar = logvar.view(logvar.size(0), logvar.size(1))
                z = z.view(z.size(0),z.size(1))
                weights = store_NLL(x, recon, mu, logvar, z)
                
                weights_agg.append(weights)
            
            weights_agg = torch.stack(weights_agg).view(-1) 
            
            NLL_loss = compute_NLL(weights_agg)
            NLL_test_indist.append(NLL_loss.detach().cpu().numpy())
            
            img = x[0].permute(1,2,0)
            img = img.detach().cpu().numpy()
            img *= 255
            img = img.astype(np.uint8)
            if opt.ic_type == 'jp2':
                img_encoded=cv2.imencode('.jp2',img)
            elif opt.ic_type == 'png':
                img_encoded=cv2.imencode('.png',img,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
            else:
                raise NotImplementedError("choose ic type between jp2 and png")
            L=len(img_encoded[1])*8
            Complexity_test_indist.append(L)
            difference_indist.append(NLL_loss.detach().cpu().numpy() - L)
            print('CIFAR VAE: image {} NLL loss {}'.format(i, NLL_loss.item()))
            
        if i >= 499:
            break
    
    difference_indist = np.asarray(difference_indist)
    np.save('./array/complexity/difference_indist.npy', difference_indist)


###########################################################################################

    NLL_test_ood = []
    Complexity_test_ood = []
    difference_ood = []
    for i, (x, _) in enumerate(test_loader_mnist):

        x = x.expand(opt.repeat,-1,-1,-1).contiguous()
        
        weights_agg  = []
       
        with torch.no_grad():
            for batch_number in range(5):
            
                x = x.to(device)
                b = x.size(0)
        
                [z,mu,logvar] = netE(x)
                recon = netG(z)
                mu = mu.view(mu.size(0),mu.size(1))
                logvar = logvar.view(logvar.size(0), logvar.size(1))
                z = z.view(z.size(0),z.size(1))
                weights = store_NLL(x, recon, mu, logvar, z)
               
                weights_agg.append(weights)
            
            weights_agg = torch.stack(weights_agg).view(-1) 
         
            NLL_loss = compute_NLL(weights_agg)
            NLL_test_ood.append( NLL_loss.detach().cpu().numpy())
            
            img = x[0].permute(1,2,0)
            img = img.detach().cpu().numpy()
            img *= 255
            img = img.astype(np.uint8)
            
            if opt.ic_type == 'jp2':
                img_encoded=cv2.imencode('.jp2',img)
            elif opt.ic_type == 'png':
                img_encoded=cv2.imencode('.png',img,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
            else:
                raise NotImplementedError("choose ic type between jp2 and png")
            L=len(img_encoded[1])*8
            Complexity_test_ood.append(L)
            difference_ood.append(NLL_loss.detach().cpu().numpy() - L)

            print('SVHN VAE: image {} NLL loss {}'.format(i, NLL_loss.item()))
        if i >= 499:
            break
    difference_ood = np.asarray(difference_ood)
        
    np.save('./array/complexity/difference_ood.npy', difference_ood)
   
  


      




    
        
    