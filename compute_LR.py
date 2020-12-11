import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import DCGAN_VAE_pixel as DVAE
import torch.nn.functional as F
import copy
from torchvision.datasets import ImageFolder


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return (img,index)


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
    parser.add_argument('--num_iter', type=int, default=100, help='number of iters to optimize')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')

    parser.add_argument('--repeat', type=int, default=200, help='repeat for comute IWAE bounds')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    
    parser.add_argument('--state_E', default='./saved_models/fmnist/netE_pixel.pth', help='path to encoder checkpoint')
    parser.add_argument('--state_G', default='./saved_models/fmnist/netG_pixel.pth', help='path to decoder checkpoint')

    opt = parser.parse_args()
    
    cudnn.benchmark = True
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    
#    dataset_cifar_test = dset.CIFAR10(root=opt.dataroot, download=True,train = False,
#                                transform=transforms.Compose([
#                                transforms.Resize((opt.imageSize)),
#                                transforms.ToTensor(),
#                            ]))
#
#    dataset_svhn = dset.SVHN(root=opt.dataroot, download=True,
#                                        transform=transforms.Compose([
#                                        transforms.Resize((opt.imageSize)),
#                                        transforms.ToTensor(),
#                            ]))
#    
#    
#
#    test_loader_cifar = torch.utils.data.DataLoader(dataset_cifar_test, batch_size=1,
#                                            shuffle=True, num_workers=int(opt.workers))
#    
#    test_loader_svhn = torch.utils.data.DataLoader(dataset_svhn, batch_size=1,
#                                            shuffle = True, num_workers=int(opt.workers))
    

    
    dataset_fmnist = dset.FashionMNIST(root=opt.dataroot, train=False, download=True, transform=transforms.Compose([
                                transforms.Resize((opt.imageSize)),
                                transforms.ToTensor(),
                            ]))
    test_loader_fmnist = torch.utils.data.DataLoader(dataset_fmnist, batch_size=opt.batch_size,
                                            shuffle=True, num_workers=int(opt.workers))
    
    dataset_mnist = dset.MNIST(root=opt.dataroot, train=False, download=True, transform=transforms.Compose([
                                transforms.Resize((opt.imageSize,opt.imageSize)),
                                transforms.ToTensor(),
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

    NLL_regret_indist = []
    NLL_indist = []

    for i, (xi, _) in enumerate(test_loader_fmnist):
       
        x = xi.expand(opt.repeat,-1,-1,-1).contiguous()
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
            
            NLL_loss_before = compute_NLL(weights_agg) 
            NLL_indist = np.append(NLL_indist, NLL_loss_before.detach().cpu().numpy())
            
        xi = xi.to(device)
        b = xi.size(0)
        netE_copy = copy.deepcopy(netE)
        netE_copy.eval()
        optimizer = optim.Adam(netE_copy.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=5e-5)
        target = Variable(xi.data.view(-1) * 255).long()
        for it in range(opt.num_iter):
            
            [z,mu,logvar] = netE_copy(xi)
            recon = netG(z)
            
            recon = recon.contiguous()
            recon = recon.view(-1,256)
            recl = loss_fn(recon, target)
            
            recl = torch.sum(recl) / b
            kld = KL_div(mu,logvar)
            
            loss =  recl + kld.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        weights_agg  = []
        with torch.no_grad():
            xi = xi.expand(opt.repeat,-1,-1,-1).contiguous()
            target = Variable(xi.data.view(-1) * 255).long()
            for batch_number in range(5):
                [z,mu,logvar] = netE_copy(xi)
                recon = netG(z)
                recon = recon.contiguous()
                mu = mu.view(mu.size(0),mu.size(1))
                logvar = logvar.view(logvar.size(0), logvar.size(1))
                z = z.view(z.size(0),z.size(1))
                weights = store_NLL(xi, recon, mu, logvar, z)
                
                weights_agg.append(weights)
         
            
            weights_agg = torch.stack(weights_agg).view(-1) 
            NLL_loss_after = compute_NLL(weights_agg) 
            print('In-dist image {} OPT: {} VAE: {} diff:{}'.format(i, NLL_loss_after.item(), NLL_loss_before.item(), NLL_loss_before.item()  - NLL_loss_after.item()))
            regret = NLL_loss_before  - NLL_loss_after
            NLL_regret_indist = np.append(NLL_regret_indist, regret.detach().cpu().numpy())
        if i >= 499: #test for 500 samples
            break
                
           
    np.save('./array/indist_nll.npy', NLL_indist)
    np.save('./array/indist_regret.npy', NLL_regret_indist)
    
    
    
###################################################################################################    
    NLL_regret_ood = []
    NLL_ood = []
    


    for i, (xi, _) in enumerate(test_loader_mnist):
        
        #noise
        #xi = np.random.randint(256, size=xi.size())/255
        #xi = torch.from_numpy(xi).float()
        
        #constant
#        level = np.random.randint(256, size=(3,))
#        temp = torch.ones(xi.size())/255
#        for k in range(opt.nc):
#            temp[:,k,:,:] = level[k]*temp[:,k,:,:]
#        xi = temp
        
        x = xi.expand(opt.repeat,-1,-1,-1).contiguous()
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
            
            NLL_loss_before = compute_NLL(weights_agg) 
            NLL_ood = np.append(NLL_ood, NLL_loss_before.detach().cpu().numpy())

        xi = xi.to(device)
        b = xi.size(0)
        netE_copy = copy.deepcopy(netE)
        netE_copy.eval()
        optimizer = optim.Adam(netE_copy.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=5e-5)

        target = Variable(xi.data.view(-1) * 255).long()
        for it in range(opt.num_iter):
            
            [z,mu,logvar] = netE_copy(xi)
            recon = netG(z)
            
            recon = recon.contiguous()
            recon = recon.view(-1,256)
            recl = loss_fn(recon, target)
            
            recl = torch.sum(recl) / b
            kld = KL_div(mu,logvar)
            
            loss =  recl + kld.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        weights_agg  = []
        with torch.no_grad():
            xi = xi.expand(opt.repeat,-1,-1,-1).contiguous()
            target = Variable(xi.data.view(-1) * 255).long()
            for batch_number in range(5):
                [z,mu,logvar] = netE_copy(xi)
                recon = netG(z)
                recon = recon.contiguous()
                mu = mu.view(mu.size(0),mu.size(1))
                logvar = logvar.view(logvar.size(0), logvar.size(1))
                z = z.view(z.size(0),z.size(1))
                weights = store_NLL(xi, recon, mu, logvar, z)
                
                weights_agg.append(weights)
         
            
            weights_agg = torch.stack(weights_agg).view(-1) 
            NLL_loss_after = compute_NLL(weights_agg) 
            print('OOD image {} OPT: {} VAE: {} diff:{}'.format(i, NLL_loss_after.item(), NLL_loss_before.item(), NLL_loss_before.item()  - NLL_loss_after.item()))
            regret = NLL_loss_before  - NLL_loss_after
            NLL_regret_ood = np.append(NLL_regret_ood, regret.detach().cpu().numpy())
        if i >= 499: #test for 500 samples
            break
                
           

    np.save('./array/ood_regret.npy', NLL_regret_ood)
    np.save('./array/ood_nll.npy', NLL_ood)

