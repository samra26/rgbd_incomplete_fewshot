import torch
from torch.nn import functional as F
from RGBDincomplete import build_model
import numpy as np
import os
import cv2
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
writer = SummaryWriter('log/run' + time.strftime("%d-%m"))
import torch.nn as nn
import argparse
import os.path as osp
import os
size_coarse = (10, 10)



class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        self.model_type = config.model_type
        self.save_folder_rgb = config.save_folder_rgb
        self.save_folder_depth = config.save_folder_depth
        self.save_folder = config.save_folder
        #self.build_model()
        self.net = build_model(self.config.network, self.config.arch,self.config.embed_dim)
        self.net_d = build_model(self.config.network, self.config.arch,self.config.embed_dim)
        #self.net.eval()
        if config.mode == 'test':
            print('Loading pre-trained model for testing from %s...' % self.config.model)
            self.net.load_state_dict(torch.load(self.config.model, map_location=torch.device('cpu')))
        if config.mode == 'train':
            if self.config.load == '':
                print("Loading pre-trained imagenet weights for fine tuning")
                self.net.RGBDInModule.load_pretrained_model(self.config.pretrained_model
                                                        if isinstance(self.config.pretrained_model, str)
                                                        else self.config.pretrained_model[self.config.network])
                # load pretrained backbone
                self.net_d.RGBDInModule.load_pretrained_model(self.config.pretrained_model
                                                        if isinstance(self.config.pretrained_model, str)
                                                        else self.config.pretrained_model[self.config.network])
            else:
                print('Loading pretrained model to resume training')
                self.net.load_state_dict(torch.load(self.config.load))  # load pretrained model
                self.net_d.load_state_dict(torch.load(self.config.load)) 
        
        if self.config.cuda:
            self.net = self.net.cuda()
            self.net_d = self.net_d.cuda()

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        #self.print_network(self.net, 'Incomplete modality RGBD SOD Structure')

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params_t = 0
        num_params=0
        for p in model.parameters():
            if p.requires_grad:
                num_params_t += p.numel()
            else:
                num_params += p.numel()
        print(name)
        print(model)
        print("The number of trainable parameters: {}".format(num_params_t))
        print("The number of parameters: {}".format(num_params))

    # build the network
    '''def build_model(self):
        self.net = build_model(self.config.network, self.config.arch)

        if self.config.cuda:
            self.net = self.net.cuda()

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        self.print_network(self.net, 'JL-DCF Structure')'''

    def test(self):
        print('Testing...')
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size, depth = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size']), \
                                           data_batch['depth']
            with torch.no_grad():
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    images = images.to(device)
                    depth = depth.to(device)

                #input = torch.cat((images, depth), dim=0)
                preds = self.net(images)
                #print(preds.shape)
                preds = F.interpolate(preds, tuple(im_size), mode='bilinear', align_corners=True)
                pred = np.squeeze(torch.sigmoid(preds)).cpu().data.numpy()

                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                multi_fuse = 255 * pred
                filename = os.path.join(self.config.test_folder, name[:-4] + '_rgbonly.png')
                cv2.imwrite(filename, multi_fuse)
        time_e = time.time()
        print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        print('Test Done!')
    
  
    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        iter_numd = len(self.train_depth_loader.dataset) // self.config.batch_size
        
        loss_vals=  []
        loss_valsd=  []
        
        for epoch in range(self.config.epoch):
            r_sal_loss = 0
            r_sal_loss_item=0
            dr_sal_loss = 0
            dr_sal_loss_item=0
            count=0
            for i, data_batch in enumerate(self.train_loader):
                count=count+1
                sal_image, sal_label= data_batch[0], data_batch[1]
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    sal_image,  sal_label= sal_image.to(device),sal_label.to(device)
                self.optimizer.zero_grad()
                sal_rgb_only = self.net(sal_image)
                sal_rgb_only_loss =  F.binary_cross_entropy_with_logits(sal_rgb_only, sal_label, reduction='sum')
                sal_rgb_only_loss = sal_rgb_only_loss/ (self.iter_size * self.config.batch_size)
                r_sal_loss += sal_rgb_only_loss.data
                r_sal_loss_item+=sal_rgb_only_loss.item() * sal_image.size(0)
                sal_rgb_only_loss.backward()
                self.optimizer.step()
                if (i + 1) % (self.show_every // self.config.batch_size) == 0:
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  sal_rgb_only_loss : %0.4f' % (
                            epoch, self.config.epoch, i + 1, iter_num, r_sal_loss ))
                    # print('Learning rate: ' + str(self.lr))
                    writer.add_scalar('training loss', r_sal_loss / (self.show_every / self.iter_size),
                                        epoch * len(self.train_loader.dataset) + i)
                   
                    fsal = sal_rgb_only[0].clone()
                    fsal = fsal.sigmoid().data.cpu().numpy().squeeze()
                    fsal = (fsal - fsal.min()) / (fsal.max() - fsal.min() + 1e-8)
                    writer.add_image('sal_rgb_final', torch.tensor(fsal), i, dataformats='HW')
                    grid_image = make_grid(sal_label[0].clone().cpu().data, 1, normalize=True)
                if count == 4:
                    for id, data_batch_d in enumerate(self.train_depth_loader):
                        rgb_image, depth_image , label= data_batch_d[0], data_batch_d[1] , data_batch_d[2]
                        if (rgb_image.size(2) != label.size(2)) or (rgb_image.size(3) != label.size(3)):
                            print('IMAGE ERROR, PASSING```')
                            continue
                        if self.config.cuda:
                            device = torch.device(self.config.device_id)
                            rgb_image, depth_image, label= rgb_image.to(device),depth_image.to(device), label.to(device)
                        self.optimizer.zero_grad()
                        sal_depth_only = self.net_d(sal_depth)
                        sal_depth_only_loss =  F.binary_cross_entropy_with_logits(sal_depth_only, label, reduction='sum')
                        sal_depth_only_loss = sal_depth_only_loss/ (self.iter_size * self.config.batch_size/4)
                        dr_sal_loss += sal_depth_only_loss.data
                        dr_sal_loss_item+=sal_depth_only_loss.item() * depth_image.size(0)
                        sal_depth_only_loss.backward()
                        self.optimizer.step()
                        if (i + 1) % (self.show_every // self.config.batch_size) == 0:
                            print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  sal_depth_only_loss : %0.4f' % (
                                epoch, self.config.epoch, i + 1, iter_numd, dr_sal_loss ))
                            # print('Learning rate: ' + str(self.lr))
                            writer.add_scalar('depth training loss', dr_sal_loss / (self.show_every / self.iter_size),
                                            epoch * len(self.train_depth_loader.dataset) + i)
                   
                            fsald = sal_depth_only[0].clone()
                            fsald = fsald.sigmoid().data.cpu().numpy().squeeze()
                            fsald = (fsald - fsald.min()) / (fsald.max() - fsald.min() + 1e-8)
                            writer.add_image('sal_depth_final', torch.tensor(fsal), i, dataformats='HW')
                            grid_image = make_grid(label[0].clone().cpu().data, 1, normalize=True)


            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/epoch_%d.pth' % (self.save_folder_rgb, epoch + 1))
                torch.save(self.net_d.state_dict(), '%s/epoch_%d.pth' % (self.save_folder_depth, epoch + 1))
            train_loss=r_sal_loss_item/len(self.train_loader.dataset)
            loss_vals.append(train_loss)
            
            print('Epoch:[%2d/%2d] | Train Loss : %.3f' % (epoch, self.config.epoch,train_loss))
            train_lossd=dr_sal_loss_item/len(self.train_depth_loader.dataset)
            loss_valsd.append(train_lossd)
            
            print('Epoch:[%2d/%2d] | DEpth Train Loss : %.3f' % (epoch, self.config.epoch,train_lossd))
            
        # save model
       
        torch.save(self.net.state_dict(), '%s/final.pth' % self.save_folder_rgb)
        torch.save(self.net_d.state_dict(), '%s/final.pth' % self.save_folder_depth)
       
        

