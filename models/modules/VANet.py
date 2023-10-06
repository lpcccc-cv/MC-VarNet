from .module_util import *
import torch
from data.IXI_dataset import complex_to_real, real_to_complex


class Encoding_Block(torch.nn.Module):
    def __init__(self, c_in, n_feat):
        super(Encoding_Block, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=c_in, out_channels=n_feat, kernel_size=3, padding=3 // 2)
        self.conv2 = torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=3 // 2)
        self.conv3 = torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=3 // 2)
        self.conv4 = torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, stride = 2, padding=3 // 2)
        self.act =  torch.nn.PReLU()

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, input):

        f_e = self.conv3(self.act(self.conv2(self.act(self.conv1(input)))))
        down = self.conv4(f_e)
        return f_e, down



class Decoding_Block(torch.nn.Module):
    def __init__(self,c_in, n_feat):
        super(Decoding_Block, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=n_feat*2, out_channels=n_feat, kernel_size=3, padding=3 // 2)
        self.conv2 = torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=3 // 2)
        self.conv3 = torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=3 // 2)
        self.conv4 = torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=3 // 2)
        self.up = torch.nn.ConvTranspose2d(c_in, n_feat, kernel_size=3, stride=2,padding=3 // 2)

        self.act =  torch.nn.PReLU()

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, input, map):

        up = self.up(input, output_size = [input.shape[0],input.shape[1],map.shape[2],map.shape[3]])
        out1 = self.act(self.conv1(torch.cat((up, map), 1)))
        out = self.conv4(self.act(self.conv3(self.act(self.conv2(out1)))))

        return out

class Feature_Decoding_End(torch.nn.Module):
    def __init__(self, n_feat, c_out):
        super(Feature_Decoding_End, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=3 // 2)
        self.conv2 = torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=3 // 2)
        self.conv3 = torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=3 // 2)
        self.conv_out = torch.nn.Conv2d(in_channels=n_feat, out_channels=c_out, kernel_size=3, padding=3 // 2)
        
        self.act =  torch.nn.PReLU()
        
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, input):
        out = self.conv3(self.act(self.conv2(self.act(self.conv1(input)))))
        out = self.conv_out(out)
        return out

class Unet_denoise(torch.nn.Module):
    def __init__(self, cin, n_feat):
        super(Unet_denoise, self).__init__()

        self.Encoding_block1 = Encoding_Block(cin, n_feat)
        self.Encoding_block2 = Encoding_Block(n_feat, n_feat)
        self.Encoding_block3 = Encoding_Block(n_feat, n_feat)

        self.Decoding_block1 = Decoding_Block(n_feat, n_feat)
        self.Decoding_block2 = Decoding_Block(n_feat, n_feat)
        self.Decoding_block_End = Feature_Decoding_End(n_feat, cin)

        self.acti = torch.nn.PReLU()
        self.reset_parameters()
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x):

        encode0, down0 = self.Encoding_block1(x)
        encode1, down1 = self.Encoding_block2(down0)
        encode2, _ = self.Encoding_block3(down1)

        decode1 = self.Decoding_block1(encode2, encode1)
        decode0 = self.Decoding_block2(decode1, encode0)
        decode0 = self.Decoding_block_End(decode0)
        return decode0+x



class VANet(nn.Module):
    def __init__(self):
        super(VANet, self).__init__()

        self.in_channel = 64
        self.channel_fea = 64
        self.iter_num = 4

        ## map to image domain (rec)
        self.map_rec_x = nn.Conv2d(in_channels=self.channel_fea, out_channels=1, kernel_size=3, padding=3 // 2)
        self.map_rec_y = nn.Conv2d(in_channels=self.channel_fea, out_channels=1, kernel_size=3, padding=3 // 2)

        # initialize variables
        self.init_x = Unet_denoise(cin=self.in_channel, n_feat=self.channel_fea)
        self.init_c = Unet_denoise(cin=self.in_channel, n_feat=self.channel_fea)

        ## denoise networks
        self.denoise_x = Unet_denoise(cin=self.in_channel, n_feat=self.channel_fea)
        self.denoise_c = Unet_denoise(cin=self.in_channel, n_feat=self.channel_fea)
        self.denoise_u = Unet_denoise(cin=self.in_channel, n_feat=self.channel_fea)

        ## CDic layer
        self.E_x = nn.Sequential(*[nn.Conv2d(in_channels=self.in_channel, out_channels=self.channel_fea, kernel_size=3, padding=3 // 2), \
                                    nn.ReLU(), \
                                   nn.Conv2d(in_channels=self.channel_fea, out_channels=self.channel_fea, kernel_size=3, padding=3 // 2)])
        self.D_x = nn.Sequential(*[nn.Conv2d(in_channels=self.channel_fea, out_channels=self.channel_fea, kernel_size=3, padding=3 // 2),\
                                    nn.ReLU(), \
                                   nn.Conv2d(in_channels=self.channel_fea, out_channels=self.in_channel, kernel_size=3, padding=3 // 2)])
        self.E_c = nn.Sequential(*[nn.Conv2d(in_channels=self.in_channel, out_channels=self.channel_fea, kernel_size=3, padding=3 // 2), \
                                    nn.ReLU(), \
                                   nn.Conv2d(in_channels=self.channel_fea, out_channels=self.channel_fea, kernel_size=3, padding=3 // 2)])
        self.D_c = nn.Sequential(*[nn.Conv2d(in_channels=self.channel_fea, out_channels=self.channel_fea, kernel_size=3, padding=3 // 2), \
                                    nn.ReLU(), \
                                   nn.Conv2d(in_channels=self.channel_fea, out_channels=self.in_channel, kernel_size=3, padding=3 // 2)])

        # learnable parameters
        self.mu_x = [nn.Parameter(torch.tensor(0.1)) for _ in range(self.iter_num)]
        self.mu_c = [nn.Parameter(torch.tensor(0.1)) for _ in range(self.iter_num)]

        self.lamda_1 = [nn.Parameter(torch.tensor(1.0)) for _ in range(self.iter_num)]
        self.gamma_x = [nn.Parameter(torch.tensor(1.0)) for _ in range(self.iter_num)]
        self.gamma_c = [nn.Parameter(torch.tensor(1.0)) for _ in range(self.iter_num)]
        self.alpha = [nn.Parameter(torch.tensor(1.0)) for _ in range(self.iter_num)]
        self.beta = [nn.Parameter(torch.tensor(1.0)) for _ in range(self.iter_num)]

    
    def data_consistency_layer(self, generated, X_k, mask):

        gene_complex = real_to_complex(generated)
        output_complex = X_k + gene_complex * (1.0 - mask)
        output_img = complex_to_real(output_complex)
        
        return output_img
    
    def forward(self, x, y, mask):
        # x: input low-quality target image [b,1,h,w]
        # y: input high-quality ref image [b,1,h,w]
        # mask: undersampling mask [b,1,h,w]

        # repeat input image to high-dimension
        x = x.repeat(1,64,1,1)
        y = y.repeat(1,64,1,1)

        # x_k
        mask = mask.repeat(1,64,1,1)
        x_k = real_to_complex(x)

        # initialize variables
        x_t0 = self.init_x(x)
        c_t0 = self.init_c(y)
        u_t0 = y-c_t0
        P_t0 = c_t0
        Q_t0 = u_t0

        for i in range(self.iter_num):
            # update x
            DC_x = self.data_consistency_layer(x_t0, x_k, mask)  ## k-sapce DC
            refine = self.gamma_x[i]*self.D_x(self.E_x(x_t0)-self.E_c(c_t0)) + self.lamda_1[i]*self.denoise_x(x_t0)  # image refine
            x_t1 = DC_x - self.mu_x[i]*refine # update x
            if i != self.iter_num-1:
                # update C and U
                c_t1 = c_t0 - self.mu_c[i]*(c_t0+u_t0-self.gamma_c[i]*self.D_c(self.E_x(x_t0)-self.E_c(c_t0))-self.alpha[i]*(P_t0-c_t0))
                u_t1 = (y + self.beta[i]*Q_t0 - c_t0)/(1+self.beta[i])
                # update P and Q
                P_t1 = self.denoise_c(c_t1)
                Q_t1 = self.denoise_u(u_t1)            

            c_t0 = c_t1
            u_t0 = u_t1
            P_t0 = P_t1
            Q_t0 = Q_t1
            x_t0 = x_t1
        
        # map feature to image
        x_out = self.map_rec_x(x_t0)
        # y_out = self.map_rec_x(c_t0+u_t0)

        return x_out
        # return x_out, y_out

        