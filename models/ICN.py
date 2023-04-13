
import math
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import torch
import argparse
import numpy as np

class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, ::2, :]

    def odd(self, x):
        return x[:, 1::2, :]

    def split(self,x):
        l = x.shape[1]
        index = (np.array(range(l)) % 24 < 6) | (np.array(range(l)) % 24 >= 18)
        index = index[::-1].copy()
        near = x[:, index, :]
        distant = x[:, ~index, :]
        return near, distant


    def forward(self, x):
        '''Returns the odd and even part'''
        # near, distant = self.split(x)
        # assert near.shape == distant.shape
        # return (near, distant)
        # print(x.shape)

        return (self.even(x), self.odd(x))




def dilation_index(dataset_name, weather):
    austin_poi = [3,  65,  97, 106,   3,  18, 148,  25,  12,  27,  51, 201,  34,
                     5, 190,  46,  81, 165,   5, 107,  74, 166,  32, 117,  23,   7,
                     30,   9,  93,  62,  26, 136, 104,  43,  12, 105,  29,   7,   6,
                     5,  87, 166,  40,  33, 111, 167,  15,  97, 206, 163,  10,  10,
                     62, 174,  82, 182, 154,  49,  15,   5, 122, 129,  29, 136, 131,
                     1,   5,   6,  29, 124,  15,  80, 166, 180, 104,   0,  88,  33,
                     33,   0, 141, 130,  54,   0, 100,   0,  38, 177,  76, 177, 134,
                     0, 185, 153,  61,   0, 112, 195,  49,  33,  84,   0, 172, 163,
                     118,  35,   5, 203, 183,  62, 174,  76,  96, 154, 135, 121, 203,
                     143, 104, 145,  33, 137,  60, 167, 126, 124, 124,   0,  73,  61,
                     80,  49, 140,   0, 192, 124,  31, 182, 166, 166, 104,  80,   0,
                     117,   0,  33,   0, 190,  84,   0,   0,  60,  33, 190, 113, 173,
                     37, 201, 131, 209, 159,   0,  93,  49,  15, 128, 139,  45, 182,
                     72,   0,   0,  48, 155, 110, 104,   0,  87,   0,   5,  73,   0,
                     137, 108,   0,  92, 124,  51,   0,  72, 153,   5, 134, 197,  47,
                     119,   0, 193,  33,  33,   5,  11, 104,  26,   0, 193, 145, 104,
                     0,  33,   0]
    austin_demo = [7,  15,  68,   7,  37, 110, 130,   3, 130,  11, 130,  48,  14,
                      142,  12,   1,  17,  49,  36, 190, 105,   9,  68,  79,  54,  37,
                      117,  53,  49,  36,  48, 128,  51,  49,  13,  11,  18,  25,  28,
                      11,   4, 210, 154,  51,  60,  82,  72, 153,  11,  36, 104,  65,
                      96,  27, 126,  96,  32, 108, 130,  33,  25, 168, 142,  86,  79,
                      51,   2, 191,  92, 108, 178, 175, 155,  24, 173, 194,  86,  85,
                      209,  64,  96,  96,  45,  85, 174,  98,  63, 101, 174,  43, 209,
                      145,  81, 167, 135, 177,  81, 127, 161, 122, 159, 124, 190, 143,
                      61, 136,  65,  63,  69,  42, 171, 178, 131, 102,  53, 178, 170,
                      190, 194, 205, 203, 199,  99,  61, 101, 182,  54,  97,  54, 126,
                      58, 135,  57, 209, 206,  94, 105, 160, 124,  95,  97,  74,  13,
                      103,  46,  94,  93, 169, 181, 150, 149, 200, 142,  47,  42, 165,
                      177, 131, 144, 137, 206,  98, 121, 197, 115, 155,  72,  93,  61,
                      197, 163, 110, 168,  74,  88, 151, 163,  95,  70, 156, 198, 148,
                      193, 139, 145, 204, 190, 112, 143, 204,  19,  67,  62, 182,  75,
                      155,  75, 163, 106, 121, 207, 165,  38, 120, 185, 194, 160, 200,
                      21,  78,  99]
    austin_tran = [48, 153,  66,  17, 157,  52, 136,  14,  28,  16,  11,  68,  90,
                      46,   7, 132,   9,   3,  32,  28, 129,  58,  37,  36, 157,  57,
                      186,  39,   8,  57,  27,  25,  39, 120,  59, 107,  23,  22,  10,
                      32,  31,  93, 126,  73, 126,  47,  13,  45,   0, 103,  61,  40,
                      5,  31, 103,  58,  68,  29,  21,  82, 103,  50,  17,  65,  29,
                      154,   2,   4,  11, 187,  74, 142, 187, 147,  70,  79, 126, 184,
                      125,  75,  56,  30,  59,  75,  86,  75, 141, 210, 147,  95,  12,
                      129, 193, 167, 131, 177,  13, 197, 140, 164, 194,  75, 162,  54,
                      135,  80, 154,  35, 191,  54, 115, 125,  60, 156, 146, 110, 190,
                      190, 177, 180,  33, 122, 153, 193,  75, 111, 202,  75, 136,  91,
                      47,  94,  15,  75, 197, 104, 128, 111, 198, 163, 150, 107,  33,
                      72,  75, 171, 114,  88, 199,  75, 140, 203, 201, 122,  65, 205,
                      113,  24, 173, 166, 166,  75, 102, 139,  99, 195, 160, 171,  67,
                      111,  75, 145, 145, 158, 185, 189,  75,  95, 137,  91, 200,  75,
                      142, 139,  77, 207, 192,  72,  75, 192, 117, 174, 189, 123,  98,
                      155,  75,  97, 138, 201, 138, 199, 126, 151,  75, 155, 150, 155,
                      75, 185, 121]
    chicago_poi = [ 45,  25,  90, 106, 117,  81,  84, 113,  92,  77,  60, 138, 125,
                        15, 120,  68, 122,  56,  87, 131,  45, 130,  38,  75,  75,   1,
                        60, 115,  86,  76, 104,  73,  42,  20,  58, 104,  54,  85, 132,
                        106,  30, 109,  32,  79,  80,  20,  51,   2,  78,  43, 127,  46,
                        111,  39,  36, 124,  17,  36,  34, 129,  26, 127,  51, 108,  71,
                        28,  30,  72,  58,  77,  25, 132,  67,  82,  63,  73, 112,   9,
                        48,  43,  44,   5,  73,  15,   6,  64, 121,  18,  90,  90,  96,
                        11,  59,  36,  68, 119,  90,  28,  72,  32,  43, 139,  37, 134,
                        112, 120,  39,  10,  63,  41,   3,  52,  76,  87,  44,  27,  36,
                        4,  32,  95, 105,  86,  16,  48,  55,  35,  32,  50,  32,  59,
                        21,  19,  38,  85, 103,  81, 105,  26,  17,  64,  36]
    chicago_demo = [ 15,  42,  16,   1,  36,  58,  71,  31,  12, 109,  75,  94,   8,
                        119,  27, 132,   2,  40,  90,  86,  83,  60, 101,  47, 119,  21,
                        132,  62, 122, 107,  87,   7, 113, 113,   0,  98,  42,  45,  31,
                        106,  54, 115,   1,  56,  73,  37,  58, 100, 107, 119, 112,  46,
                        73,  83,  40,  56,  96,  61,  46,  91,  21,  57,  27,  28,  47,
                        65,  98, 131,  75,  75,  83,   6,  73,  52,  68,  69, 116,  78,
                        77,  72,  95, 120,  99,  45, 118,  45,  19, 123, 106, 102, 101,
                        58,  93,  92,  10,  71,  56,  97, 132,  82,  47,  22,  64,  61,
                        40,  84,  39,  48, 104, 125, 101,  42,  50,  33, 124,  41,  76,
                        123,  84,  24,  52,  48,  64, 117, 126, 138, 125,   7,  62, 117,
                        7,  67,  15, 114, 112,  27, 111, 107, 125,  47,  38]
    chicago_tran = [ 11,   3,   5,   1,  16,   2,   8,  47,   6,  43,  50,   0,  51,
                        26,   9, 116,  61,  57,  65,  28,  11,  27,  31,  37,  23,  11,
                        13,  36,  19,  14,  97,  22,  44,  46,   3, 115,  27,  23,  31,
                        75,   8,   1,  96,   9,  32,  47,  53,  23,  44,  89,  10,  12,
                        34,  20,  17,  60,   5,  17, 115,  24,  32,  41,  27,  67, 103,
                        78, 106, 106, 104, 135,  80,  44,  45,  88,  97,  89,  45, 107,
                        94, 122, 129,  39, 133, 140, 114, 100,  94, 107,  73,  75,  73,
                        135, 132,  84,  78,  51,  42,  30,  66, 126,  85,  44, 122,  64,
                        68, 112,  67,  87,  58, 103,  99, 123, 105, 138,  84,  58,  15,
                        19, 131, 139, 138, 120,  90, 111, 127, 105,  99, 124, 137,  80,
                        114, 118,  92,  82, 118,  69, 130, 128, 120, 119,  83]
    chicago_half_poi = [ 51,  28, 101, 107, 137,  90,  94,  86, 133, 105,   7,  67, 103,
                        145,  16, 140,  76, 142,  63,  56, 125,  51, 152,  43,  84,  84,
                        81,  77,   1,  67, 135,  96,  85, 120, 119,  81, 159,  47,  21,
                        65, 120,  61,  95, 154, 123,  33, 127,  37,  88,  89,  77,  21,
                        58,   2,  87,  48,  19, 149,  52, 131,  44,  41, 144,  18,  41,
                        39, 151,  29, 149,  58, 126,  79,  31,  37,  33,  80,  65,  50,
                        28, 154,  75,  91,  70, 149,  81, 132,   7,  54,  48,  49,   5,
                        81,   7,  16,   6,  71, 141,  19,  46, 116, 101, 110, 103,  12,
                        161,  66,  41, 123,  76, 139, 101,  31,  80, 117,  37,  48,  99,
                        113,  42,  34, 132, 140, 119,  44,  11,  20,  70,  46,  31,  95,
                        44,  59,  85,  97,  49,  30,  41,   4,  37, 109, 121,  96,  17,
                        54,  62,  40,  37,   4, 142,  57,  37,  66,  22,  20,  79,  34,
                        119,  90, 121,  29,  18, 104,  41]
    chicago_half_demo = [ 73,  47,  17,  41,  41,   2,  79,   3,  35,  13, 127,  84,   2,
                        9, 139,  30, 154,   2,  45, 130,  96,  93,  67, 128,  53, 139,
                        28, 154,  26,  17,  69, 142, 124,  97,  61,   8,  92, 133, 133,
                        83,  27,  47,  51,  35, 123,  61, 135,   1,  63,  81,  68,  42,
                        65, 115, 124, 139, 105, 132,  52,  81,  93,  45,  63, 110, 103,
                        52, 102,  22,  50,  30,  31,  53,  72, 103, 112, 153,  84,  84,
                        38,   6,  81,  59,  76,  39,  77, 136,  87,  86,  56, 109, 140,
                        114,  36,  25, 138,  51,  20, 143,  44, 113, 118, 117,  65,  64,
                        128,  56, 105,  21,  11,  89,  63, 111,  27, 109, 152,  53,  92,
                        130,  71,  50,  45,  94, 110,  44,  54, 116, 120, 145, 104,  32,
                        117,  47,  57,  37, 144,  46,  85, 143,  94,  25, 148, 151,  71,
                        137, 146, 160, 145,  35,   8,   8,  69, 137, 114,  75,  16, 134,
                        132,  30, 131, 124, 145,  53,  43]
    chicago_half_tran = [ 12,   3,   5,   1,   7,   2,   9,  68,  53,   6,  48,  57,   0,
                        58,  29,  10, 136,  68,  64,  72,  31,  60,  69,  35,  42,  24,
                        20,  70,  12,  14,  69,  20,  15, 111,  59,  23,  64,  49,  52,
                        3, 135,  76,  24,  35,  84,   9,   1, 110,  10,  79, 111,  53,
                        60,  24,  49, 100, 100,  11,  13,  34,  21,  18,  50,   5,  18,
                        135,  25,  37,  46,  22,  27, 119,  87,  94, 112, 123, 120, 157,
                        89,  49, 107,  99, 111,  29, 100, 144, 124, 108, 140, 151,  44,
                        155,  93, 141, 134, 115, 116, 124, 101,  81,  84,  81, 157, 150,
                        120, 154,  94,  80, 128,  58,  47,  33, 128, 148, 146,  95,  96,
                        99, 142,  71, 104, 145,  66,  75,  97, 146,  65, 103, 108, 125,
                        114, 143, 121, 160,  94,  40,  16,  20, 153, 161, 160, 140, 101,
                        131, 149, 121, 114, 148, 147, 144, 103,  89, 134, 138, 105,  91,
                        138,  77, 152, 150, 140, 139, 141]
                    
    if dataset_name == 'Austin':
        poi = austin_poi
        demo = austin_demo
        tran = austin_tran
    elif dataset_name == 'Chicago':
        poi = chicago_poi
        demo = chicago_demo
        tran = chicago_tran
    elif dataset_name == 'Chicago_Halfyear':
        poi = chicago_half_poi
        demo = chicago_half_demo
        tran = chicago_half_tran
    else:
        raise Exception('No dilation index available!')
    
    if weather:
        poi += [-3,-2,-1]
        demo += [-3,-2,-1]
        tran += [-3,-2,-1]
    
    return poi, demo, tran



class Interactor(nn.Module):
    #### batch_size * channels * 2 * time_window
    #### in_planes = input_dim
    def __init__(self, in_planes, splitting=True,
                 kernel = 5, dropout=0.5, groups = 1, hidden_size = 1, INN = True, 
                 dataset = 'Chicago', ablation = None, weather = True):
        super(Interactor, self).__init__()
        self.modified = INN
        self.kernel_size = kernel
        self.dilation = 1
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.groups = groups
        self.dataset = dataset
        self.ablation = ablation
        self.weather = weather
        if self.kernel_size % 2 == 0:
            pad_l = self.dilation * (self.kernel_size - 2) // 2 + 1 #by default: stride==1
            pad_r = self.dilation * (self.kernel_size) // 2 + 1 #by default: stride==1

        else:
            pad_l = self.dilation * (self.kernel_size - 1) // 2 + 1 # we fix the kernel size of the second layer as 3.
            pad_r = self.dilation * (self.kernel_size - 1) // 2 + 1
        self.splitting = splitting
        self.split = Splitting()

        modules_P = []
        modules_U = []
        modules_psi = []
        modules_phi = []
        prev_size = 1

        size_hidden = self.hidden_size
        n_channel = 4
        if self.ablation:
            n_channel = 3

        modules_P += [
            nn.ReplicationPad2d((pad_l, pad_r,0,0)),

            nn.Conv2d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=(n_channel,self.kernel_size), dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Dropout(self.dropout),
            nn.Conv2d(int(in_planes * size_hidden), in_planes,
                      kernel_size=(1,3), stride=1, groups= self.groups),
            nn.Tanh()
        ]
        modules_U += [
            nn.ReplicationPad2d((pad_l, pad_r,0,0)),
            nn.Conv2d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=(n_channel,self.kernel_size), dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(int(in_planes * size_hidden), in_planes,
                      kernel_size=(1,3), stride=1, groups= self.groups),
            nn.Tanh()
        ]

        modules_phi += [
            nn.ReplicationPad2d((pad_l, pad_r,0,0)),
            nn.Conv2d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=(n_channel,self.kernel_size), dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(int(in_planes * size_hidden), in_planes,
                      kernel_size=(1,3), stride=1, groups= self.groups),
            nn.Tanh()
        ]
        modules_psi += [
            nn.ReplicationPad2d((pad_l, pad_r,0,0)),
            nn.Conv2d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=(n_channel,self.kernel_size), dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(int(in_planes * size_hidden), in_planes,
                      kernel_size=(1,3), stride=1, groups= self.groups),
            nn.Tanh()
        ]
        self.phi = nn.Sequential(*modules_phi)
        self.psi = nn.Sequential(*modules_psi)
        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

        self.poi_index, self.demo_index, self.tran_index = dilation_index(self.dataset, self.weather)


    def forward(self, x):

        if self.splitting:
            (x_even, x_odd) = self.split(x)

        else:
            (x_even, x_odd) = x

        if self.modified:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)

            ### channel dilation
            x_even_0 = x_even.unsqueeze(2)
            x_odd_0 = x_odd.unsqueeze(2)

            x_even_chn2 = torch.cat((x_even_0, x_even_0[:,self.poi_index,:,:], x_even_0[:,self.demo_index,:,:],
                                    x_even_0[:,self.tran_index,:,:]),2)
            x_odd_chn2 = torch.cat((x_odd_0, x_odd_0[:, self.poi_index, :, :], x_odd_0[:, self.demo_index, :, :],
                                    x_odd_0[:, self.tran_index, :, :]), 2)
            
            if self.weather:
                x_even = x_even[:,]


            if self.ablation == 'poi':
                x_even_chn2 = x_even_chn2[:,:,[0,2,3],:]
                x_odd_chn2 = x_odd_chn2[:,:,[0,2,3],:]
            elif self.ablation == 'demo':
                x_even_chn2 = x_even_chn2[:,:,[0,1,3],:]
                x_odd_chn2 = x_odd_chn2[:,:,[0,1,3],:]
            elif self.ablation == 'tran':
                x_even_chn2 = x_even_chn2[:,:,[0,1,2],:]
                x_odd_chn2 = x_odd_chn2[:,:,[0,1,2],:]

            d = x_odd.mul(torch.exp(self.phi(x_even_chn2).reshape(x_even.shape)))
            c = x_even.mul(torch.exp(self.psi(x_odd_chn2).reshape(x_odd.shape)))

            d_0 = d.unsqueeze(2)
            c_0 = c.unsqueeze(2)

            d_ch2 = torch.cat((d_0, d_0[:,self.poi_index,:,:], d_0[:,self.demo_index,:,:], d_0[:,self.tran_index,:,:]),2)
            c_ch2 = torch.cat((c_0, c_0[:, self.poi_index, :, :], c_0[:, self.demo_index, :, :], c_0[:, self.tran_index, :, :]), 2)

            if self.ablation == 'poi':
                d_ch2 = d_ch2[:,:,[0,2,3],:]
                c_ch2 = c_ch2[:,:,[0,2,3],:]
            elif self.ablation == 'demo':
                d_ch2 = d_ch2[:,:,[0,1,3],:]
                c_ch2 = c_ch2[:,:,[0,1,3],:]
            elif self.ablation == 'tran':
                d_ch2 = d_ch2[:,:,[0,1,2],:]
                c_ch2 = c_ch2[:,:,[0,1,2],:]

            x_even_update = c + self.U(d_ch2).reshape(d.shape)
            x_odd_update = d - self.P(c_ch2).reshape(c.shape)

            return (x_even_update, x_odd_update)

        else:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)

            x_even_0 = x_even.unsqueeze(2)

            x_even_chn2 = torch.cat((x_even_0, x_even_0[:, self.poi_index, :, :], x_even_0[:, self.demo_index, :, :],
                                     x_even_0[:, self.tran_index, :, :]), 2)

            if self.ablation == 'poi':
                x_even_chn2 = x_even_chn2[:,:,[0,2,3],:]
            elif self.ablation == 'demo':
                x_even_chn2 = x_even_chn2[:,:,[0,1,3],:]
            elif self.ablation == 'tran':
                x_even_chn2 = x_even_chn2[:,:,[0,1,2],:]

            # x_even_chn2 = torch.cat((x_even_0, x_even_0[:, self.demo_index, :, :], x_even_0[:, self.tran_index, :, :]), 2)

            ### poi/demo only
            # x_even_chn2 = torch.cat((x_even_0, x_even_0[:, self.all_index, :, :]), 2)

            d = x_odd - self.P(x_even_chn2).reshape(x_even.shape)

            d_0 = d.unsqueeze(2)

            d_ch2 = torch.cat((d_0, d_0[:, self.poi_index, :, :], d_0[:, self.demo_index, :, :], d_0[:, self.tran_index, :, :]), 2)
            if self.ablation == 'poi':
                d_ch2 = d_ch2[:,:,[0,2,3],:]
            elif self.ablation == 'demo':
                d_ch2 = d_ch2[:,:,[0,1,3],:]
            elif self.ablation == 'tran':
                d_ch2 = d_ch2[:,:,[0,1,2],:]

            # d_ch2 = torch.cat((d_0, d_0[:, self.demo_index, :, :], d_0[:, self.tran_index, :, :]), 2)

            # d_ch2 = torch.cat((d_0, d_0[:, self.all_index, :, :]), 2)

            c = x_even + self.U(d_ch2).reshape(d.shape)

            return (c, d)


class InteractorLevel(nn.Module):
    def __init__(self, in_planes, kernel, dropout, groups , hidden_size, INN, 
                 dataset, ablation, weather):
        super(InteractorLevel, self).__init__()
        self.level = Interactor(in_planes = in_planes, splitting=True,
                 kernel = kernel, dropout=dropout, groups = groups, hidden_size = hidden_size, 
                 INN = INN, dataset = dataset, ablation = ablation, weather=weather)

    def forward(self, x):
        (x_even_update, x_odd_update) = self.level(x)
        return (x_even_update, x_odd_update)

class LevelICN(nn.Module):
    def __init__(self,in_planes, kernel_size, dropout, groups, hidden_size, INN, 
                 dataset, ablation, weather):
        super(LevelICN, self).__init__()
        self.interact = InteractorLevel(in_planes= in_planes, kernel = kernel_size, dropout = dropout, groups =groups , 
                                        hidden_size = hidden_size, INN = INN, dataset = dataset, ablation = ablation,
                                        weather=weather)

    def forward(self, x):
        (x_even_update, x_odd_update) = self.interact(x)
        return x_even_update.permute(0, 2, 1), x_odd_update.permute(0, 2, 1) #even: B, T, D odd: B, T, D

class ICN_Tree(nn.Module):
    def __init__(self, in_planes, current_level, kernel_size, dropout, groups, hidden_size, INN, 
                 dataset, ablation, weather):
        super().__init__()
        self.current_level = current_level


        self.workingblock = LevelICN(
            in_planes = in_planes,
            kernel_size = kernel_size,
            dropout = dropout,
            groups= groups,
            hidden_size = hidden_size,
            INN = INN, dataset = dataset,
            ablation = ablation,
            weather=weather)

        if current_level!=0:
            self.ICN_Tree_odd=ICN_Tree(in_planes, current_level-1, kernel_size, dropout,
                                       groups, hidden_size, INN, dataset, ablation,weather)
            self.ICN_Tree_even=ICN_Tree(in_planes, current_level-1, kernel_size, dropout, 
                                        groups, hidden_size, INN, dataset, ablation,weather)
    
    def zip_up_the_pants(self, even, odd):
        even = even.permute(1, 0, 2)
        odd = odd.permute(1, 0, 2) #L, B, D
        even_len = even.shape[0]
        odd_len = odd.shape[0]
        mlen = min((odd_len, even_len))
        _ = []
        for i in range(mlen):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        if odd_len < even_len: 
            _.append(even[-1].unsqueeze(0))
        return torch.cat(_,0).permute(1,0,2) #B, L, D
        
    def forward(self, x):
        x_even_update, x_odd_update= self.workingblock(x)
        # We recursively reordered these sub-series. You can run the ./utils/recursive_demo.py to emulate this procedure. 
        if self.current_level ==0:
            return self.zip_up_the_pants(x_even_update, x_odd_update)
        else:
            return self.zip_up_the_pants(self.ICN_Tree_even(x_even_update), self.ICN_Tree_odd(x_odd_update))

class EncoderTree(nn.Module):
    def __init__(self, in_planes,  num_levels, kernel_size, dropout, groups, hidden_size, INN, 
                 dataset, ablation, weather):
        super().__init__()
        self.levels=num_levels
        self.ICN_Tree = ICN_Tree(
            in_planes = in_planes,
            current_level = num_levels-1,
            kernel_size = kernel_size,
            dropout =dropout ,
            groups = groups,
            hidden_size = hidden_size,
            INN = INN, dataset = dataset,
            ablation= ablation,
            weather=weather)
        
    def forward(self, x):

        x= self.ICN_Tree(x)

        return x

class ICN(nn.Module):
    def __init__(self, output_len, input_len, input_dim = 9, hid_size = 1, num_stacks = 1,
                num_levels = 3, num_decoder_layer = 1, concat_len = 0, groups = 1, kernel = 5, dropout = 0.5,
                 single_step_output_One = 0, input_len_seg = 0, positionalE = False, modified = True, RIN=False, 
                 dataset = 'Chicago', ablation = None, weather = True):
        super(ICN, self).__init__()

        self.input_dim = input_dim
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_size = hid_size
        self.num_levels = num_levels
        self.groups = groups
        self.modified = modified
        self.kernel_size = kernel
        self.dropout = dropout
        self.single_step_output_One = single_step_output_One
        self.concat_len = concat_len
        self.pe = positionalE
        self.RIN=RIN
        self.num_decoder_layer = num_decoder_layer
        self.dataset = dataset
        self.ablation = ablation
        self.weather = weather

        self.blocks1 = EncoderTree(
            in_planes=self.input_dim,
            num_levels = self.num_levels,
            kernel_size = self.kernel_size,
            dropout = self.dropout,
            groups = self.groups,
            hidden_size = self.hidden_size,
            INN =  modified, dataset = self.dataset,
            ablation = self.ablation,
            weather = self.weather)

        if num_stacks == 2: # we only implement two stacks at most.
            self.blocks2 = EncoderTree(
                in_planes=self.input_dim,
                num_levels = self.num_levels,
                kernel_size = self.kernel_size,
                dropout = self.dropout,
                groups = self.groups,
                hidden_size = self.hidden_size,
                INN =  modified, dataset = self.dataset,
                ablation = self.ablation,
                weather = self.weather)

        self.stacks = num_stacks

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self.projection1 = nn.Conv1d(self.input_len, self.output_len, kernel_size=1, stride=1, bias=False)
        self.div_projection = nn.ModuleList()
        self.overlap_len = self.input_len//4
        self.div_len = self.input_len//6

        if self.num_decoder_layer > 1:
            self.projection1 = nn.Linear(self.input_len, self.output_len)
            for layer_idx in range(self.num_decoder_layer-1):
                div_projection = nn.ModuleList()
                for i in range(6):
                    lens = min(i*self.div_len+self.overlap_len,self.input_len) - i*self.div_len
                    div_projection.append(nn.Linear(lens, self.div_len))
                self.div_projection.append(div_projection)

        if self.single_step_output_One: # only output the N_th timestep.
            if self.stacks == 2:
                if self.concat_len:
                    self.projection2 = nn.Conv1d(self.concat_len + self.output_len, 1,
                                                kernel_size = 1, bias = False)
                else:
                    self.projection2 = nn.Conv1d(self.input_len + self.output_len, 1,
                                                kernel_size = 1, bias = False)
        else: # output the N timesteps.
            if self.stacks == 2:
                if self.concat_len:
                    self.projection2 = nn.Conv1d(self.concat_len + self.output_len, self.output_len,
                                                kernel_size = 1, bias = False)
                else:
                    self.projection2 = nn.Conv1d(self.input_len + self.output_len, self.output_len,
                                                kernel_size = 1, bias = False)

        # For positional encoding
        self.pe_hidden_size = input_dim
        if self.pe_hidden_size % 2 == 1:
            self.pe_hidden_size += 1
    
        num_timescales = self.pe_hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0

        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                max(num_timescales - 1, 1))
        temp = torch.arange(num_timescales, dtype=torch.float32)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)

        ### RIN Parameters ###
        if self.RIN:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, input_dim))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, input_dim))
    
    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32, device=x.device)  # tensor([0., 1., 2., 3., 4.], device='cuda:0')
        temp1 = position.unsqueeze(1)  # 5 1
        temp2 = self.inv_timescales.unsqueeze(0)  # 1 256
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)  # 5 256
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)  #[T, C]
        signal = F.pad(signal, (0, 0, 0, self.pe_hidden_size % 2))
        signal = signal.view(1, max_length, self.pe_hidden_size)
    
        return signal

    def forward(self, x):
        assert self.input_len % (np.power(2, self.num_levels)) == 0 # evenly divided the input length into two parts. (e.g., 32 -> 16 -> 8 -> 4 for 3 levels)
        if self.pe:
            pe = self.get_position_encoding(x)
            if pe.shape[2] > x.shape[2]:
                x += pe[:, :, :-1]
            else:
                x += self.get_position_encoding(x)

        ### activated when RIN flag is set ###
        if self.RIN:
            print('/// RIN ACTIVATED ///\r',end='')
            means = x.mean(1, keepdim=True).detach()
            #mean
            x = x - means
            #var
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            # affine
            # print(x.shape,self.affine_weight.shape,self.affine_bias.shape)
            x = x * self.affine_weight + self.affine_bias

        # the first stack
        res1 = x
        x = self.blocks1(x)
        x += res1
        if self.num_decoder_layer == 1:
            x = self.projection1(x)
        else:
            x = x.permute(0,2,1)
            for div_projection in self.div_projection:
                output = torch.zeros(x.shape,dtype=x.dtype).cuda()
                for i, div_layer in enumerate(div_projection):
                    div_x = x[:,:,i*self.div_len:min(i*self.div_len+self.overlap_len,self.input_len)]
                    output[:,:,i*self.div_len:(i+1)*self.div_len] = div_layer(div_x)
                x = output
            x = self.projection1(x)
            x = x.permute(0,2,1)

        if self.stacks == 1:
            ### reverse RIN ###
            if self.RIN:
                x = x - self.affine_bias
                x = x / (self.affine_weight + 1e-10)
                x = x * stdev
                x = x + means

            return x

        elif self.stacks == 2:
            MidOutPut = x
            if self.concat_len:
                x = torch.cat((res1[:, -self.concat_len:,:], x), dim=1)
            else:
                x = torch.cat((res1, x), dim=1)

            # the second stack
            res2 = x
            x = self.blocks2(x)
            x += res2
            x = self.projection2(x)
            
            ### Reverse RIN ###
            if self.RIN:
                MidOutPut = MidOutPut - self.affine_bias
                MidOutPut = MidOutPut / (self.affine_weight + 1e-10)
                MidOutPut = MidOutPut * stdev
                MidOutPut = MidOutPut + means

            if self.RIN:
                x = x - self.affine_bias
                x = x / (self.affine_weight + 1e-10)
                x = x * stdev
                x = x + means

            return x, MidOutPut


def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--window_size', type=int, default=96)
    parser.add_argument('--horizon', type=int, default=12)

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--groups', type=int, default=1)

    parser.add_argument('--hidden-size', default=1, type=int, help='hidden channel of module')
    parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
    parser.add_argument('--kernel', default=3, type=int, help='kernel size')
    parser.add_argument('--dilation', default=1, type=int, help='dilation')
    parser.add_argument('--positionalEcoding', type=bool, default=True)

    parser.add_argument('--single_step_output_One', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='Chicago', choices=['Austin', 'Chicago'])
    parser.add_argument('--ablation', type=str, default=None, choices=[None, 'poi', 'demo', 'tran'])
    parser.add_argument('--weather', type=bool, default=True)

    args = parser.parse_args()

    model = ICN(output_len = args.horizon, input_len= args.window_size, input_dim = 9, hid_size = args.hidden_size, num_stacks = 1,
                num_levels = 3, concat_len = 0, groups = args.groups, kernel = args.kernel, dropout = args.dropout,
                 single_step_output_One = args.single_step_output_One, positionalE =  args.positionalEcoding, modified = True, 
                 dataset= args.dataset, ablation = args.ablation, weather=args.weather).cuda()
    x = torch.randn(32, 96, 9).cuda()
    y = model(x)
    print(y.shape)
