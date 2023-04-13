
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
