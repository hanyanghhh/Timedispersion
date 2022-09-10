
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn.functional import conv1d


class Backward_model(nn.Module):
    def __init__(self,resolution_ratio=4,nonlinearity="tanh"):
        super(Backward_model, self).__init__()
        self.resolution_ratio = resolution_ratio
        self.activation =  nn.ReLU() if nonlinearity=="relu" else nn.Tanh()

        self.cnn1 = nn.Sequential(nn.Conv1d(in_channels=1,
                                           out_channels=16,
                                           kernel_size=9,
                                           padding=4,
                                           dilation=1),
                                  nn.GroupNorm(num_groups=1,
                                               num_channels=16))

        self.cnn2 = nn.Sequential(nn.Conv1d(in_channels=1,
                                           out_channels=16,
                                           kernel_size=9,
                                           padding=12,
                                           dilation=3),
                                  nn.GroupNorm(num_groups=1,
                                               num_channels=16))

        self.cnn3 = nn.Sequential(nn.Conv1d(in_channels=1,
                                           out_channels=16,
                                           kernel_size=9,
                                           padding=24,
                                           dilation=6),
                                  nn.GroupNorm(num_groups=1,
                                               num_channels=16))

        self.cnn = nn.Sequential(self.activation,
                                 nn.Conv1d(in_channels=48,
                                           out_channels=32,
                                           kernel_size=7,
                                           padding=3),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=32),
                                 self.activation,

                                 nn.Conv1d(in_channels=32,
                                           out_channels=16,
                                           kernel_size=5,
                                           padding=2),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=16),
                                 self.activation,

                                 nn.Conv1d(in_channels=16,
                                           out_channels=8,
                                           kernel_size=3,
                                           padding=1),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=8),
                                 self.activation,

                                 nn.Conv1d(in_channels=8,
                                           out_channels=1,
                                           kernel_size=1,
                                           padding=0),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=1),
                                 self.activation
                                 )

        self.gru = nn.GRU(input_size=2001,
                          hidden_size=2001,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)

        #self.out0 = nn.Linear(in_features=2001*4, out_features=2001*2)
        self.out1 = nn.Linear(in_features=2001*2, out_features=2001)
        #self.out1 = nn.Linear(in_features=32, out_features=2001)


        self.gru_out = nn.GRU(input_size=1,
                              hidden_size=16,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True)
        self.out = nn.Linear(in_features=32, out_features=1)


        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        #print("input is",x.shape)
        cnn_out1 = self.cnn1(x)
        cnn_out2 = self.cnn2(x)
        cnn_out3 = self.cnn3(x)
        cnn_out = self.cnn(torch.cat((cnn_out1,cnn_out2,cnn_out3),dim=1))
        #cnn_out = self.cnn(torch.cat((cnn_out1, cnn_out2), dim=1))

        tmp_x = x
        rnn_out, _ = self.gru(tmp_x)
        #rnn_out = self.out0(rnn_out)
        rnn_out = self.out1(rnn_out)
        #rnn_out = rnn_out.transpose(-1, -2)
        #rnn_out = self.out1(rnn_out)


        #x = cnn_out
        #print(" cnn_out is ",cnn_out.shape)
        #print(" rnn_out  is", rnn_out.shape)
        #print(" x = rnn_out + cnn_out  is", x.shape)\]
        x = rnn_out + cnn_out
        #x = self.up(x)
        #print(" x = self.up(x) is", x.shape)

        tmp_x = x.transpose(-1, -2)
        x, _ = self.gru_out(tmp_x)
        #print(" x, _ = self.gru_out(tmp_x) is", x.shape)

        x = self.out(x)
        #print(" x = self.out(x) is", x.shape)
        x = x.transpose(-1,-2)
        #print(x.shape)
        return x


class Forward_model(nn.Module):
    def __init__(self,resolution_ratio=4,nonlinearity="tanh"):
        super(Forward_model, self).__init__()
        self.resolution_ratio = resolution_ratio
        self.activation =  nn.ReLU() if nonlinearity=="relu" else nn.Tanh()

        self.cnn1 = nn.Sequential(nn.Conv1d(in_channels=1,
                                           out_channels=16,
                                           kernel_size=9,
                                           padding=4,
                                           dilation=1),
                                  nn.GroupNorm(num_groups=1,
                                               num_channels=16))

        self.cnn2 = nn.Sequential(nn.Conv1d(in_channels=1,
                                           out_channels=16,
                                           kernel_size=9,
                                           padding=12,
                                           dilation=3),
                                  nn.GroupNorm(num_groups=1,
                                               num_channels=16))

        self.cnn3 = nn.Sequential(nn.Conv1d(in_channels=1,
                                           out_channels=16,
                                           kernel_size=9,
                                           padding=24,
                                           dilation=6),
                                  nn.GroupNorm(num_groups=1,
                                               num_channels=16))

        self.cnn = nn.Sequential(self.activation,
                                 nn.Conv1d(in_channels=48,
                                           out_channels=32,
                                           kernel_size=7,
                                           padding=3),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=32),
                                 self.activation,

                                 nn.Conv1d(in_channels=32,
                                           out_channels=16,
                                           kernel_size=5,
                                           padding=2),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=16),
                                 self.activation,

                                 nn.Conv1d(in_channels=16,
                                           out_channels=8,
                                           kernel_size=3,
                                           padding=1),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=8),
                                 self.activation,

                                 nn.Conv1d(in_channels=8,
                                           out_channels=1,
                                           kernel_size=1,
                                           padding=0),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=1),
                                 self.activation
                                 )

        self.gru = nn.GRU(input_size=2001,
                          hidden_size=2001*2,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)

        self.out0 = nn.Linear(in_features=2001*4, out_features=2001*2)
        self.out1 = nn.Linear(in_features=2001 * 2, out_features=2001 )
        #self.out1 = nn.Linear(in_features=32, out_features=2001)


        self.gru_out = nn.GRU(input_size=1,
                              hidden_size=16,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True)
        self.out = nn.Linear(in_features=32, out_features=1)


        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        #print("input is",x.shape)
        cnn_out1 = self.cnn1(x)
        cnn_out2 = self.cnn2(x)
        cnn_out3 = self.cnn3(x)
        cnn_out = self.cnn(torch.cat((cnn_out1,cnn_out2,cnn_out3),dim=1))
        #cnn_out = self.cnn(torch.cat((cnn_out1, cnn_out2), dim=1))

        tmp_x = x
        rnn_out, _ = self.gru(tmp_x)
        rnn_out = self.out0(rnn_out)
        rnn_out = self.out1(rnn_out)
        #rnn_out = rnn_out.transpose(-1, -2)
        #rnn_out = self.out1(rnn_out)


        #x = cnn_out
        #print(" cnn_out is ",cnn_out.shape)
        #print(" rnn_out  is", rnn_out.shape)
        #print(" x = rnn_out + cnn_out  is", x.shape)\]
        x = rnn_out + cnn_out
        #x = self.up(x)
        #print(" x = self.up(x) is", x.shape)

        tmp_x = x.transpose(-1, -2)
        x, _ = self.gru_out(tmp_x)
        #print(" x, _ = self.gru_out(tmp_x) is", x.shape)

        x = self.out(x)
        #print(" x = self.out(x) is", x.shape)
        x = x.transpose(-1,-2)
        #print(x.shape)
        return x