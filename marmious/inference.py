import numpy as np
import torch.autograd
from matplotlib.patches import ConnectionPatch
import numpy as np
import functools
import numpy as np
import torch.autograd
from matplotlib.patches import ConnectionPatch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import os
from torch.utils import data
from collections import defaultdict
from torch.nn import init

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
class Normalization:
    def __init__(self, mean_val=None,std_val=None):
        self.mean_val = mean_val
        self.std_val = std_val

    def normalize(self, x):
        return (x-self.mean_val)/self.std_val

    def unnormalize(self, x):
        return x*self.std_val + self.mean_val


def zone_and_linked2(ax, axins, zone_left, zone_right, x, y, linked='bottom',
                     a=0.0004, x_ratio=0.00005, y_ratio=0.005):
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    xlim_left = x[zone_left] - (x[zone_right] - x[zone_left]) * x_ratio
    xlim_right = x[zone_right] + (x[zone_right] - x[zone_left]) * x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data) - (np.max(y_data) - np.min(y_data)) * y_ratio
    ylim_top = np.max(y_data) + (np.max(y_data) - np.min(y_data)) * y_ratio

    #z_data = np.hstack([zi[zone_left:zone_right] for zi in z])
    #zlim_bottom = np.min(z_data) - (np.max(z_data) - np.min(z_data)) * z_ratio
    #zlim_top = np.max(z_data) + (np.max(z_data) - np.min(z_data)) * z_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)
    #axins.set_ylim(zlim_bottom, zlim_top)

    ax.plot([xlim_left, xlim_right, xlim_right, xlim_left, xlim_left],
            [ylim_bottom, ylim_bottom, ylim_top, ylim_top, ylim_bottom], "black")

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_left, ylim_bottom)
        xyA_2, xyB_2 = (xlim_right, ylim_top), (xlim_right, ylim_bottom)
    elif linked == 'top':
        xyA_1, xyB_1 = (xlim_left, ylim_bottom), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_right, ylim_top)
    elif linked == 'left':
        xyA_1, xyB_1 = (xlim_right, ylim_top), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_left, ylim_bottom)
    elif linked == 'right':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_right, ylim_top)
        xyA_2, xyB_2 = (xlim_left, ylim_bottom), (xlim_right, ylim_bottom)

    con = ConnectionPatch(xyA=xyA_1, xyB=xyB_1, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2, xyB=xyB_2, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)

def zone_and_linked1(ax, axins, zone_left, zone_right, x, y, linked='bottom',
                    x_ratio=0.05, y_ratio=0.05):
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    xlim_left = x[zone_left] - (x[zone_right] - x[zone_left]) * x_ratio
    xlim_right = x[zone_right] + (x[zone_right] - x[zone_left]) * x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data) - (np.max(y_data) - np.min(y_data)) * y_ratio
    ylim_top = np.max(y_data) + (np.max(y_data) - np.min(y_data)) * y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left, xlim_right, xlim_right, xlim_left, xlim_left],
            [ylim_bottom, ylim_bottom, ylim_top, ylim_top, ylim_bottom], "black")

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_left, ylim_bottom)
        xyA_2, xyB_2 = (xlim_right, ylim_top), (xlim_right, ylim_bottom)
    elif linked == 'top':
        xyA_1, xyB_1 = (xlim_left, ylim_bottom), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_right, ylim_top)
    elif linked == 'left':
        xyA_1, xyB_1 = (xlim_right, ylim_top), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_left, ylim_bottom)
    elif linked == 'right':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_right, ylim_top)
        xyA_2, xyB_2 = (xlim_left, ylim_bottom), (xlim_right, ylim_bottom)

    con = ConnectionPatch(xyA=xyA_1, xyB=xyB_1, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2, xyB=xyB_2, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)
def zone_and_linked(axins, zone_left, zone_right, x, y, linked='bottom',
                    x_ratio=1e-13, y_ratio=1e-13):

    """缩放内嵌图形，并且进行连线
    zone_and_linked(ax, axins, 1300, 1800, range(0, 2001),
                [y, y_pred], 'right')
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    xlim_left = x[zone_left] - (x[zone_right] - x[zone_left]) * x_ratio
    xlim_right = x[zone_right] + (x[zone_right] - x[zone_left]) * x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data) - (np.max(y_data) - np.min(y_data)) * y_ratio
    ylim_top = np.max(y_data) + (np.max(y_data) - np.min(y_data)) * y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)


def get_sin_curve(_amplitude, _signal_frequency, _sample_frequency, _phase_position, _time_series):
    # timeVary_sample：采样时间间隔

    timeVary_sample = 1 / _sample_frequency

    # samplePoints_num为时间序列采样点个数

    samplePoints_num = _time_series / timeVary_sample

    samplePoints_num = np.arange(samplePoints_num)

    sin = _amplitude * np.sin(
        2 * np.pi * _signal_frequency * samplePoints_num * timeVary_sample + _phase_position * (np.pi / 180))

    return sin
frequency = 2001
sin_signal = get_sin_curve(_amplitude=1, _signal_frequency=100, _sample_frequency=frequency, _phase_position=0,
                           _time_series=1)
"""

:param _amplitude: 振幅

:param _signal_frequency: 信号频率

:param _sample_frequency: 采样频率（Hz）

:param _phase_position: 相位

:param _time_series: 时间序列长度（s）

:return:

"""


# 生成一个 具有20个 0~（8001-1）之间的整数

def mea_std_norm(x):
	"""Normalize the input by its mean and standard deviation"""
	# x = (x - np.mean(x)) / np.std(x)
	x = x * np.std(x) + np.mean(x)
	return x

#model = CRNN0(1,1,1,2001)
#model = CRNN()
#model = inverse_model58() xxx
#model = inverse_model525()



class inverse_model522(nn.Module):
    def __init__(self,resolution_ratio=4,nonlinearity="tanh"):
        super(inverse_model522, self).__init__()
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
            #elif isinstance(m, nn.Linear):
                #m.bias.data.zero_()

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

class inverse_model619(nn.Module):
    def __init__(self,resolution_ratio=4,nonlinearity="tanh"):
        super(inverse_model619, self).__init__()
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

file_params1 = "./weight/backwardmodel_test499_loss0.0003094713379554874"###model_522 + 619
Gxy = torch.load(file_params1).cuda()  # 得到权重值



file_params2 = "./weight/forwardmodel_test499_loss0.0003094713379554874"###model_522 + 619
Gyx = torch.load(file_params2).cuda()  # 得到权重值

k=500
x = np.fromfile("./data/2D/0pinsan_test_time/record_1.2ms_shot102_2001x550.bin", dtype=np.float32).reshape(550, 2001)#1550x2001 double
x = (x[k, ::])
x_normalization = Normalization(mean_val=np.mean(x),
                                      std_val=np.std(x))
x = x_normalization.normalize(x)


# Impedance
y = np.fromfile("./data/2D/0pinsan_test_notime/record_0.2ms_no_time_dispersion_shot102_2001x550.bin", dtype=np.float32).reshape(550, 2001)
y = (y[k, ::])

y_normalization = Normalization(mean_val=np.mean(y), std_val=np.std(y))
y = y_normalization.normalize(y)



print("y.shape",y.shape) #torch.Size([13601, 1, 8001])
print("loading the Trained Model finished")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x1 = torch.tensor(x,dtype=torch.float).view(1, 1, 2001).to(device)
print("x1:" ,x1.shape)

y_pred = Gxy(x1).squeeze()
y_pred = y_pred.cpu().detach().numpy()
y_pred = y_pred



y1 = torch.tensor(y,dtype=torch.float).view(1, 1, 2001).to(device)
print("x1:" ,y1.shape)
x_pred = Gyx(y1).squeeze()
x_pred = x_pred.cpu().detach().numpy()
x_pred = x_pred




print("finished")
x = x_normalization.unnormalize(x)
y = y_normalization.unnormalize(y)
y_pred = y_normalization.unnormalize(y_pred)
x_pred = x_normalization.unnormalize(x_pred)

plt.plot(y, color="black", label="True AI", linestyle='dotted', linewidth=1.5)
plt.plot(y_pred, color="red", label='Predict', linewidth=0.5)
plt.plot(x, color="blue", label='Input', linewidth=1.0)
plt.show()
# 绘制主图
a=3.6 * (10 ** -3)
fig, axins = plt.subplots(1, 1, figsize=(14, 6))

# 在缩放图中也绘制主图所有内容，然后根据限制横纵坐标来达成局部显示的目的
axins.plot(y, color="black", label="Target", linestyle='dotted', linewidth=2)
axins.plot(y_pred, color="red", label='Predict', linewidth=0.5)
axins.plot(x, color="blue", label='Input', linewidth=1.0)
axins.set_title("Trace No.{index}".format(index=500), fontsize=30)
axins.set_xlabel("Time (s)", fontsize=30, labelpad=10.5)
axins.set_ylabel("Amplitude", fontsize=30, labelpad=10.5)
axins.set_xticks([1060, 1080, 1100, 1120, 1140, 1160, 1180])
axins.set_xticklabels([3.81, 3.88, 3.95, 4.02, 4.09, 4.16, 4.23])
labels = axins.get_xticklabels() + axins.get_yticklabels()
[label.set_fontsize(20) for label in labels]
# 局部显示并且进行连线
zone_and_linked(axins, 1060, 1180, range(0, 2001),[y, y_pred, x], 'right')
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.098, right=0.95)
axins.legend(prop={'size': 20})

plt.savefig('./Results_notimepred500.eps', format='pdf', dpi=500, bbox_inches='tight')
plt.savefig('./Results_notimepred500.pdf', dpi=500, bbox_inches='tight')
plt.show()

fig, axins = plt.subplots(1, 1, figsize=(14, 6))
# 在缩放图中也绘制主图所有内容，然后根据限制横纵坐标来达成局部显示的目的
axins.plot(x, color="black", label="Target", linestyle='dotted', linewidth=2)
axins.plot(x_pred, color="red", label='Predict', linewidth=0.5)
axins.plot(y, color="blue", label='Input', linewidth=1.0)
axins.set_title("Trace No.{index}".format(index=500), fontsize=30)
axins.set_xlabel("Time (s)", fontsize=30, labelpad=10.5)
axins.set_ylabel("Amplitude", fontsize=30, labelpad=10.5)
axins.set_xticks([1060, 1080, 1100, 1120, 1140, 1160, 1180])
axins.set_xticklabels([3.81, 3.88, 3.95, 4.02, 4.09, 4.16, 4.23])

labels = axins.get_xticklabels() + axins.get_yticklabels()
[label.set_fontsize(20) for label in labels]
# 局部显示并且进行连线
zone_and_linked(axins, 1060, 1180, range(0, 2001),[x, x_pred, y], 'right')
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.098, right=0.95)
axins.legend(prop={'size': 20})

plt.savefig('./Results_timepred500.eps', format='pdf',dpi=500, bbox_inches='tight')
plt.savefig('./Results_timepred500.pdf', dpi=500, bbox_inches='tight')
plt.show()








