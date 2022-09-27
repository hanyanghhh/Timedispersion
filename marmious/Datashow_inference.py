import numpy as np
import torch.autograd
from matplotlib.patches import ConnectionPatch
import numpy as np
import torch.nn as nn
import glob
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torch.utils import data
from tqdm import tqdm
from collections import defaultdict
from torch.nn import init



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



#model1 = torch.load("/data/hany/weight/backwardmodel_test499_loss0.0003094713379554874")  ###model_522 + 619
#model2 = torch.load("/data/hany/weight/forwardmodel_test499_loss0.0003094713379554874")  ###model_522 + 619

class Normalization:
    def __init__(self, mean_val=None,std_val=None):
        self.mean_val = mean_val
        self.std_val = std_val

    def normalize(self, x):
        return (x-self.mean_val)/self.std_val

    def unnormalize(self, x):
        return x*self.std_val + self.mean_val

def metrics(y,x):
    #x: reference signal
    #y: estimated signal
    if torch.is_tensor(x):
        if x.is_cuda:
            x = x.cpu()
        x = x.numpy()
    if torch.is_tensor(y):
        if y.is_cuda:
            y = y.cpu()
        y = y.numpy()

    #corrlation
    x_mean = np.mean(x, axis=-1, keepdims=True)
    y_mean = np.mean(y, axis=-1, keepdims=True)
    x_std = np.std(x, axis=-1, keepdims=True)
    y_std = np.std(y, axis=-1, keepdims=True)
    corr = np.mean((x-x_mean)*(y-y_mean), axis=-1,keepdims=True)/(x_std*y_std)

    #coefficeint of determination (r2)
    S_tot = np.sum((x-x_mean)**2, axis=-1, keepdims=True)
    S_res = np.sum((x - y)**2, axis=-1, keepdims=True)

    r2 = (1-S_res/S_tot)

    return torch.tensor(corr), torch.tensor(r2)

def display_results(loss, property_corr, property_r2):
    property_corr = torch.mean(torch.cat(property_corr), dim=0).squeeze()
    property_r2 = torch.mean(torch.cat(property_r2), dim=0).squeeze()
    loss = torch.mean(torch.tensor(loss))
    #print("loss: {:.4f}\nCorrelation: {:0.4f}\nr2 Coeff.  : {:0.4f}".format(loss,property_corr,property_r2))
    #print("loss: {:.4f}".format(loss))
    print("loss: {:.2e}\nCorrelation: {:.4f}\nr2 Coeff.  : {:.4f}".format(loss, property_corr, property_r2))
    return loss.item()


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


def zone_and_linked_xychange(axins, zone_left, zone_right, x, y, linked='bottom',
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
    ylim_right = y[zone_left] - (y[zone_right] - y[zone_left]) * y_ratio
    ylim_left = y[zone_right] + (y[zone_right] - y[zone_left]) * y_ratio

    x_data = np.hstack([xi[zone_left:zone_right] for xi in x])
    xlim_bottom = np.min(x_data) - (np.max(x_data) - np.min(x_data)) * x_ratio
    xlim_top = np.max(x_data) + (np.max(x_data) - np.min(x_data)) * x_ratio

    axins.set_xlim(350, 530)
    axins.set_ylim(ylim_left, ylim_right)
    a1 = 3.6 * (10 ** -3)
    a2 = 10 * (10 ** -3)
    ax.set_xticks([360, 380, 400, 420, 440, 460, 480, 500, 520])
    ax.set_xticklabels([1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
    ax.set_yticks([900, 950, 1000, 1050, 1100, 1150, 1200, 1250])
    ax.set_yticklabels([3.24, 3.42, 3.60, 3.78, 3.96, 4.14, 4.32, 4.50])
def zone_and_linked_xychange_plt(axins, zone_left, zone_right, x, y, linked='bottom',
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
    ylim_right = y[zone_left] - (y[zone_right] - y[zone_left]) * y_ratio
    ylim_left = y[zone_right] + (y[zone_right] - y[zone_left]) * y_ratio

    x_data = np.hstack([xi[zone_left:zone_right] for xi in x])
    xlim_bottom = np.min(x_data) - (np.max(x_data) - np.min(x_data)) * x_ratio
    xlim_top = np.max(x_data) + (np.max(x_data) - np.min(x_data)) * x_ratio

    #axins.set_xlim(0, 450)
    #axins.set_ylim(ylim_left, ylim_right)

    ax.plot([350, 530, 530, 350, 350],
            [ylim_left, ylim_left, ylim_right, ylim_right, ylim_left], "red")


def sort_list_IDs(list_IDs):
    list_nums = [int(i.split(".")[0]) for i in list_IDs]
    list_sort = sorted(enumerate(list_nums), key=lambda x: x[1])
    list_index = [i[0] for i in list_sort]
    list_IDs_new = [list_IDs[i] for i in list_index]
    return list_IDs_new
class Reshape(object):
    """Reshape the data to a given size."""

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, data):
        data = np.reshape(data, self.output_size)

        return data

class ToTensor(object):
    """Convert ndarrys in sample to Tensors"""

    def __call__(self, data):
        data = data.transpose((0, 1))
        return data

def mea_std_norm(x):
    """Normalize the input by its mean and standard deviation"""
    x = (x - np.mean(x)) / np.std(x)
    return x
class Dataset(data.Dataset):
    def __init__(self, root_dir, list_IDs, transform=None, only_load_input=False):
        'Initialization'
        self.list_IDs = list_IDs
        self.root_dir = root_dir
        self.transform = transform
        self.only_load_input = only_load_input
        self.seis_dir = os.path.join(root_dir, 'x')
        self.reserve_bit = 0
        if not self.only_load_input:
            self.rgt_dir = os.path.join(root_dir, 'y')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        ID = self.list_IDs[index]
        seis_path = os.path.join(self.seis_dir, ID)
        if not self.only_load_input:
            rgt_path = os.path.join(self.rgt_dir, ID)
        X = np.fromfile(seis_path, dtype=np.float32)
        X = X.astype(np.float32)
        #print("X.shape is",X.shape)
        #X = X[:2000,]
        frequency = 2001
        #sin_signal = get_sin_curve(_amplitude=1, _signal_frequency=280, _sample_frequency=frequency, _phase_position=0, _time_series=1)
        #X = X + sin_signal
        X = self.transform(X)
        X = mea_std_norm(X)
        X = torch.from_numpy(X)
        if not self.only_load_input:
            Y = np.fromfile(rgt_path, dtype=np.float32)
            #Y = Y[:2000,]
            #Y = Y + sin_signal
            # Y= np.fromfile(seis_path, dtype=np.int8)
            Y = self.transform(Y)
            Y = mea_std_norm(Y)
            Y = torch.from_numpy(Y)
            return X, Y
        else:
            return X


if __name__ == '__main__':



    Gxy = torch.load("./weight/backwardmodel_test499_loss0.0003094713379554874").cuda()
    Gyx = torch.load("./weight/forwardmodel_test499_loss0.0003094713379554874").cuda()


    criterion = nn.SmoothL1Loss(reduction="sum")
    predicted_y = []
    true_y = []
    predicted_x = []
    true_x = []
    test_property_corr = []
    test_property_r2 = []
    Gxy.eval()
    Gyx.eval()
    print("\nTesting the model\n")

    with torch.no_grad():
        test_loss = []
        for k in range(0, 550, 1):
            f1 = np.fromfile("/data/hany/0pinsan_test_notrain_time/record_1.2ms_shot102_2001x550.bin",dtype=np.float32).reshape(550, 2001)
            x = f1[k, ::]

            f2 = np.fromfile("/data/hany/0pinsan_test_notrain_notime/record_0.2ms_no_time_dispersion_shot102_2001x550.bin",dtype=np.float32).reshape(550, 2001)  # 1550x2001 double
            y = f2[k, ::]

            x_normalization = Normalization(mean_val=np.mean(x),
                                            std_val=np.std(x))

            y_normalization = Normalization(mean_val=np.mean(y),
                                      std_val=np.std(y))

            x = x_normalization.normalize(x)
            y = y_normalization.normalize(y)


            x = torch.tensor(x, dtype=torch.float).view(1, 1, 2001).cuda()
            y = torch.tensor(y, dtype=torch.float).view(1, 1, 2001).cuda()

            y_pred = Gxy(x)
            corr, r2 = metrics(y_pred.detach(), y.detach())
            test_property_corr.append(corr)
            test_property_r2.append(r2)

            x_rec = Gyx(y_pred)

            seismic_loss = criterion(x_rec, x) / np.prod(x.shape)


            property_loss = criterion(y_pred, y) / np.prod(y.shape)
            unlabeledloss = criterion(x_rec, x) / np.prod(x.shape)
            loss = 1 * property_loss + 0.2 * unlabeledloss
            test_loss.append(loss.item())

            x_pred = Gyx(y)


            x = x_normalization.unnormalize(x)
            x_pred = x_normalization.unnormalize(x_pred)

            
            y = y_normalization.unnormalize(y)
            y_pred = y_normalization.unnormalize(y_pred)
            

            true_x.append(x)
            predicted_x.append(x_pred)
            true_y.append(y)
            predicted_y.append(y_pred)

        display_results(test_loss, test_property_corr, test_property_r2)

        predicted_x = torch.cat(predicted_x, dim=0)
        true_x = torch.cat(true_x, dim=0)
        
        predicted_y = torch.cat(predicted_y, dim=0)
        true_y = torch.cat(true_y, dim=0)


        if torch.cuda.is_available():
            predicted_x = predicted_x.cpu()
            true_x = true_x.cpu()
            
            predicted_y = predicted_y.cpu()
            true_y = true_y.cpu()

        predicted_x = predicted_x.numpy()
        true_x = true_x.numpy()
        
        predicted_y = predicted_y.numpy()
        true_y = true_y.numpy()
        
        #######################
        import matplotlib
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        c=1
        d=2e2
        left = true_x.min()/d
        right = true_x.max()/d

        font_xy=10
        font_car=10
        font_label=15

        ##########################################
        # plot absolute different true-predicted #
        ##########################################
        a1=3.6*(10**-3)
        a2=10*(10**-3)


        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        cax = ax.imshow(true_x[:, 0].T/c, cmap='Greys', aspect=0.6, vmin=left, vmax=right)


        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontsize(10) for label in labels]
        zone_and_linked_xychange(ax, 900, 1280, [true_x[:, 0].T/c], range(0,2001), 'right')
        #zone_and_linked_xychange_plt(ax, 900, 1280, [true_x[:, 0].T / c], range(0, 2001), 'right')
        fig.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95)
        ax.set_xlabel("Offset (km)", fontsize=font_label, labelpad=8.5)
        ax.set_ylabel("Time (s)", fontsize=font_label, labelpad=8.5)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        cbar = fig.colorbar(cax, shrink=0.6, pad=0.02)
        cbar.ax.tick_params(labelsize=font_car)
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        #fig, ax = plt.subplots()
        #abs(true_impedance[:, 0].T-predicted_impedance[:, 0].T)
        dif = abs(true_y[:, 0].T - predicted_y[:, 0].T)


        cax = ax.imshow(dif, cmap='Greys', aspect=0.6, vmin=left, vmax=right)
        ax.set_yticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
        ax.set_yticklabels([0, 0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2])
        ax.set_xticks([20, 120,  220,  320, 420, 520])
        ax.set_xticklabels([-2, -1, 220*0,  1, 2,  3])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontsize(10) for label in labels]
        zone_and_linked_xychange(ax, 900, 1280, [true_x[:, 0].T / c], range(0, 2001), 'right')
        #zone_and_linked_xychange_plt(ax, 900, 1280, [true_x[:, 0].T / c], range(0, 2001), 'right')
        fig.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95)
        ax.set_xlabel("Offset (km)", fontsize=font_label, labelpad=8.5)
        ax.set_ylabel("Time (s)", fontsize=font_label, labelpad=8.5)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        cbar = fig.colorbar(cax, shrink=0.6, pad=0.02)
        cbar.ax.tick_params(labelsize=font_car)
        plt.savefig('./2d_Dif_Results_notime_true.eps', dpi=500, format='pdf', facecolor='w', edgecolor='w',bbox_inches='tight')
        plt.savefig('./2d_Dif_Results_notime_true.pdf', dpi=500, format='pdf', facecolor='w', edgecolor='w',bbox_inches='tight')
        plt.show()



        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        dif = abs(true_x[:, 0].T - predicted_x[:, 0].T)


        cax = ax.imshow(dif, cmap='Greys', aspect=0.6, vmin=left, vmax=right)
        ax.set_yticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
        ax.set_yticklabels([0, 0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2])
        ax.set_xticks([20, 120,  220,  320, 420, 520])
        ax.set_xticklabels([-2, -1, 0,  1, 2,  3])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontsize(10) for label in labels]
        zone_and_linked_xychange(ax, 900, 1280, [true_x[:, 0].T/c], range(0,2001), 'right')
        #zone_and_linked_xychange_plt(ax, 900, 1280, [true_x[:, 0].T / c], range(0, 2001), 'right')
        fig.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95)
        ax.set_xlabel("Offset (km)", fontsize=font_label, labelpad=8.5)
        ax.set_ylabel("Time (s)", fontsize=font_label, labelpad=8.5)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        cbar = fig.colorbar(cax, shrink=0.6, pad=0.02)
        cbar.ax.tick_params(labelsize=font_car)
        plt.savefig('./2d_Dif_Results_time_true.eps', dpi=500, format='pdf', facecolor='w', edgecolor='w',bbox_inches='tight')
        plt.savefig('./2d_Dif_Results_time_true.pdf', dpi=500, facecolor='w', edgecolor='w',bbox_inches='tight')
        plt.show()




        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        #fig, ax = plt.subplots()
        cax = ax.imshow(predicted_y[:, 0].T/c, cmap='Greys', aspect=0.6, vmin=left, vmax=right)
        ax.set_yticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
        ax.set_yticklabels([0, 0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2])
        ax.set_xticks([20, 120,  220,  320, 420, 520])
        ax.set_xticklabels([-2, -1, 0,  1, 2,  3])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontsize(10) for label in labels]
        zone_and_linked_xychange(ax, 900, 1280, [predicted_y[:, 0].T/c], range(0,2001), 'right')
        #zone_and_linked_xychange_plt(ax, 900, 1280, [true_x[:, 0].T / c], range(0, 2001), 'right')
        fig.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95)
        ax.set_xlabel("Offset (km)", fontsize=font_label, labelpad=8.5)
        ax.set_ylabel("Time (s)", fontsize=font_label, labelpad=8.5)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        cbar = fig.colorbar(cax, shrink=0.6, pad=0.02)
        cbar.ax.tick_params(labelsize=font_car)
        plt.savefig('./2dResults_notime_pred.eps', dpi=500,  format='pdf',facecolor='w', edgecolor='w',bbox_inches='tight')
        plt.savefig('./2dResults_notime_pred.pdf', dpi=500, format='pdf', facecolor='w', edgecolor='w',
                    bbox_inches='tight')
        plt.show()




        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        #fig, ax = plt.subplots()
        cax = ax.imshow(true_y[:, 0].T/c, cmap='Greys', aspect=0.6, vmin=left, vmax=right)
        ax.set_yticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
        ax.set_yticklabels([0, 0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2])
        ax.set_xticks([20, 120,  220,  320, 420, 520])
        ax.set_xticklabels([-2, -1, 0,  1, 2,  3])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontsize(10) for label in labels]
        zone_and_linked_xychange(ax, 900, 1280, [true_y[:, 0].T/c], range(0,2001), 'right')
        #zone_and_linked_xychange_plt(ax, 900, 1280, [true_x[:, 0].T / c], range(0, 2001), 'right')
        fig.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95)
        ax.set_xlabel("Offset (km)", fontsize=font_label, labelpad=8.5)
        ax.set_ylabel("Time (s)", fontsize=font_label, labelpad=8.5)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        cbar = fig.colorbar(cax, shrink=0.6, pad=0.02)
        cbar.ax.tick_params(labelsize=font_car)
        plt.savefig('./2dResults_notime_true.eps', dpi=500,  format='pdf',facecolor='w', edgecolor='w',bbox_inches='tight')
        plt.savefig('./2dResults_notime_true.pdf', dpi=500, format='pdf', facecolor='w', edgecolor='w',bbox_inches='tight')
        plt.show()

        ############################
        # plot predicted impedance #
        ############################
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        cax = ax.imshow( predicted_x[:, 0].T/c, cmap='Greys', aspect=0.6, vmin=left, vmax=right)
        ax.set_yticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
        ax.set_yticklabels([0, 0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2])
        ax.set_xticks([20, 120,  220,  320, 420, 520])
        ax.set_xticklabels([-2, -1, 0,  1, 2,  3])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontsize(10) for label in labels]
        zone_and_linked_xychange(ax, 900, 1280, [predicted_x[:, 0].T/c], range(0,2001), 'right')
        #zone_and_linked_xychange_plt(ax, 900, 1280, [true_x[:, 0].T / c], range(0, 2001), 'right')
        fig.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95)
        ax.set_xlabel("Offset (km)", fontsize=font_label, labelpad=8.5)
        ax.set_ylabel("Time (s)", fontsize=font_label, labelpad=8.5)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        cbar = fig.colorbar(cax, shrink=0.6, pad=0.02)
        cbar.ax.tick_params(labelsize=font_car)
        plt.savefig('./2dResults_time_pred.eps', dpi=500,  format='pdf',facecolor='w', edgecolor='w',bbox_inches='tight')
        plt.savefig('./2dResults_time_pred.pdf', dpi=500, format='pdf', facecolor='w', edgecolor='w',bbox_inches='tight')
        plt.show()


        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        cax = ax.imshow(true_x[:, 0].T/c, cmap='Greys', aspect=0.6, vmin=left, vmax=right)
        ax.set_yticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
        ax.set_yticklabels([0, 0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2])
        ax.set_xticks([20, 120,  220,  320, 420, 520])
        ax.set_xticklabels([-2, -1, 0,  1, 2,  3])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontsize(10) for label in labels]
        zone_and_linked_xychange(ax, 900, 1280, [true_x[:, 0].T/c], range(0,2001), 'right')
        #zone_and_linked_xychange_plt(ax, 900, 1280, [true_x[:, 0].T / c], range(0, 2001), 'right')
        fig.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95)
        ax.set_xlabel("Offset (km)", fontsize=font_label, labelpad=8.5)
        ax.set_ylabel("Time (s)", fontsize=font_label, labelpad=8.5)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        cbar = fig.colorbar(cax, shrink=0.6, pad=0.02)
        cbar.ax.tick_params(labelsize=font_car)
        plt.savefig('./2dResults_time_true.eps', dpi=500,  format='pdf',facecolor='w', edgecolor='w',bbox_inches='tight')
        plt.savefig('./2dResults_time_true.pdf', dpi=500, format='pdf', facecolor='w', edgecolor='w',bbox_inches='tight')
        plt.show()




        ##########################################
        # plot absolute different true-predicted #
        ##########################################






            
        
        
        
    
    
    
    
