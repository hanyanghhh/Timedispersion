import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import os
from torch.utils import data
from collections import defaultdict
from models_pretraining import Backward_model, Forward_model
import functools
from torch.nn import init
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

random_seed =14 #14 0.7
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
class CRNN(nn.Module):
	"""

	Input:
		X: (n_samples, n_channel, n_length)
		Y: (n_samples)

	Output:
		out: (n_samples)

	Pararmetes:
		n_classes: number of classes

	"""

	def __init__(self, input_dim, out_dim, n_len_seg, n_classes, verbose=False):
		super(CRNN, self).__init__()

		self.n_len_seg = n_len_seg
		self.n_classes = n_classes
		self.input_dim = input_dim
		self.out_dim = out_dim

		#self.device = device
		self.verbose = verbose

		# # (batch, channels, length)
		# self.layer_cnn=nn.Sequential(
		# 	nn.Conv1d(in_channels=self.in_channels,
		# 						 out_channels=self.out_channels,
		# 						 kernel_size=1,
		# 						 stride=1),
		# 	nn.LeakyReLU(0.2,True)
		# )
		# self.cnn = nn.Conv1d(in_channels=self.in_channels,
		# 					 out_channels=self.out_channels,
		# 					 kernel_size=1,
		# 					 stride=1)
		#self.dropout=nn.Dropout(p=0.1)
		# (batch, seq, feature)
		self.rnn1 = nn.GRU(input_size=2001,
						   hidden_size=2001,
						   num_layers=1,
						   bias=False,
						   batch_first=True,
						   bidirectional=True)

		self.layer_linear1 = nn.Sequential(
			nn.Linear(2001*2, 2001, bias=False)
			# 	nn.LeakyReLU(0.2, True)
		)
	def forward(self, x):

		# self.n_channel, self.n_length = x.shape[-2], x.shape[-1]
		# assert (self.n_length % self.n_len_seg == 0), "Input n_length should divided by n_len_seg"
		# self.n_seg = self.n_length // self.n_len_seg
		out = x
		#print(out.shape)
		out = out.permute(0, 1, 2)
		# out = self.layer_cnn(out)
		#out=self.dropout(out)
		#print(out.shape)
		#out = out.permute(0, 2, 1)#(batch_size,seq_len,input_size)
		#print(out.shape)
		#_, (out) = self.rnn(out)#(numlayers,batchsize,hiddensize)
		out, h0 = self.rnn1(out)#(batchsize,seg_len,hiddensize)
		#print("out,h0",out.shape,h0.shape)
		#print("out,h2", out.shape, h2.shape)
		#out = out.permute(0, 2, 1)
		#print("out.shape:{}".format(out.shape))
		out=self.layer_linear1(out)
		#print("out.shape:{}".format(out.shape))
		#print('out.shape:', out.shape)
		#print(out.shape)
		#out = torch.squeeze(out, dim=-1)#(batch_size,hiddensize)
		#print('out.shape:',out.shape)
		#print(out.shape)
		#out = self.dense(out)#(batch_size,n_classes)
		#out=self.layer_linear1(out)
		#out=self.dropout(out)
		#out=self.layer_linear2(out)

		return out

def sort_list_IDs(list_IDs):
    list_nums = [int(i.split(".")[0]) for i in list_IDs]
    list_sort = sorted(enumerate(list_nums), key=lambda x: x[1])
    list_index = [i[0] for i in list_sort]
    list_IDs_new = [list_IDs[i] for i in list_index]
    return list_IDs_new


class Transformer(nn.Module):
    """

    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    Output:
        out: (n_samples)

    Pararmetes:
        n_classes: number of classes

    """

    def __init__(self, input_dim, out_dim, n_len_seg, n_classes, verbose=False):
        super(Transformer, self).__init__()

        self.n_len_seg = n_len_seg
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.out_dim = out_dim

        # self.device = device
        self.verbose = verbose


        self.rnn1 = nn.Transformer(nhead=16, num_encoder_layers=12)

        self.layer_linear1 = nn.Sequential(
            nn.Linear(2001 * 2, 2001, bias=False)
            # 	nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        # self.n_channel, self.n_length = x.shape[-2], x.shape[-1]
        # assert (self.n_length % self.n_len_seg == 0), "Input n_length should divided by n_len_seg"
        # self.n_seg = self.n_length // self.n_len_seg
        out = x
        # print(out.shape)
        out = out.permute(0, 1, 2)

        out, h0 = self.rnn1(out)  # (batchsize,seg_len,hiddensize)

        out = self.layer_linear1(out)

        # out=self.layer_linear2(out)

        return out


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


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def plot_acc_epoch(loss_hist):
    ''' Plot  loss for training and validation data '''
    # Loss
    #plt.subplot(111)
    fig, axins = plt.subplots(1, 1, figsize=(14, 6), dpi=500)
    #y =[pow(10,i) for i in range(-20,1)]
    axins.plot(loss_hist['train'], 'b-', lw=1.5, label='Training loss')
    axins.plot(loss_hist['val'], 'r-', lw=1.5, label='Testing loss')
    #plt.title('Training and validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.legend(loc='upper right')
    #plt.savefig("/data/hany/Losscurve_Marious.pdf", dpi=500)
    plt.savefig('/data/hany/Losscurve_Marious.pdf', format='pdf', dpi=500, bbox_inches='tight')
    plt.savefig("/data/hany/Losscurve_Marious.eps", format='pdf', dpi=500, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    batch_size = 50
    model1_path = ""
    model2_path = ""
    #data_dir = "/data/max/signal_process/dataset/train_100"
    #data_dir_val = "/data/max/signal_process/dataset/test_100"
    #data_dir = "/data/hany/0pinsan_dataset_train/"
    #data_dir_val = "/data/hany/0pinsan_dataset_test/"

    data_dir = "./data/1trace/train5shot/"
    data_dir_val = "./data/1trace/test7shot/"

    #data_dir = "/data/hany/0pinsan_dataset_train_10shot/"
    #data_dir_val = "/data/hany/0pinsan_dataset_test_10shot/"
    # a = np.fromfile("./temp_3.6ms_1_no_time_dispersion.bin", dtype=np.float32)
    # a = a.reshape(737, 2001)
    # x = a[:, :]
    # c = np.fromfile("./temp_3.6ms_1.bin", dtype=np.float32)
    # c = c.reshape(737, 2001)
    # y = c[:, :]
    # q = y
    # x=torch.from_numpy(x)
    # y = torch.from_numpy(y)
    # dataset=TensorDataset(x,y)
    # Get train file list
    data_path = os.path.join(data_dir, "x")
    data_list = os.listdir(data_path)
    list_IDs = sort_list_IDs(data_list)
    dataset = Dataset(root_dir=data_dir, list_IDs=list_IDs,
                      transform=transforms.Compose([
                          Reshape((1, 2001)),
                          ToTensor()
                      ]))

    # =========================#
    # =========val=============#
    # Get valid file list
    data_path = os.path.join(data_dir_val, "x")
    data_list = os.listdir(data_path)
    list_IDs = sort_list_IDs(data_list)
    # Valid dataset
    if True:
        dataset_val = Dataset(root_dir=data_dir_val, list_IDs=list_IDs,
                              transform=transforms.Compose([
                                  Reshape((1, 2001)),
                                  ToTensor()
                              ]))
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False,
                                    num_workers=0, drop_last=False)


        dataset_unlabel = Dataset(root_dir=data_dir_val, list_IDs=list_IDs,
                              transform=transforms.Compose([
                                  Reshape((1, 2001)),
                                  ToTensor()
                              ]), only_load_input=True
                                  )

        dataloader_unlabel = DataLoader(dataset_unlabel, batch_size=batch_size, shuffle=True, num_workers=0)


    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, drop_last=False)  # (1,1,2001)
    dataloaders = {'train': dataloader, 'val': dataloader_val}

    #model = CRNN(1, 1, 1, 2001)
    # model = inverse_model58()
    #model = inverse_model516()
    #model = inverse_model522init()

    #model1 = inverse_model522()
    #model2 = inverse_model619()



    model = CRNN(1, 1, 1, 2001)
    model = model.cuda()

    # print(summary(model, (1, 64)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model1_path != '':
        # ------------------------------------------------------#
        #   权值文件请看README
        # ------------------------------------------------------#
        print('Load weights {}.'.format(model1_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model1_dict = model.state_dict()
        pretrained_dict = torch.load(model1_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model1_dict[k]) == np.shape(v)}
        model1_dict.update(pretrained_dict)
        model.load_state_dict(model1_dict)



    # model = ResNet1D().to(device)
    #criterion = nn.SmoothL1Loss(reduction='sum')
    # criterion = nn.MSELoss()
    #optimizer = torch.optim.RMSprop(listmodel1.parameters(), lr=1e-4)
    optimizer = torch.optim.RMSprop(list(model.parameters()), lr=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8,verbose=True)
    # running_loss=0.0
    metrics = defaultdict(float)
    epochs = int(501)
    print_time = epochs // 10
    #loss_hist = {'train': []}
    loss_hist = {'train': [], 'val': []}

    for epoch in range(epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                criterion = nn.SmoothL1Loss()
            else:
                model.eval()  # Set model to evaluate mode
                criterion = nn.SmoothL1Loss()
            metrics = defaultdict(float)
            loss_epoch = 0.0
            for batch_index, (x, y) in enumerate(tqdm(dataloaders[phase])):  # x.shape=(batch_size,channel,len)
                with torch.set_grad_enabled(phase == 'train'):
                    # print('x.shape',x.shape)
                    # y_pred = model(x).squeeze()
                    x = x.to(device)
                    y = y.to(device)
                    y_pred = model(x)
                    # y=torch.squeeze(y, dim=1)
                    # y_pred=y_pred.reshape(36,1,2001)
                    # print("y_pred:{}".format(y_pred.shape))
                    # print("y:{}".format(y.shape))
                    loss = criterion(y_pred, y)
                    loss_epoch += loss.item()
                    if phase == 'train':
                        optimizer.zero_grad()  # Initialize gradients
                        loss.backward()  # Calculate gradients
                        optimizer.step()  # Update parameters
            if phase == 'train':
                pass

            epoch_loss = (loss_epoch / (batch_index + 1))
            metrics['loss'] = epoch_loss
            loss_hist[phase].append(metrics['loss'])
            tqdm.write('=> Train Epoch:{0}'
                       ' Loss={Loss:.15f}'
                       '\nLearning rate: {1}'
                       '\nphase={2}'
                       '\n-------------------------------------------------------------------------------------------'.format(
                epoch, get_lr(optimizer), phase, Loss=epoch_loss))

            lr_scheduler.step()

        if epoch % 50 == 0:
            # file_params1= "/data/hany/weight/model1_522+619_test{0}_loss{1}.pth".format(epoch, epoch_loss)
            # torch.save(model1.state_dict(), file_params1)
            file_params = "./weight/backwardCRNNmodel_test{0}_loss{1}".format(epoch, epoch_loss)
            torch.save(model, file_params)

            print("model has been stored!")
        if epoch == 499:
            # file_params1= "/data/hany/weight/model1_522+619_test{0}_loss{1}.pth".format(epoch, epoch_loss)
            # torch.save(model1.state_dict(), file_params1)
            file_params = "./weight/backwardCRNNmodel_test{0}_loss{1}".format(epoch, epoch_loss)
            torch.save(model, file_params)

            print("model has been stored!")
    # file_params="LTSM_700.pth"#####3##
    # torch.save(model.state_dict(), file_params)
    print("model has been stored!")
    plot_acc_epoch(loss_hist)
    # fig, ax = plt.subplots(3, figsize=(10, 10), constrained_layout=True)
    # ax[0].plot(x[500].real, label='real')
    # ax[0].plot(x[500].imag, label='imaginary')
    # ax[0].set_title("x")
    # ax[0].legend(loc="upper left")
    # ax[1].plot(a.reshape(737,2001)[500])
    # ax[1].set_title("y")
    # ax[2].plot(y_pred[500].cpu().detach().numpy())
    # ax[2].set_title("y_pred")
    # fig.suptitle("Random signal", fontsize=20)
    # fig.show()
    # plt.show()


# fig, ax = plt.subplots(3, figsize=(10, 10), constrained_layout=True)
# ax[0].plot(x[500].real, label='real')
# ax[0].plot(x[500].imag, label='imaginary')
# ax[0].set_title("x")
# ax[0].legend(loc="upper left")
# ax[1].plot(a.reshape(737,2001)[500])
# ax[1].set_title("y")
# ax[2].plot(y_pred[500].cpu().detach().numpy())
# ax[2].set_title("y_pred")
# fig.suptitle("Random signal", fontsize=20)
# fig.show()
# plt.show()
