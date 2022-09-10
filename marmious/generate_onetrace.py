import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def show():
    #x = np.fromfile("/data/hany/1pinsan_SEAMdataset_train/x/00480.bin", dtype=np.float32)  # 1550x2001 double
    x = np.fromfile("./data/0pinsan_dataset_test/x/00400.bin", dtype=np.float32)  # 1550x2001 double
    ax1 = plt.subplot(1, 1, 1)

    # 第一条线
    plt.plot(x, color='red', label=' no time dispersion', linewidth=1.8)  # 绘制，指定颜色、标签、线宽，标签采用latex格式
    #plt.legend(loc='upper right', frameon=False)  # 绘制图例，指定图例位置
    plt.show()



    #y = np.fromfile("/data/hany/1pinsan_SEAMdataset_train/y/00480.bin", dtype=np.float32)  # 1550x2001 double
    y = np.fromfile("./data/0pinsan_dataset_test/y/00400.bin", dtype=np.float32)  # 1550x2001 double

    # 第一条线
    plt.plot(y, color='black', label=' no time dispersion', linewidth=1.8)  # 绘制，指定颜色、标签、线宽，标签采用latex格式
    #plt.legend(loc='upper right', frameon=False)  # 绘制图例，指定图例位置
    plt.show()

def originalshow():
    k=500
    #x = np.fromfile("/data/hany/SEAMrecord/SEAM_notime_test/record_shot61_2ms2001x325.bin", dtype=np.float32).reshape(325,2001)[k,::]  # 1550x2001 double
    x = np.fromfile("./data/0pinsan_test_notime/record_0.2ms_no_time_dispersion_shot102_2001x550.bin", dtype=np.float32).reshape(550,2001)[k,::]  # 1550x2001 double
    ax1 = plt.subplot(1, 1, 1)

    # 第一条线
    plt.plot(x, color='red', label=' no time dispersion', linewidth=0.8)  # 绘制，指定颜色、标签、线宽，标签采用latex格式
    plt.legend(loc='upper right', frameon=False)  # 绘制图例，指定图例位置



    #y = np.fromfile("/data/hany/SEAMrecord/SEAM_time_test/record_shot61_time_dispersion2001x325.bin",dtype=np.float32).reshape(325,2001)[k,::]  # 1550x2001 double
    y = np.fromfile("/data/hany/0pinsan_test_time/record_1.2ms_shot102_2001x550.bin", dtype=np.float32).reshape(550, 2001)[k, ::]  # 1550x2001 double

    # 第一条线
    plt.plot(y, color='black', label='time dispersion', linewidth=0.8)  # 绘制，指定颜色、标签、线宽，标签采用latex格式
    plt.legend(loc='upper right', frameon=False)  # 绘制图例，指定图例位置
    plt.show()

if __name__ == '__main__':
    #show()
    #originalshow()

    data_dir = "./data/2D/0pinsan_train_time"
    #data_dir_val = "./data/2D/0pinsan_test_time"

    #num_trainfile = glob.glob(r'./data/2D/0pinsan_test_time/*.bin')
    #num_testfile = glob.glob(r'/data/hany/0pinsan_test_x/*.bin')
    
    list1 = [x for x in sorted(os.listdir(data_dir))]
    print(len(list1))
    print(list1)
    #list2 = [x for x in sorted(os.listdir(data_dir_val))]


    for i in range(0, len(list1)):
        for k in range(0, 550, 1):
            f1 = np.fromfile(os.path.join(data_dir, list1[i]), dtype=np.float32)
            f1 = f1.reshape(550, 2001)
            data = f1[k, ::]
            #binPath = os.path.join('./data/1trace/0pinsan_dataset_train/x', '%05d.bin' % (0+k+i*550))
            binPath = os.path.join('./data/1trace/0pinsan_dataset_test/x', '%05d.bin' % (0+k+i*550))
            data.tofile(binPath)
