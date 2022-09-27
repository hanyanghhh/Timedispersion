# Timedispersion
## Environment

* Pytorch 1.9.0+cu111
* Python 3.7.10

## Data preparing

* First, download a dataset, e.g. 1trace

If you want to change the dataset, you can pass those to the command line, such as:
```bash
$ python3 generate_onetrace.py  \
    --data_dir=./data/2D/dataset_time \
    --binPath=./data/1trace/train/x
```
Set `file x` for data with time dispersion, `file y` for data without time dispersion


## Pre-Training

```bash
cd marm
$ torch python time2notime_sei.py
```

## Inference
After exporting model, you can use it for inference. For example:

```bash
$ torch python inference.py --model1 ./weight/backwardmodel_test499_loss0.0003094713379554874 \
                     --model2 ./weight/forwardmodel_test499_loss0.0003094713379554874 \
                     --x ./data/1trace/test/x \
                     --y ./data/1trace/test/y \
                     --plt.savefig('./Results_notimepred500.pdf')
```
```bash
$ torch python Datashow_inference.py --Gxy ./weight/backwardmodel_test499_loss0.0003094713379554874 \
                     --Gyx ./weight/forwardmodel_test499_loss0.0003094713379554874 \
                     --f1 ./data/1trace/test/x \
                     --f2 ./data/1trace/test/y \
                     --plt.savefig('./2d_Results_notime_pred.pdf')
```


## Training

```bash
cd seam
$ torch python time2notime_seiSEAM_pretraing_init0.py
```

## Inference
After exporting model, you can use it for inference. For example:

```bash
$ torch python SEAM_inference.py --model1 ./weight/backwardmodel_test499_loss0.0003094713379554874 \
                     --model2 ./weight/forwardmodel_test499_loss0.0003094713379554874 \
                     --x ./data/1trace/test/x \
                     --y ./data/1trace/test/y \
                     --plt.savefig('./Results_notimepred50.pdf')
```
```bash
$ torch python SEAMDatashow_inference.py --model1 ./pre_weight/backwardmodel_test499_loss0.0003094713379554874 \
                     --model2 ./pre_weight/forwardmodel_test499_loss0.0003094713379554874 \
                     --Gxy ./weight/init0model1_test250\
                     --Gyx ./weight/init0model2_test250\
                     --f1 ./data/1trace/test/x \
                     --f2 ./data/1trace/test/y \
                     --plt.savefig('./2d_Results_notime_pred.pdf')
```
