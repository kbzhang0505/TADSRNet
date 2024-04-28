"""
PyTorch 1.6 implementation of the following paper:

 Usage:
    Start tensorboard:
    ```bash
    tensorboard --logdir=logger --port=6006
    ```
    Run the main.py:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python main.py --exp_id=0   ```

 

"""

import numpy as np
import random
from scipy import stats
from argparse import ArgumentParser
import yaml
import os
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics.metric import Metric
from Dataset import IQADataset
from ignite.contrib.handlers import ProgressBar
import time
from tensorboardX import SummaryWriter


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def loss_fn(y_pred, y):
    return F.l1_loss(y_pred, y)


def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


# ==============性能评估=================
class IQAPerformance(Metric):
    """
    Evaluation of IQA methods using SROCC, KROCC, PLCC, RMSE, MAE, OR.

    `update` must receive output of the form (y_pred, y).
    """

    def reset(self):
        self._y_pred = []
        self._y = []

    def update(self, output):
        y_pred, y = output
        self._y.append(y[0].item())
        self._y_pred.append(torch.mean(y_pred).item())

    def compute(self):
        sq = np.reshape(np.asarray(self._y), (-1,))

        q = np.reshape(np.asarray(self._y_pred), (-1,))

        srocc = stats.spearmanr(sq, q)[0]
        krocc = stats.stats.kendalltau(sq, q)[0]
        plcc = stats.pearsonr(sq, q)[0]
        rmse = np.sqrt(((sq - q) ** 2).mean())
        mae = np.abs((sq - q)).mean()

        return srocc, krocc, plcc, rmse, mae


def TAM(feat, conv_CW, conv_CH, conv_HW):
    output = []
    size_b = feat.shape[0]
    for each in range(size_b):
        CW_MaxPool = torch.max(feat[each], dim=1).values
        CW_AvgPool = torch.mean(feat[each], dim=1)
        CW = torch.stack([CW_MaxPool, CW_AvgPool], dim=0).unsqueeze(0)
        CW = conv_CW(CW).squeeze(0).permute(1, 0, 2)
        CW = feat[each] * CW

        CH_MaxPool = torch.max(feat[each], dim=2).values
        CH_AvgPool = torch.mean(feat[each], dim=2)
        CH = torch.stack([CH_MaxPool, CH_AvgPool], dim=0).unsqueeze(0)
        CH = conv_CH(CH).squeeze(0).permute(1, 2, 0)
        CH = feat[each] * CH

        HW_MaxPool = torch.max(feat[each], dim=0).values
        HW_AvgPool = torch.mean(feat[each], dim=0)
        HW = torch.stack([HW_MaxPool, HW_AvgPool], dim=0).unsqueeze(0)
        HW = conv_HW(HW).squeeze(0)
        HW = feat[each] * HW

        output.append(CW + CH + HW)
    output = torch.stack(output, dim=0)
    return output


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input


class TADSRNet(nn.Module):
    def __init__(self):
        super(TADSRNet, self).__init__()
        self.feature_low = nn.Conv2d(3, 25, 3)
        self.feature_c1 = nn.Conv2d(25, 50, 5)
        self.feature_c2 = nn.Conv2d(25, 50, 7)

        self.dscm_1_1 = nn.Conv2d(25, 25, 3)
        self.dscm_1_2 = nn.Conv2d(25, 25, 5)
        self.dscm_fanconv_1 = nn.ConvTranspose2d(25, 25, 3, 1, 0)

        self.dscm_2_1 = nn.Conv2d(50, 50, 3)
        self.dscm_2_2 = nn.Conv2d(50, 50, 5)
        self.dscm_fanconv_2 = nn.ConvTranspose2d(50, 50, 3, 1, 0)

        self.dscm_3_1 = nn.Conv2d(150, 150, 3)
        self.dscm_3_2 = nn.Conv2d(150, 150, 5)
        self.dscm_fanconv_3 = nn.ConvTranspose2d(150, 150, 3, 1, 0)

        self.maxpool = nn.AdaptiveMaxPool2d(1)

        self.linear = nn.Sequential(nn.Linear(350, 800), nn.ReLU(), nn.Dropout(),
                                    nn.Linear(800, 800), nn.ReLU(),
                                    nn.Linear(800, 1))

        self.convCW = BaseConv(2, 1, 7, activation=nn.Sigmoid())
        self.convCH = BaseConv(2, 1, 7, activation=nn.Sigmoid())
        self.convHW = BaseConv(2, 1, 7, activation=nn.Sigmoid())

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))

        x_low = self.feature_low(x)
        x_c1 = self.feature_c1(x_low)
        x_c2 = self.feature_c2(x_low)

        x_obvious = TAM(x_low, self.convCW, self.convCH, self.convHW)

        x_1_1 = self.dscm_1_1(x_obvious)
        x_1_2 = self.dscm_1_2(x_obvious)
        x_1_2 = self.dscm_fanconv_1(x_1_2)
        x_1 = torch.cat((x_1_1, x_1_2), 1)

        x_2_1 = self.dscm_2_1(x_1)
        x_2_2 = self.dscm_2_2(x_1)
        x_2_2 = self.dscm_fanconv_2(x_2_2)
        x_2 = torch.cat((x_2_1, x_2_2, x_c1), 1)

        x_3_1 = self.dscm_3_1(x_2)
        x_3_2 = self.dscm_3_2(x_2)
        x_3_2 = self.dscm_fanconv_3(x_3_2)
        x_3 = torch.cat((x_3_1, x_3_2, x_c2), 1)

        pool = self.maxpool(x_3)
        pool = pool.squeeze(3).squeeze(2)
        q = self.linear(pool)
        return q


# ==================读取数据库里的图片和标签===================
def get_data_loaders(config, train_batch_size, exp_id=0):
    train_dataset = IQADataset(config, exp_id, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=True,
                                               num_workers=0)

    val_dataset = IQADataset(config, exp_id, 'val')
    val_loader = torch.utils.data.DataLoader(val_dataset)

    if config['test_ratio']:
        test_dataset = IQADataset(config, exp_id, 'test')
        test_loader = torch.utils.data.DataLoader(test_dataset)
        return train_loader, val_loader, test_loader
    return train_loader, val_loader


# =================开始训练=======================
def run(train_batch_size, epochs, lr, weight_decay, config, exp_id, disable_gpu=False):
    # 初始化TensorboardX
    writer = SummaryWriter()
    # ======将数据读入DATALODER中=========
    if config['test_ratio']:  # 0.2
        train_loader, val_loader, test_loader = get_data_loaders(config, train_batch_size, exp_id)
    else:
        train_loader, val_loader = get_data_loaders(config, train_batch_size, exp_id)

    # ===========设置GPU====================
    device = torch.device("cuda" if not disable_gpu and torch.cuda.is_available() else "cpu")
    # ==========读入模型===========
    model = TADSRNet()
    model = model.to(device)
    print(model)
    # ===========优化器设置==============
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # ===========训练器设置=============
    global best_criterion
    global path_checkpoint
    best_criterion = -1  # SROCC>=-1
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'IQA_performance': IQAPerformance()},
                                            device=device)
    # ===========进度条=============
    pbar = ProgressBar()
    pbar.attach(trainer, ['loss'])
    # =========计算模型参数量=========
    params = params_count(model)
    print("该模型的参数量：{}".format(params))

    global best_epoch

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        SROCC, KROCC, PLCC, RMSE, MAE = metrics['IQA_performance']
        print("Validation Results - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f}%"
              .format(engine.state.epoch, SROCC, KROCC, PLCC, RMSE, MAE))
        writer.add_scalar("validation/SROCC", SROCC, engine.state.epoch)
        writer.add_scalar("validation/KROCC", KROCC, engine.state.epoch)
        writer.add_scalar("validation/PLCC", PLCC, engine.state.epoch)
        writer.add_scalar("validation/RMSE", RMSE, engine.state.epoch)
        writer.add_scalar("validation/MAE", MAE, engine.state.epoch)
        global best_criterion
        global best_epoch
        if SROCC > best_criterion:
            best_criterion = SROCC
            best_epoch = engine.state.epoch
            path_checkpoint = "_best_epoch.pkl"
            torch.save(model.state_dict(), path_checkpoint)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_testing_results(engine):
        if config["test_ratio"] > 0 and config['test_during_training']:
            evaluator.run(test_loader)
            writer.add_scalar("test_pre", engine.state.output, engine.state.epoch)
            metrics = evaluator.state.metrics
            SROCC, KROCC, PLCC, RMSE, MAE = metrics['IQA_performance']
            print("Testing Results    - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f}%"
                  .format(engine.state.epoch, SROCC, KROCC, PLCC, RMSE, MAE))
            writer.add_scalar("testing/SROCC", SROCC, engine.state.epoch)
            writer.add_scalar("testing/KROCC", KROCC, engine.state.epoch)
            writer.add_scalar("testing/PLCC", PLCC, engine.state.epoch)
            writer.add_scalar("testing/RMSE", RMSE, engine.state.epoch)
            writer.add_scalar("testing/MAE", MAE, engine.state.epoch)

    @trainer.on(Events.COMPLETED)
    def final_testing_results(engine):
        if config["test_ratio"]:
            model.load_state_dict(torch.load('./_best_epoch.pkl'))
            evaluator.run(test_loader)
            metrics = evaluator.state.metrics
            SROCC, KROCC, PLCC, RMSE, MAE = metrics['IQA_performance']
            global best_epoch
            print("Final Test Results - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f}%"
                  .format(best_epoch, SROCC, KROCC, PLCC, RMSE, MAE))
            path_test = "final_test_best_epoch"
            np.save(path_test, (SROCC, KROCC, PLCC, RMSE, MAE))

    trainer.run(train_loader, max_epochs=epochs)

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch TADSRNet')
    parser.add_argument("--seed", type=int, default=96675202)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: QADS:0.0001 MA：0.0005)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='config file path (default: config.yaml)')
    parser.add_argument('--exp_id', default='0', type=str,
                        help='exp id (default: 0)')
    parser.add_argument('--database', default='QADS', type=str,  
                        help='database name (default: QADS)')
    parser.add_argument('--model', default='TADSRNet', type=str,
                        help='model name (default: TADSRNet)')
    parser.add_argument('--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument("--log_dir", type=str, default="logger",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')

    args = parser.parse_args(args=[])

    torch.utils.backcompat.broadcast_warning.enabled = True

    with open(args.config, mode='r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print('exp id: ' + args.exp_id)
    print('database: ' + args.database)
    print('model: ' + args.model)
    config.update(config[args.database])
    config.update(config[args.model])

    run(args.batch_size, args.epochs, args.lr, args.weight_decay, config, args.exp_id,
        args.disable_gpu)

