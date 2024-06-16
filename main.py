#!/usr/bin/env python
import argparse
import copy
import logging
import os
import shutil
import time
import warnings

import numpy as np
import torchvision

from flcore.servers.serveramp import FedAMP
from flcore.servers.serverapfl import APFL
from flcore.servers.serverapple import APPLE
from flcore.servers.serveravg import FedAvg
from flcore.servers.serverbabu import FedBABU
from flcore.servers.serverbn import FedBN
from flcore.servers.serverdistill import FedDistill
from flcore.servers.serverditto import Ditto
from flcore.servers.serverdyn import FedDyn
from flcore.servers.serverfomo import FedFomo
from flcore.servers.servergen import FedGen
from flcore.servers.serverlocal import Local
from flcore.servers.servermoon import MOON
from flcore.servers.servermtl import FedMTL
from flcore.servers.serverpFedMe import pFedMe
from flcore.servers.serverper import FedPer
from flcore.servers.serverperavg import PerAvg
from flcore.servers.serverphp import FedPHP
from flcore.servers.serverproto import FedProto
from flcore.servers.serverprox import FedProx
from flcore.servers.serverrep import FedRep
from flcore.servers.serverrod import FedROD
from flcore.servers.serverscaffold import SCAFFOLD
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.bilstm import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.models import *
from flcore.trainmodel.resnet import *
# from system.utils.data_utils import dataset_path
from utils.mem_utils import MemReporter
from utils.result_utils import average_data

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)

# hyper-params for Text tasks
vocab_size = 98635
max_len = 200
emb_dim = 32


def run(args):
    time_list = []
    reporter = MemReporter()
    model_str = args.model
    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "mlr":  # convex
            if "mnist" in args.dataset:
                args.model = Mclr_Logistic(1 * 28 * 28, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = Mclr_Logistic(3 * 32 * 32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

        elif model_str == "cnn":  # non-convex
            if "mnist" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "omniglot" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
                # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "dnn":  # non-convex
            if "mnist" in args.dataset:
                args.model = DNN(1 * 28 * 28, 100, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = DNN(3 * 32 * 32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)

        elif model_str == "resnet":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
            if "fmnist" in args.dataset:
                args.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False).to(args.device)
            # args.model = MyMixModel(model)
            # args.model = resnet(pretrained=False, num_classes=args.num_classes).to(args.device)
            # args.model = torchvision.models.resnet18(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

            # args.model = resnet18(num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)

        elif model_str == "alexnet":
            args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)

            # args.model = alexnet(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "googlenet":
            args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False,
                                                      num_classes=args.num_classes).to(args.device)

            # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "mobilenet_v2":
            args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)

            # args.model = mobilenet_v2(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "lstm":
            args.model = LSTMNet(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(
                args.device)

        elif model_str == "bilstm":
            args.model = BiLSTM_TextClassification(input_size=vocab_size, hidden_size=emb_dim,
                                                   output_size=args.num_classes,
                                                   num_layers=1, embedding_dropout=0, lstm_dropout=0,
                                                   attention_dropout=0,
                                                   embedding_length=emb_dim).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(
                args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(hidden_dim=emb_dim, max_len=max_len, vocab_size=vocab_size,
                                 num_classes=args.num_classes).to(args.device)

        elif model_str == "Transformer":
            args.model = TransformerModel(ntoken=vocab_size, d_model=emb_dim, nhead=2, d_hid=emb_dim, nlayers=2,
                                          num_classes=args.num_classes).to(args.device)

        elif model_str == "AmazonMLP":
            args.model = AmazonMLP().to(args.device)

        elif model_str == "harcnn":
            if args.dataset == 'har':
                args.model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9),
                                    pool_kernel_size=(1, 2)).to(args.device)
            elif args.dataset == 'pamap':
                args.model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9),
                                    pool_kernel_size=(1, 2)).to(args.device)

        else:
            raise NotImplementedError

        print(args.model)

        # select algorithm
        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            if args.pretrained:
                checkpoint_path = r'D:\PycharmProjects\PFL-Non-IID\results\checkpoint_fed.pth.tar'
                checkpoint = torch.load(checkpoint_path)
                args.model.load_state_dict(checkpoint['state_dict'])
            # if args.privacy and args.dp_sigma < 10:
            #     args.model = ModuleValidator.fix(args.model)
            server = FedAvg(args, i)

        elif args.algorithm == "Local":
            server = Local(args, i)

        elif args.algorithm == "FedMTL":
            server = FedMTL(args, i)

        elif args.algorithm == "PerAvg":
            server = PerAvg(args, i)

        elif args.algorithm == "pFedMe":
            server = pFedMe(args, i)

        elif args.algorithm == "FedProx":
            server = FedProx(args, i)

        elif args.algorithm == "FedFomo":
            server = FedFomo(args, i)

        elif args.algorithm == "FedAMP":
            server = FedAMP(args, i)

        elif args.algorithm == "APFL":
            server = APFL(args, i)

        elif args.algorithm == "FedPer":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPer(args, i)

        elif args.algorithm == "Ditto":
            server = Ditto(args, i)

        elif args.algorithm == "FedRep":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedRep(args, i)

        elif args.algorithm == "FedPHP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPHP(args, i)

        elif args.algorithm == "FedBN":
            server = FedBN(args, i)

        elif args.algorithm == "FedROD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedROD(args, i)

        elif args.algorithm == "FedProto":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedProto(args, i)

        elif args.algorithm == "FedDyn":
            server = FedDyn(args, i)

        elif args.algorithm == "MOON":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = MOON(args, i)

        elif args.algorithm == "FedBABU":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedBABU(args, i)

        elif args.algorithm == "APPLE":
            server = APPLE(args, i)

        elif args.algorithm == "FedGen":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGen(args, i)

        elif args.algorithm == "SCAFFOLD":
            server = SCAFFOLD(args, i)

        elif args.algorithm == "FedDistill":
            server = FedDistill(args, i)

        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times, time=server.save_time)

    print("All done!")

    reporter.report()



if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="实验目标，是测试还是训练")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"], help="CPU还是CUDA")
    parser.add_argument('-did', "--device_id", type=str, default="0", help="设备id，cuda0就完事")
    parser.add_argument('-data', "--dataset", type=str, default="mnist", help="用啥数据集")
    parser.add_argument('-nb', "--num_classes", type=int, default=10, help="标签类数")
    parser.add_argument('-m', "--model", type=str, default="cnn", help="模型")
    parser.add_argument('-lbs', "--batch_size", type=int, default=16, help="批量")
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="本地学习率")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False, help="本地延迟更新")
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99, help="本地延迟更新率")
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000, help="全局训练轮数")
    parser.add_argument('-ls', "--local_steps", type=int, default=1, help="本地训练轮数？")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg", help="算法")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="客户端采样率")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="随机客户端采样率Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=2,
                        help="客户端总数")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="预训练轮数")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="训练轮数")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="评价轮数")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items', help="保存路径")
    parser.add_argument('-ab', "--auto_break", type=bool, default=False, help="是否自动break")
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2, help="本地迭代轮数")
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="客户端掉队占比")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="慢训练客户端占比")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="慢通信客户端占比")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="慢速客户端最大训练时间")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP
    parser.add_argument('-bt', "--beta", type=float, default=0.0,
                        help="Average moving parameter for pFedMe, Second learning rate of Per-FedAvg, \
                        or L1 regularization weight of FedTransfer")
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="正则化权重")
    parser.add_argument('-mu', "--mu", type=float, default=0,
                        help="Proximal rate for FedProx")
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0,
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # APFL
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_steps", type=int, default=1)
    # MOON
    parser.add_argument('-ta', "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument('-fts', "--fine_tuning_steps", type=int, default=1)
    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
    # SCAFFOLD
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)


    # --defense my
    # mine
    parser.add_argument('-dlg', "--dlg_eval", type=str, choices=["None", "DLG", "IDLG"], default="None",
                        help="是否dlg攻击")
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100, help="每隔多少轮一次dlg攻击")
    parser.add_argument('-dlgo', "--dlg_optimizer", type=str, choices=["LBFGS", "adam"], default="LBFGS",
                        help="dlg优化器")
    # parser.add_argument('-mix', "--mix_on", choices=[None, 'x', 'rep'], default=None)
    # parser.add_argument('-mixp', "--mix_alpha", type=float, default=5)
    parser.add_argument('-df', "--defense", type=str, default=None,
                        choices={"None", "gaussian", "laplace", "dptest", "mix_x", "manifold", "patch", "my", "my_hard", 'soteria'}, help="是否防御")
    parser.add_argument('-dfp', "--defense_param", type=float, default=5.0, help="防御参数，隐私预算或剪枝率")
    parser.add_argument('-aby', "--aby_mix", choices=["0", "1", "-1"], default=0, help="混合不同标签样本")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0, help="dp防御参数，高斯噪声的sigma")
    parser.add_argument('--use_data', type=str, default=None, help="数据集情况")
    parser.add_argument('--privacy', action="store_true", default=False, help="dp")
    parser.add_argument('--use_sot', action="store_true", default=False, help="使用soteria")
    parser.add_argument('--pretrained', action="store_true", default=False, help="用训练好的模型")
    parser.add_argument('--initnorm', type=float, default=0.5, help="初始自适应裁剪阈值")



    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_steps))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("防御策略为: {}".format(args.defense))
    if args.defense is not None:
        print("防御参数: {}".format(args.defense_param))
        print("特殊混合: {}".format(args.aby_mix))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("DLG attack evaluate: {}".format(args.dlg_eval))
    if args.dlg_eval:
        print("DLG attack evaluate round gap: {}".format(args.dlg_gap))
    print("=" * 50)

    # if args.dataset == "mnist" or args.dataset == "fmnist":
    #     generate_mnist('../dataset/mnist/', args.num_clients, 10, args.niid)
    # elif args.dataset == "Cifar10" or args.dataset == "Cifar100":
    #     generate_cifar10('../dataset/Cifar10/', args.num_clients, 10, args.niid)
    # else:
    #     generate_synthetic('../dataset/synthetic/', args.num_clients, 10, args.niid)

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA],
    #     profile_memory=True, 
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    #     ) as prof:
    # with torch.autograd.profiler.profile(profile_memory=True) as prof:
    run(args)

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")
