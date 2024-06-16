import argparse
import copy
import os
import random
from datetime import time

import h5py
import numpy
import torchvision
import torch
import torch.nn as nn
from scipy.stats import ks_2samp, normaltest, shapiro, kstest
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import numpy as np

from system.utils.privacy import to_one_hot
from trainmodel.models import BaseHeadSplit
from utils import privacy

# 高斯机制实现函数
# def add_noise_using_gaussian_mechanism(gradient, epsilon, delta=0.0):
#     if delta == 0.0:
#         delta = 1 / gradient.shape[0]
#     sigma = math.sqrt(2 * math.log(1.25/delta)) / epsilon
#     noise = np.random.normal(0, sigma, gradient.shape)
#     return gradient + noise
#
# # 示例
# # 假设我们有一个梯度向量 g
# g = torch.randn(2, 3, 4)
# epsilon = 5
# delta = 1e-5

# 对梯度应用高斯机制添加噪声
# noisy_g = add_noise_using_gaussian_mechanism(g, epsilon, delta)
#
# print("原始梯度: ", g)
# print("加入高斯噪声后的梯度: ", noisy_g)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# test4noise(device=device)
import torch.nn as nn
import torch
import math
import matplotlib.pyplot as plt

def look_x_and_pro():
    x = range(98)  # 假设x的范围是0到100
    dim = 100
    y = [math.comb(dim, round((1 + i / 100) / 2 * dim)) for i in x]  # 根据概率公式计算猜中概率

    plt.plot(x, y)  # 绘制曲线
    plt.xlabel('x')  # 设置x轴标签
    plt.ylabel(r'猜中概率')  # 设置y轴标签
    plt.title(r'猜中概率与x的关系')  # 设置图表标题

    plt.show()  # 显示图表
    print(min(y))

def look_model():
    input = torch.ones((64, 3, 32, 32))
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)

    writer = SummaryWriter("../logs_model")
    writer.add_graph(model, input)
    writer.close()


def test4noise(grad=None, device=None):
    if grad is None:
        grad = torch.randn(2, 3, 4)

    grad = torch.Tensor(grad)
    print("加噪前:{}".format(grad))

    grad += privacy.get_noise(grad, defense_param=0.1, defense="gaussian")

    print("加噪后:{}".format(grad))


def generate_mask(mask, targets):
    unique_targets = torch.unique(targets)  # 获取 targets 中的不重复元素
    # 创建一个空的 mask，形状为 (64, 512)
    output_mask = torch.zeros_like(mask)
    for target in unique_targets:
        # 找到 targets 中等于当前不重复元素的位置
        target_indices = torch.where(targets == target)[0]
        # 将对应位置的 mask 取并集
        output_mask[target_indices] = torch.any(mask[target_indices], dim=0).int()
    return output_mask


#


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

def pred():
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
                        choices={"None", "gaussian", "laplace", "dptest", "mix_x", "manifold", "patch", "my",
                                 "my_hard"}, help="是否防御")
    parser.add_argument('-dfp', "--defense_param", type=float, default=5.0, help="防御参数，隐私预算或剪枝率")
    parser.add_argument('-aby', "--aby_mix", choices=["0", "1", "-1"], default=0, help="混合不同标签样本")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0, help="dp防御参数，高斯噪声的sigma")
    parser.add_argument('--use_data', type=str, default=None, help="数据集情况")
    parser.add_argument('--privacy', action="store_true", default=False, help="dp")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # 设置随机数种子
    # setup_seed(20)
# 预处理数据以及训练模型

def myvar(data, axis=0):
    return np.mean(np.square(data - np.mean(data, axis=axis)), axis=axis)

def KSGeneral():
    setup_seed(1213)
    import itertools
    # 定义ResNet模型结构
    class ResNet(nn.Module):
        def __init__(self):
            super(ResNet, self).__init__()
            self.resnet = torchvision.models.resnet18(pretrained=False)
            self.fc = nn.Linear(1000, 10)  # 假设输出类别为10

        def forward(self, x):
            x = self.resnet(x)
            x = self.fc(x)
            return x

    args = pred()
    args.model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    args.head = copy.deepcopy(args.model.fc)
    args.model.fc = nn.Identity()
    args.model = BaseHeadSplit(args.model, args.head)
    # 加载训练好的模型权重
    model = args.model
    # checkpoint_path = r'D:\PycharmProjects\PFL-Non-IID\results\checkpoint_fed.pth.tar'
    # checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint['state_dict'])

    # 加载CIFAR-10数据集
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_dataset = torchvision.datasets.CIFAR10(root=r'D:\PycharmProjects\PFL-Non-IID\dataset\cifar10\rawdata', train=True, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 生成特征并进行分类统计
    class_features = [[] for _ in range(10)]  # 存储每个类别的特征
    with torch.no_grad():
        model.eval()

        # for images, labels in list(itertools.islice(test_loader, 200)):
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # flag = predicted == labels
            feature = model.feature.squeeze().cpu().numpy()
            class_features[labels].append(feature)

    # 计算概率分布并进行KS检验
    # total_samples = len(test_dataset)
    # class_distributions = []
    mean_per_dimension = []
    variance_per_dimension = []
    for class_idx in range(10):
        features = np.vstack(class_features[class_idx])  # (1w, 512)
        ###
        #   File "D:\PycharmProjects\PFL-Non-IID\system\test.py", line 253, in KSTest
        #     features = np.vstack(class_features[class_idx])
        #   File "<__array_function__ internals>", line 180, in vstack
        #   File "D:\ProgramData\Anaconda3\envs\demo9\lib\site-packages\numpy\core\shape_base.py", line 282, in vstack
        #     return _nx.concatenate(arrs, 0)
        #   File "<__array_function__ internals>", line 180, in concatenate
        # ValueError: need at least one array to concatenate
        #
        # ###
        # feature_count = features.shape[0]
        mean_per_dimension.append(np.mean(features, axis=0))  # (512,)
        variance_per_dimension.append(myvar(features, axis=0))

        # probability_distribution, _ = np.histogram(features, bins='auto', density=True)
        # probability_distribution /= feature_count
        # probability_distributions.append(probability_distribution)
    features_array = np.concatenate(class_features)  # (5w, 512)
    total_variance = myvar(features_array, axis=0)  # (512,)

    # 进行KS检验
    # ks_statistic, p_value = ks_2samp(probability_distributions[0], probability_distributions[1])
    #
    # print("KS statistic:", ks_statistic)
    # print("p-value:", p_value)

    # 创建.h5文件并保存全样本概率分布
    # with h5py.File('../results/probability/probability_distribution_all_samples.h5', 'w') as f:
    #     # for i, arr in enumerate(probability_distributions):
    #     #     f.create_dataset(f'data_{i}', data=arr)
    #     f.create_dataset('probability_distribution', data=probability_distributions)

    # 创建.h5文件并保存分类概率分布
    with h5py.File('../results/probability/probability_distribution_per_class.h5', 'w') as f:
        for class_idx, probability_distribution in enumerate(mean_per_dimension):
            f.create_dataset(f'mean_class_{class_idx}', data=probability_distribution)
        for class_idx, probability_distribution in enumerate(variance_per_dimension):
            f.create_dataset(f'var_class_{class_idx}', data=probability_distribution)
        f.create_dataset(f'global_var', data=total_variance)

def KSTest():
    setup_seed(1213)
    args = pred()
    args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)
    args.head = copy.deepcopy(args.model.fc)
    args.model.fc = nn.Identity()
    args.model = BaseHeadSplit(args.model, args.head)
    # 加载训练好的模型权重
    model = args.model
    # checkpoint_path = r'D:\PycharmProjects\PFL-Non-IID\results\checkpoint_fed.pth.tar'
    # checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint['state_dict'])
    model.to(args.device)

    # 加载CIFAR-10数据集
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_dataset = torchvision.datasets.CIFAR10(root=r'D:\PycharmProjects\PFL-Non-IID\dataset\cifar10\rawdata',
                                                train=True, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    random.seed()
    args.idx = random.randint(1, len(test_loader.dataset))
    main_x, main_y = test_loader.dataset[args.idx]
    imgs, targets = [], []
    imgs.append(main_x.to(args.device))
    targets.append(torch.tensor(main_y, device=args.device))
    temp = args.idx + 1
    # for img_num in range(100):  # 选10个样本，每个混合100个特征出来，KS检测1000次
        # 采样用于混合的非攻击数据，img_mix_num代表本次有多少个样本进行聚合
    for i in range(100):
        # 向后寻找符合条件的样本准备混合，2000个保证能够找到
        mix_x, mix_y = test_loader.dataset[temp + i]
        mix_x, mix_y = torch.tensor(mix_x), torch.tensor(mix_y)
        # if (mix_y == label and args.mix_ab_label == 0) or (mix_y != label and args.mix_ab_label == 1):
        #     print("选择第{}个图片为副样本".format(temp + i))
        to_mix_img = mix_x.to(args.device)
        imgs.append(to_mix_img)
        targets.append(mix_y.to(args.device))

    imgs = torch.stack(imgs, dim=0)
    targets = to_one_hot(torch.stack(targets, dim=0), 10)
    with torch.no_grad():
        model.eval()
        features, features_mix = model.KS_forward(x=imgs, target=targets)
        # _ = model.forward()
        # features = model.base(x=imgs)

    # h5分布
    with h5py.File('../results/probability/probability_distribution_per_class.h5', 'r') as f:
        # 创建空数组
        mean_distributions = []
        var_distributions = []
        global_var_distributions = []

        # 遍历.h5文件中的每个数据集
        for key in f.keys():
            if "mean_class" in key:
                mean_distributions.append(f[key][:])
            elif "var_class" in key:
                var_distributions.append(f[key][:])
            elif "global" in key:
                global_var_distributions.append(f[key][:])

    # KS检验
    # 全部numpy
    features_mix = features_mix.cpu().numpy()
    features = features.cpu().numpy()
    for i in range(100):
        main_y = main_y
        mix_y = torch.argmax(targets[i + 1]).cpu().numpy()
        # if main_y != mix_y:
        #     ks_mean_main, p_mean_main = ks_2samp(features[i + 1], mean_distributions[main_y])
        #     ks_var_main, p_var_main = ks_2samp(features[i + 1], var_distributions[main_y])
        #     ks_global_main, p_global_main = ks_2samp(features[i + 1], global_var_distributions[main_y])
        ks_statistic, ks_pvalue = kstest(features_mix[i + 1], 'norm', args=(mean_distributions[main_y], var_distributions[main_y] + 0.000001))
        # p值小的离谱
        ks_mean, p_mean = ks_2samp(features_mix[i + 1], mean_distributions[main_y])
        ks_var, p_var = ks_2samp(features_mix[i + 1], var_distributions[main_y])
        ks_global, p_global = ks_2samp(features_mix[i + 1], global_var_distributions[0])
        print("KS statistic:", ks_mean)
        print("p-value:", p_mean)

        ###
        # p_value_main =
        # p_value = 1.3264105853957495e-118 异标签混合分布符合预期结果
        # ###








# model = MyModel()
# params_dict = {}
# for name, param in model.named_parameters():
#     params_dict[name] = param.detach().clone()
#
# print(params_dict)   sys.argv
# look_model()
if __name__ == "__main__":
    # KSGeneral()
    # KSTest()
    look_x_and_pro()