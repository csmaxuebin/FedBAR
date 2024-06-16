import copy
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data


class Client(object):
    """
    Base class for clients in federated learning.
    客户端基类
    """

    def __init__(self, args, id, train_samples, test_samples, server, **kwargs):
        self.global_feature = None
        self.loss_alpha, self.loss_beta = 1, 1
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name
        self.server = server  # 属于哪个服务器
        self.old_fc_param = None
        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps
        self.gradient = None
        self.aby_mix = args.aby_mix
        self.global_gradient = None
        self.round_feature = None
        self.domain_count = None
        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.defense = args.defense
        self.defense_param = args.defense_param
        self.dp_sigma = args.dp_sigma
        self.sample_rate = self.batch_size / self.train_samples
        self.use_sot = args.use_sot

        # self.loss = self.regmixup_criterion if args.mix_on else nn.CrossEntropyLoss()
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay
        self.leak1 = 1
        self.leak2 = 1
        self.norm2 = False
        self.max_norm = args.initnorm

    def regmixup_criterion(self, pred, y_a, y_b, portion=0, criterion=nn.functional.cross_entropy, reweighted=None):
        softmax = nn.Softmax(dim=1).cuda()
        if portion.ndim == 0:  # 最终变成(batchsize, 2)的格式，统一计算
            portion = torch.cat(((torch.ones(y_a.size(0)).cuda() * portion).unsqueeze(dim=1),
                                 (torch.ones(y_a.size(0)).cuda() * (1 - portion)).unsqueeze(dim=1)), dim=1)
        elif portion.ndim == 1:  # 最终变成(batchsize, 2)的格式，统一计算
            portion = torch.cat((portion.unsqueeze(dim=1),
                                 (1 - portion).unsqueeze(dim=1)), dim=1)
        else:
            portion = torch.reshape(torch.cat((portion.unsqueeze(dim=1),
                                     (1 - portion).unsqueeze(dim=1)), dim=1), (len(portion), 2))
        if reweighted is not None:
            return self.loss_alpha * criterion(softmax(pred), y_a, reduction="none") * portion[:, 0:1].squeeze() \
                   + portion[:, 1:2].squeeze() * criterion(softmax(pred), y_b, reduction="none") + \
                   self.loss_beta * criterion(softmax(pred), reweighted, reduction="none")
            # return portion[:, 1:2].squeeze() * criterion(softmax(pred), y_b) + \
            #        self.loss_beta * criterion(softmax(pred), reweighted)
        else:
            return portion[:, 0:1].squeeze() * criterion(softmax(pred), y_a) + \
                   portion[:, 1:2].squeeze() * criterion(softmax(pred), y_a)
    # def regmixup_criterion(self, pred, y_a, y_b, portion=0, criterion=nn.functional.cross_entropy, reweighted=None):
    #     if reweighted is not None:
    #         return self.loss_alpha * criterion(pred, y_a) * \
    #                portion + criterion(pred, y_b) * (1. - portion) + \
    #                self.loss_beta * criterion(pred, reweighted)
    #     else:
    #         return portion * criterion(pred, y_a) + (1 - portion) * criterion(pred, y_b)
    def regmixup_BCE(self, pred, y_a, y_b, portion=0, criterion=nn.BCELoss().cuda(), reweighted=None):
        softmax = nn.Softmax(dim=1).cuda()
        if portion.ndim == 0:  # 最终变成(batchsize, 2)的格式，统一计算
            portion = torch.cat(((torch.ones(y_a.size(0)).cuda() * portion).unsqueeze(dim=1),
                                 (torch.ones(y_a.size(0)).cuda() * (1 - portion)).unsqueeze(dim=1)), dim=1)
        if portion.ndim == 1:  # 最终变成(batchsize, 2)的格式，统一计算
            portion = torch.cat((portion.unsqueeze(dim=1),
                                 (1 - portion).unsqueeze(dim=1)), dim=1)
        if reweighted is not None:
            return self.loss_alpha * criterion(softmax(pred), y_a) * portion[:, 0:1].squeeze() \
                   + portion[:, 1:2].squeeze() * criterion(softmax(pred), y_b) + \
                   self.loss_beta * criterion(softmax(pred), reweighted)
            # return portion[:, 1:2].squeeze() * criterion(softmax(pred), y_b) + \
            #        self.loss_beta * criterion(softmax(pred), reweighted)
        else:
            return portion[:, 0:1].squeeze() * criterion(softmax(pred), y_a) + \
                   portion[:, 1:2].squeeze() * criterion(softmax(pred), y_a)
    ### 旧的loss
    # def regmixup_criterion(self, pred, y_a, y_b, lam, criterion=nn.functional.cross_entropy, reweighted=None):
    #     if reweighted is not None:
    #         return self.loss_alpha * self.loss(pred, y_a) * \
    #                lam + self.loss(pred, y_b) * (1. - lam) + \
    #                self.loss_beta * self.loss(pred, reweighted)
    #     else:
    #         return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    ###

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        # 这里调整客户端训练用的数据集上下限bound, -1代表全部采用
        train_data = read_client_data(self.dataset, self.id, is_train=True, bound=-1)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        # 这里调整客户端测试用的数据集上下限bound
        test_data = read_client_data(self.dataset, self.id, is_train=False, bound=-1)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
        if self.global_gradient is not None:
            self.global_gradient = list(model.head.parameters())[0] - self.old_fc_param
        else:
            self.global_gradient = 0
        self.old_fc_param = list(model.head.parameters())[0]

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                # losses += loss.item() * y.shape[0]
                losses += sum(loss)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))

    def get_feature_from_gradient(self, gradient, parameters=None):
        """

        Args:
            gradient:fc层的梯度
            parameters:fc层的模型参数

        Returns:

        """
        g = gradient  # 取出梯度
        w = parameters if parameters is not None else self.model.parameters[-1]  # 取出参数

        # 计算K
        udldu = np.dot(g.reshape(-1), w.reshape(-1))  # AttributeError: 'int' object has no attribute 'reshape'第一轮怎么办
        lr = 0.01
        u = torch.tensor(0).to(self.device).requires_grad_(True)  # u就是模型输出的预测值
        udldu = torch.tensor(udldu).to(self.device)
        optimizer = torch.optim.Adam([u], lr=lr)
        loss_fn = nn.MSELoss()
        for i in range(30000):
            optimizer.zero_grad()
            udldu_ = -u / (1 + torch.exp(u))
            l = loss_fn(udldu_, udldu)
            l.backward()
            optimizer.step()
        # udldu_ = -u / (1 + torch.exp(u))
        u = u.detach()  # 传入梯度与参数的点乘之和

        # For simplicity assume y as known here. For details please refer to the paper.
        # 他的意思是假设y已知，这并不影响我使用它 如何假设一个合理的Y 他要求y为标签的类别值，如第7类 放弃

        # y = torch.tensor([-1 if n == 0 else n for n in y]).reshape(-1, 1)
        # y = y.mean() if y.mean() != 0 else 0.1
        y = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1,
                          0.1, 0.1, 0.1, 0.1, 0.1])
        # 假设一个均匀嗷

        k = -y / (1 + np.exp(u))
        k = k.reshape(1, -1)

        x_ = [g / c for g, c in zip(g, k) if c != 0]
        x_ = torch.mean(torch.tensor(x_))
        return x_

    def set_round_feature(self, feature):
        self.round_feature = feature

    def to_one_hot(self, inp, num_classes=10):
        """
        creates a one hot encoding that is a representation of categorical variables as binary vectors for the given label.
        Args:
            self: label of a sample.
            num_classes: the number of labels or classes that we have in the multi class classification task.

        Returns:
            one hot encoding vector of the specific target.
        """
        y_onehot = torch.FloatTensor(inp.size(0), num_classes)
        y_onehot.zero_()

        y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)

        return torch.autograd.Variable(y_onehot.cuda(), requires_grad=False)
