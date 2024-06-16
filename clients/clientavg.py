import copy

import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *

from system.utils.privacy import initialize_dp, get_dp_params, mixup_data, get_my_sot_mask, is_mixlabel_leak, ref_cor, \
    ref_cor_y, Soteria4batch


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.rateavg = 0
        self.max_norm = 0.
        self.adapt = 0.5

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        self.model.round_feature = []
        self.rateavg = 0
        # differential privacy
        # if self.privacy:
        #     self.model, self.optimizer, trainloader, privacy_engine = \
        #         initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)

        start_time = time.time()
        base = copy.deepcopy(self.model.state_dict())
        head = copy.deepcopy(self.model.head.state_dict())
        reference_base = [param.clone() for param in self.model.parameters()]  # 初始化参考点参数
        reference_head = [param.clone() for param in self.model.head.parameters()]  # 初始化参考点参数

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            # leak1, leak2 = 0, 0
            # batch_time = time.time()
            feature4mix = self.round_feature
            self.cors = -1
                # feature4mix = self.get_feature_from_gradient(gradient=self.global_gradient, parameters=self.model.parameters)
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                    y = y.to(self.device)
                if self.defense != "None":
                    y = self.to_one_hot(y, num_classes=self.num_classes).to(self.device)  # TODO 这target是单数字的，要转成onehot向量，抄patch的
                # y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                if self.defense == 'mix_x':
                    self.optimizer.zero_grad()
                    x, y_a, y_b, lam, _ = mixup_data(x=x, targets=y, alpha=self.defense_param, beta=self.defense_param
                                                     , aby_mix=self.aby_mix)
                    output = self.model(x)
                    loss = self.loss(output, y_b)
                    # loss = self.regmixup_criterion(pred=output, y_a=y_a, y_b=y_b, portion=lam
                    #                                , criterion=self.loss)
                elif self.defense == 'manifold':
                    self.optimizer.zero_grad()
                    output, y_a, y_b, lam, rate, _ = self.model(x, y, alpha=self.defense_param, beta=self.defense_param
                                                                , aby_mix=self.aby_mix, mask_rate=100)
                    # mask_rate统一改成mask_type吧  然后传防御模式的str done
                    # loss = self.regmixup_criterion(pred=output, y_a=y_a, y_b=y_b, portion=lam
                    #                                , criterion=self.loss)
                    loss = self.loss(output, y_b)

                    self.rateavg += rate
                elif self.defense == 'patch':
                    self.optimizer.zero_grad()
                    output, y_a, y_b, lam, rate, reweighted = self.model(x, y, alpha=self.defense_param,
                                                                         beta=self.defense_param
                                                                         , aby_mix=self.aby_mix,
                                                                         mask_rate=0)
                    loss = self.regmixup_criterion(pred=output, y_a=y_a, y_b=y_b, portion=lam
                                                   , reweighted=reweighted)
                    self.rateavg += rate

                elif self.defense == 'my':
                    self.optimizer.zero_grad()
                    if self.round_feature is None:
                        feature4mix = -1
                    output, y_a, y_b, lam, rate, reweighted, indices = self.model(x, y, alpha=self.defense_param,
                                                                                  beta=self.defense_param
                                                                                  , aby_mix=self.aby_mix,
                                                                                  mask_rate=-1)
                    # loss = self.regmixup_criterion(pred=output, y_a=y_a, y_b=y_b, lam=lam
                    #                                , criterion=self.loss, reweighted=reweighted)
                    # if self.round_feature is None:
                    #     loss = self.loss(output, y)
                    # else:
                    #     loss = y_b * self.loss(output, y)
                    loss = self.regmixup_criterion(pred=output, y_a=y_a, y_b=y_b, portion=lam
                                                   , reweighted=reweighted)
                    self.rateavg += rate
                elif self.defense == 'my_hard':
                    self.optimizer.zero_grad()
                    if self.round_feature is None:
                        feature4mix = -1
                    output, y_a, y_b, lam, rate, reweighted = self.model(x, y, alpha=self.defense_param,
                                                                         beta=self.defense_param
                                                                         , aby_mix=-1,
                                                                         mask_rate=-2)
                    # loss = self.regmixup_criterion(pred=output, y_a=y_a, y_b=y_b, lam=lam
                    #                                , criterion=self.loss, reweighted=reweighted)
                    # if self.round_feature is None:
                    #     loss = self.loss(output, y)
                    # else:
                    #     loss = y_b * self.loss(output, y)
                    loss = self.regmixup_criterion(pred=output, y_a=y_a, y_b=y_b, portion=lam
                                                   , reweighted=reweighted)
                    self.rateavg += rate
                # elif self.defense == 'soteria':
                #     self.optimizer.zero_grad()
                #     output, sot_mask = self.model(x, mask_rate=self.defense_param)
                #     loss = self.loss(output, y)

                # elif self.defense == 'mix_sot':
                #     self.optimizer.zero_grad()
                #     features = self.model.base(x)
                #     masks = get_sot_mask()
                #
                #     output, y_a, y_b, lam = self.model(x, y, alpha=self.defense_param, beta=self.defense_param
                #                                        , aby_mix=self.aby_mix)
                #     # loss = self.loss(output, y)
                #     loss = self.regmixup_criterion(pred=output, y_a=y_a, y_b=y_b, lam=lam
                #                                    , criterion=self.loss)
                #     pass

                else:
                    if self.defense == 'soteria':
                        x.requires_grad = True
                    self.model.zero_grad()
                    output = self.model(x)
                    loss = self.loss(output, y)

                if self.defense == 'soteria':
                    # rate是留存率  --use_sot
                    # sot_mask = get_my_sot_mask(model=self.model, x=x, y=y, indice=indices, pruning_rate=rate)
                    #掩码用异类特征Uxishu的
                    # sot_mask = indices
                    feature = self.model.get_ori_feature()
                    # 这个代表激活次数，看看和梯度大小有什么关系，看不出什么关系
                    # sot_mask = torch.gt(feature, torch.mean(feature, dim=1, keepdim=True)).int()
                    # 得到的只有一批样本的聚合梯度，如何使用掩码？
                    sot_mask = Soteria4batch(model=self.model, x=x, feature=feature, pruning_rate=self.defense_param)
                    loss.mean().backward()
                    self.model.head.weight.grad.data = self.model.head.weight.grad.data * (1 - torch.Tensor(sot_mask).cuda())
                    self.model.detach_feature()
                    # temp = time.time() - batch_time
                    # print("一个batch完成，用时{}秒".format(temp))
                    # batch_time = time.time()
                else:
                    # 汇总梯度

                    # loss[0].backward()
                    loss.mean().backward()

                # 裁剪
                # torch.nn.utils.clip_grad_value_(self.model.head.parameters(), 1.0)  # 或者使用clip_grad_value_进行梯度值的裁剪
                # temp = torch.tensor(np.percentile(self.model.head.weight.grad.flatten().cpu().numpy(), [0, 99]),
                #                     device="cuda")
                # upper_bound, lower_bound = temp[0], temp[1]
                # for param in self.model.head.parameters():
                #     param.grad.detach().clamp_(max=upper_bound)
                # 先099后m1n2，训练损失降不下来，疑似梯度爆炸
                # if self.norm2:
                #     torch.nn.utils.clip_grad_norm_(self.model.head.parameters(), max_norm=1.0, norm_type=2.0)  # 或者使用clip_grad_value_进行梯度值的裁剪
                # else:
                #     temp = torch.tensor(np.percentile(self.model.head.weight.grad.flatten().cpu().numpy(), [0, 99]), device="cuda")
                #     lower_bound, upper_bound = temp[0], temp[1]
                #     for param in self.model.head.parameters():
                #         param.grad.detach().clamp_(max=upper_bound)
                # 一组batchsize为m的梯度，为其中每个样本分别找到其γ分位点C，同时也是裁剪阈值？
                # 计算b=((1-γ) * 梯度小于C的个数 - γ * 梯度大于C的个数)/m
                # 更新C为C*exp(-学习率*(b-γ))

                if self.cors == -1:
                    _, refd = ref_cor_y(self.model.head.weight.grad.data, ground=self.model.feature,
                                        mix=1, y=y)
                    self.cors = refd
                self.fcgrad = list(np.around(self.model.head.weight.grad.data.cpu().numpy(), decimals=4).reshape(512, 10))
                # main_y = torch.argmax(torch.sum(y, dim=0), dim=0)
                # sot_mask = get_my_sot_mask(model=self.model, x=x, y=y, main_y=main_y, pruning_rate=90)
                # self.model.head.weight.grad.data = self.model.head.weight.grad.data * (1 - torch.Tensor(sot_mask).cuda())

                # is_leak1, is_leak2 = is_mixlabel_leak(torch.sum(self.model.head.weight.grad, dim=1), torch.sum(y, dim=0))
                # leak1 += 1 if is_leak1 else 0
                # leak2 += 1 if is_leak2 else 0
                self.optimizer.step()
                # break  # 测cor用的
            # if self.norm2:  #要用整轮的更新量计算自适应maxnorm来裁剪
            if self.defense in ["mix_x", "manifold", "patch", "my", "my_hard", "soteria"]:
                self.max_norm = \
                    clip_updates(self.model.parameters(), self.model.head.parameters(), reference_base, reference_head,
                                 max_norm =self.max_norm, adapt=self.adapt)

            # update = clip_update(old_param=[base, head], new_param=[self.model.state_dict(), self.model.head.state_dict()])
            # self.max_norm =
            # torch.nn.utils.clip_grad_norm_(self.model.head.parameters(), max_norm=self.max_norm,
            #                                norm_type=2.0)  # 或者使用clip_grad_value_进行梯度值的裁剪

            # self.model.load_state_dict(base)
            # self.model.head.load_state_dict(head)
            # with torch.no_grad():
            #     for name, param in self.model.named_parameters():
            #         param.data += update[name]
            # else:
            #     temp = torch.tensor(np.percentile(self.model.head.weight.grad.flatten().cpu().numpy(), [0, 99]),
            #                         device="cuda")
            #     lower_bound, upper_bound = temp[0], temp[1]
            #     for param in self.model.head.parameters():
            #         param.grad.detach().clamp_(max=upper_bound)

# 完美   不会发现我用谁混合的 失败  还是会发现
        # self.round_feature = torch.mean(torch.stack(self.model.round_feature, dim=0), dim=0)  # (batch_count, 512) --> (1, 512)
        # self.model.cpu()
        # 平均稀疏率
        self.rateavg /= len(trainloader)
        # self.cors /= len(trainloader)
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        # 推理成功率
        # self.leak1 = leak1 / len(trainloader)
        # self.leak2 = leak2 / len(trainloader)

        self.model.detach_feature()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        # if self.privacy:
        #     eps, DELTA = get_dp_params(privacy_engine)
        #     print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

def clip_updates(base_params, head_params, reference_base, reference_head, max_norm=0., adapt=0.5, lrate=0.2):
    updates = []
    for param, ref_param in zip(base_params, reference_base):
        update = param - ref_param  # 计算参数的更新量
        updates.append(update.flatten())
    # for param, ref_param in zip(head_params, reference_head):
    #     update = param - ref_param  # 计算参数的更新量
    #     updates.append(update.flatten())
    flattened_tensors = [tensor.flatten() for tensor in updates]

    # 将所有张量拼接为一个长向量
    stacked_tensor = torch.cat(flattened_tensors)
    if max_norm == 0.:
        # max_norm = torch.quantile(stacked_tensor, adapt)
        max_norm = 2
    else:
        lens = len(stacked_tensor)
        b = 1 - torch.sum(abs(stacked_tensor) > max_norm).int() / lens
        max_norm = max_norm * torch.exp(-lrate * (b - adapt))


    for param, ref_param in zip(base_params, reference_base):
        update = param - ref_param  # 计算参数的更新量
        norm = torch.norm(update)
        if norm > max_norm:
            param.data.copy_(ref_param + update * (max_norm / norm))
    # for param, ref_param in zip(head_params, reference_head):
    #     norm = torch.norm(update)
    #     if norm > max_norm:
    #         param.data.copy_(ref_param + update * (max_norm / norm))

    return max_norm


def clip_update(old_param, new_param):
    base = old_param[0]
    head = old_param[1]
    update = {}
    base_after = new_param[0]
    head_after = new_param[1]
    for name, param_before in base.items():
        param_after = base_after[name]
        update[name] = param_after - param_before
    for name, param_before in head.items():
        param_after = head_after[name]
        update[name] = param_after - param_before
    max_norm = 1  # 设置裁剪的范数阈值
    total_norm = 0.0  # 更新量的总范数
    for name, value in update.items():
        total_norm += torch.norm(value).item() ** 2  # 计算每个参数更新量的范数的平方

    total_norm = total_norm ** 0.5  # 取平方根得到更新量的总范数

    clip_coef = max_norm / (total_norm + 1e-6)  # 计算裁剪系数
    if clip_coef < 1:
        for name, value in update.items():
            update[name] *= clip_coef  # 裁剪更新量
    return update
