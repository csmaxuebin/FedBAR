import time

import numpy as np
import torch
from flcore.clients.clientbase import Client

from system.utils.privacy import mixup_data, ref_cor_y


class clientBABU(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.rateavg = 0
        self.max_norm = 0.
        self.adapt = 0.5
        self.fine_tuning_steps = args.fine_tuning_steps

        for param in self.model.head.parameters():
            param.requires_grad = False

    def train(self):
        trainloader = self.load_train_data()

        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        reference_base = [param.clone() for param in self.model.parameters()]  # 初始化参考点参数
        reference_head = [param.clone() for param in self.model.head.parameters()]  # 初始化参考点参数

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            self.cors = -1
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.defense != "None":
                    y = self.to_one_hot(y, num_classes=self.num_classes).to(self.device)
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
                else:
                    self.optimizer.zero_grad()
                    output = self.model(x)
                    loss = self.loss(output, y)

                # self.optimizer.zero_grad()
                # output = self.model(x)
                # loss = self.loss(output, y)
                # loss.backward()
                loss.mean().backward()
                # if self.cors == -1:
                #     _, refd = ref_cor_y(self.model.head.weight.grad.data, ground=self.model.feature,
                #                         mix=1, y=y)
                #     self.cors = refd
                # self.fcgrad = list(np.around(self.model.head.weight.grad.data.cpu().numpy(), decimals=4).reshape(512, 10))
                self.optimizer.step()

        # self.model.cpu()
            self.max_norm = \
                clip_updates(self.model.parameters(), self.model.head.parameters(), reference_base, reference_head,
                             max_norm=self.max_norm, adapt=self.adapt)

        self.rateavg /= len(trainloader)
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, base):
        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

    def fine_tune(self, which_module=['base', 'head']):
        trainloader = self.load_train_data()

        start_time = time.time()

        self.model.train()

        if 'head' in which_module:
            for param in self.model.head.parameters():
                param.requires_grad = True

        if 'base' not in which_module:
            for param in self.model.head.parameters():
                param.requires_grad = False

        for step in range(self.fine_tuning_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.defense != "None":
                    y = self.to_one_hot(y, num_classes=self.num_classes).to(self.device)
                # self.optimizer.zero_grad()
                # output = self.model(x)
                # loss = self.loss(output, y)
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
                    self.optimizer.zero_grad()
                    output = self.model(x)
                    loss = self.loss(output, y)
                loss.mean().backward()
                self.optimizer.step()

        self.train_time_cost['total_cost'] += time.time() - start_time

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
        max_norm = torch.quantile(stacked_tensor, adapt)
    else:
        lens = len(stacked_tensor)
        b = 1 - torch.sum(stacked_tensor > max_norm).int() / lens
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

class clientBABU_sorc(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.fine_tuning_steps = args.fine_tuning_steps

        for param in self.model.head.parameters():
            param.requires_grad = False

    def train(self):
        trainloader = self.load_train_data()

        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, base):
        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

    def fine_tune(self, which_module=['base', 'head']):
        trainloader = self.load_train_data()

        start_time = time.time()

        self.model.train()

        if 'head' in which_module:
            for param in self.model.head.parameters():
                param.requires_grad = True

        if 'base' not in which_module:
            for param in self.model.head.parameters():
                param.requires_grad = False

        for step in range(self.fine_tuning_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

        self.train_time_cost['total_cost'] += time.time() - start_time
