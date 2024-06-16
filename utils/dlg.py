# -*- coding: utf-8 -*-
import logging
import math
import sys

import breaching
import numpy as np
import torch
import torch.nn.functional as F


class attack:
    def __init__(self, device=None, model="resnet18"):
        logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
        logger = logging.getLogger()

        self.cfg = breaching.get_config(overrides=["case=5_small_batch_imagenet", "attack=seethroughgradients"])

        if device == None:
            self.device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = device
        torch.backends.cudnn.benchmark = self.cfg.case.impl.benchmark
        self.setup = dict(device=device, dtype=getattr(torch, self.cfg.case.impl.dtype))

        self.cfg.case.data.partition = "unique-class"
        self.cfg.case.user.user_idx = 0

        self.cfg.case.user.num_data_points = 1

        self.cfg.case.model = "resnet18"  # also options are resnet50ssl or resnetmoco

        # In this paper, there are no public buffers, but users send their batch norm statistic updates in combination
        # with their gradient updates to the server:
        self.cfg.case.server.provide_public_buffers = False
        self.cfg.case.user.provide_buffers = True

    def attack(self, model, loss):
        attacker = breaching.attacks.prepare_attack(model, loss, self.cfg.attack, self.setup)
        # breaching.utils.overview(server, user, attacker) 只是报一下数据
        metadata = dict(
            num_data_points=self.num_data_points if self.provide_num_data_points else None,
            labels=data["labels"].sort()[0] if self.provide_labels else None,
            local_hyperparams=None,
        )
        shared_data = dict(
            gradients=shared_grads, buffers=shared_buffers if self.provide_buffers else None, metadata=metadata
        )
        true_user_data = dict(data=data[self.data_key], labels=data["labels"], buffers=shared_buffers)

        server_payload = server.distribute_payload()
        shared_data, true_user_data = user.compute_local_updates(server_payload)

        reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], {}, dryrun=cfg.dryrun)


# https://github.com/jackfrued/Python-1/blob/master/analysis/compression_analysis/psnr.py
def psnr(original, contrast):
    mse = np.mean((original - contrast) ** 2) / 3
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return PSNR


def DLG(net, origin_grad, target_inputs, opti="LBFGS", is_idlg=False):
    '''

    Args:
        opti: 优化器
        is_idlg: 用不用IDLG
        net: 客户端模型
        origin_grad:客户端上传的模型更新量
        target_inputs:客户端的logit输出

    Returns:psnr值

    '''
    criterion = torch.nn.MSELoss()
    cnt = 0
    psnr_val = 0
    for idx, (gt_data, gt_out) in enumerate(target_inputs):
        # 生成伪造样本和伪造标签
        dummy_data = torch.randn_like(gt_data, requires_grad=True)
        dummy_out = torch.randn_like(gt_out, requires_grad=True)

        if is_idlg:
            # TODO 暂停使用  要逐样本攻击   那还不如GGL了
            last_weight_min = torch.argmin(torch.sum(origin_grad[-2], dim=-1), dim=-1)
            dummy_out = last_weight_min.detach().reshape((1,)).requires_grad_(False)
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            if opti == "LBFGS":
                optimizer = torch.optim.LBFGS([dummy_data])
            elif opti == "adam":
                optimizer = torch.optim.Adam([dummy_data], lr=0.1)
            else:
                raise ValueError()
        else:
            if opti == "LBFGS":
                optimizer = torch.optim.LBFGS([dummy_data, dummy_out])
            elif opti == "adam":
                optimizer = torch.optim.Adam([dummy_data, dummy_out], lr=0.1)
            else:
                raise ValueError()

        history = [gt_data.data.cpu().numpy(), F.sigmoid(dummy_data).data.cpu().numpy()]
        for iters in range(100):
            def closure():
                optimizer.zero_grad()

                dummy_pred = net(F.sigmoid(dummy_data))
                dummy_loss = criterion(dummy_pred, dummy_out)
                dummy_grad = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                grad_diff = 0
                for gx, gy in zip(dummy_grad, origin_grad):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()

                return grad_diff

            optimizer.step(closure)

        # plt.figure(figsize=(3*len(history), 4))
        # for i in range(len(history)):
        #     plt.subplot(1, len(history), i + 1)
        #     plt.imshow(history[i])
        #     plt.title("iter=%d" % (i * 10))
        #     plt.axis('off')

        # plt.savefig(f'dlg_{algo}_{cid}_{idx}' + '.pdf', bbox_inches="tight")

        history.append(F.sigmoid(dummy_data).data.cpu().numpy())

        p = psnr(history[0], history[2])
        if not math.isnan(p):
            psnr_val += p
            cnt += 1

    if cnt > 0:
        return psnr_val / cnt
    else:
        return None


def idlg(net, origin_grad, target_inputs):
    criterion = torch.nn.MSELoss()
    cnt = 0
    psnr_val = 0
    for idx, (gt_data, gt_out) in enumerate(target_inputs):
        # generate dummy data and label
        dummy_data = torch.randn_like(gt_data, requires_grad=True)
        dummy_out = torch.randn_like(gt_out, requires_grad=True)

        optimizer = torch.optim.LBFGS([dummy_data, dummy_out])

        history = [gt_data.data.cpu().numpy(), F.sigmoid(dummy_data).data.cpu().numpy()]
        for iters in range(100):
            def closure():
                optimizer.zero_grad()

                dummy_pred = net(F.sigmoid(dummy_data))
                dummy_loss = criterion(dummy_pred, dummy_out)
                dummy_grad = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                grad_diff = 0
                for gx, gy in zip(dummy_grad, origin_grad):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()

                return grad_diff

            optimizer.step(closure)

        # plt.figure(figsize=(3*len(history), 4))
        # for i in range(len(history)):
        #     plt.subplot(1, len(history), i + 1)
        #     plt.imshow(history[i])
        #     plt.title("iter=%d" % (i * 10))
        #     plt.axis('off')

        # plt.savefig(f'dlg_{algo}_{cid}_{idx}' + '.pdf', bbox_inches="tight")

        history.append(F.sigmoid(dummy_data).data.cpu().numpy())

        p = psnr(history[0], history[2])
        if not math.isnan(p):
            psnr_val += p
            cnt += 1

    if cnt > 0:
        return psnr_val / cnt
    else:
        return None
