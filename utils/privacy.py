from decimal import Decimal

import numpy as np
import torch
from opacus import PrivacyEngine
from scipy.special import comb

MAX_GRAD_NORM = 1.0
DELTA = 1e-5


def representation_ori(model, ground_truth, pruning_rate=10):
    # start_time = time.time()
    feature_fc1_graph = model.get_feature()
    deviation_f1_target = torch.zeros_like(feature_fc1_graph)  # r'==r
    deviation_f1_x_norm = torch.zeros_like(feature_fc1_graph)  # x'!=x
    for f in range(deviation_f1_x_norm.size(1)):  # range(0, 2304) 2304=feature_fc1_graph.shape
        deviation_f1_target[:, f] = 1  # 把deviation_f1_target的第f列设置成1 （为啥
        # loss调用后向传播，所以feature_fc1_graph = loss
        feature_fc1_graph.backward(deviation_f1_target, retain_graph=True)
        # 进行一次backward之后，各个节点的值会清除，这样进行第二次backward会报错，如果加上retain_graph=True后,可以再来一次backward。
        deviation_f1_x = ground_truth  # 不知道为什么都还是0
        # deviation_f1_x是被攻击图片ground_truth的梯度数据，(1, 3, 32, 32)
        deviation_f1_x_norm[:, f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1) / (
                feature_fc1_graph.data[:, f] + 0.1)
        # 相当于把梯度平铺了，然后计算平铺层的Frobenius范数(全元素平方和)，然后除以loss+0.1  为啥
        model.zero_grad()  # 清零模型梯度
        # ground_truth.grad.data.zero_()  # 清零图片梯度
        deviation_f1_target[:, f] = 0  # 归零第一行设为1的值
    deviation_f1_x_norm_sum = deviation_f1_x_norm.sum(axis=0)
    thresh = np.percentile(deviation_f1_x_norm_sum.flatten().cpu().numpy(), pruning_rate)
    mask = np.where(abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)
    # input_gradient = torch.autograd.grad(loss, model.parameters())
    model.detach_feature()
    # print("rep耗时{:4f}秒".format(time.time() - start_time))
    return mask


def representation_ori_lookfor(model, ground_truth, pruning_rate=10):
    if pruning_rate < 1:
        pruning_rate *= 100
    # 获得模型的特征向量
    feature_fc1_graph = model.get_feature()
    # 初始化目标特征向量和输入样本的中间变量的形状和类型
    deviation_f1_target = torch.zeros_like(feature_fc1_graph)
    deviation_f1_x_norm = torch.zeros_like(feature_fc1_graph)
    # 对每个特征向量进行遍历
    for f in range(deviation_f1_x_norm.size(1)):
        # 目标向量对该特征向量取值为1，其他取值为0
        deviation_f1_target[:, f] = 1
        # 反向传播目标向量，获得输入样本的中间变量
        feature_fc1_graph.backward(deviation_f1_target, retain_graph=True)
        # 初始化输入样本的中间变量，即上文所说的d_i，由于求的是向量的范数，因此需要对输入样本进行展平后再求范数
        deviation_f1_x = ground_truth
        deviation_f1_x_norm[:, f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1) / (
                feature_fc1_graph.data[:, f] + 0.1)
        # 清除梯度
        model.zero_grad()
        # 目标向量记得要重置
        deviation_f1_target[:, f] = 0
    # 对所有特征向量的输入样本中间变量的范数进行求和并排序，得到剪枝的阈值
    deviation_f1_x_norm_sum = deviation_f1_x_norm.sum(axis=0)
    thresh = np.percentile(deviation_f1_x_norm_sum.flatten().cpu().numpy(), pruning_rate)
    # 根据阈值生成mask，即小于阈值的位置取0，大于等于阈值的位置取1
    mask = np.where(abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)
    # 返回mask
    model.detach_feature()
    return mask


def representation_after(input_gradient, model, ground_truth, pruning_rate=10):
    """
    Defense proposed in the Soteria paper.
    param:
        - input_gradient: the input_gradient
        - model: the ResNet-18 model
        - ground_truth: the benign image (for learning perturbed representation)
        - pruning_rate: the prune percentage
    Note: This implementation only works for ResNet-18
    """
    device = input_gradient[0].device

    gt_data = ground_truth.clone()
    gt_data.requires_grad = True

    # register forward hook to get intermediate layer output
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = input[0]

        return hook

    # for ResNet-18
    # handle = model.fc.register_forward_hook(get_activation('flatten'))
    out = model(gt_data)

    feature_graph = activation['flatten']

    deviation_target = torch.zeros_like(feature_graph)
    deviation_x_norm = torch.zeros_like(feature_graph)
    for f in range(deviation_x_norm.size(1)):
        deviation_target[:, f] = 1
        feature_graph.backward(deviation_target, retain_graph=True)
        deviation_f1_x = gt_data.grad.data
        deviation_x_norm[:, f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1) / (
                (feature_graph.data[:, f]) + 1e-10)
        model.zero_grad()
        gt_data.grad.data.zero_()
        deviation_target[:, f] = 0

    # prune r_i corresponding to smallest ||dr_i/dX||/||r_i||
    deviation_x_norm_sum = deviation_x_norm.sum(axis=0)
    thresh = np.percentile(deviation_x_norm_sum.flatten().cpu().numpy(), pruning_rate)
    mask = np.where(abs(deviation_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)

    # print('Soteria mask: ', sum(mask))
    #
    # gradient = [grad for grad in input_gradient]
    # # apply mask
    # gradient[-2] = gradient[-2] * torch.Tensor(mask).to(device)

    # handle.remove()

    return mask


def initialize_dp(model, optimizer, data_loader, dp_sigma):
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=dp_sigma,
        max_grad_norm=MAX_GRAD_NORM,
    )

    return model, optimizer, data_loader, privacy_engine


def _get_sensitivity(gradient_layer):
    sens = torch.max(torch.abs(gradient_layer.flatten()))
    return sens


def get_noise(gradient_layer, defense_param, defense="laplace", device=None):
    # TODO 手动加噪
    # sensitivity = _get_sensitivity(gradient_layer)
    sensitivity = 1
    noise = torch.zeros_like(gradient_layer)
    # sensitivity = torch.max(torch.abs(gradient))

    if defense == "laplace":
        noise = _get_laplace_noise(gradient_layer.shape, defense_param, sensitivity, device=device)
    elif defense == "gaussian":
        noise = _get_gaussian_noise(gradient_layer.shape, defense_param, sensitivity, device=device)
    elif defense == "test":
        noise = torch.zeros(size=gradient_layer.shape, device=device)
    # print("噪声为:{}".format(noise))

    return noise


def _get_gaussian_noise(data_shape, sigma, sensitivity=1, device=None):
    return torch.normal(0, sigma, data_shape).to(device)


def _get_laplace_noise(data_shape, epsilon, sensitivity=1.0, device=None):
    """
       laplace noise
    """
    noise_scale = torch.tensor(sensitivity / torch.tensor(epsilon), device=device)
    mean = torch.tensor(0.0, device=device)
    noise = torch.normal(mean=mean, std=noise_scale, size=data_shape).to(device)
    # noise = torch.normal(mean=torch.tensor(0.0, device=device), std=torch.tensor(1.0, device=device), size=data_shape).to(device)

    return noise


def calibrating_sampled_gaussian(sampling_rate=1.0, eps=4.0, bad_event_delta=1e-5, iters=1, err=1e-3):
    """
    Calibrate noise to privacy budgets
    """
    sigma_max = 100
    sigma_min = 0.1

    def binary_search(left, right):
        mid = (left + right) / 2

        lbd = _search_dp(sampling_rate, mid, bad_event_delta, iters)
        ubd = _search_dp(sampling_rate, left, bad_event_delta, iters)

        if ubd > eps and lbd > eps:  # min noise & mid noise are too small
            left = mid
        elif ubd > eps > lbd:  # mid noise is too large
            right = mid
        else:
            print("an error occurs in func: binary search!")
            return -1
        return left, right

    # check
    if _search_dp(sampling_rate, sigma_max, bad_event_delta, iters) > eps:
        print("noise > 100")
        return -1

    while sigma_max - sigma_min > err:
        sigma_min, sigma_max = binary_search(sigma_min, sigma_max)
    return sigma_max


def _search_dp(sampling_rate, sigma, bad_event, iters=1):
    """
    Given the sampling rate, variance of Gaussian noise, and privacy parameter delta,
    this function returns the corresponding DP budget.
    """
    min_dp = 1e5
    for alpha in list(range(2, 101)):
        rdp = iters * _compute_rdp(alpha, sampling_rate, sigma)
        dp = _rdp2dp(rdp, bad_event, alpha)
        min_dp = min(min_dp, dp)
    return min_dp


def _compute_rdp(alpha, sampling_rate, sigma):
    """
    RDP for subsampled Gaussian mechanism, Ref:
    - Mironov, Ilya, Kunal Talwar, and Li Zhang. R\'enyi differential privacy of the sampled gaussian mechanism. arXiv preprint 2019.
    """
    sum_ = Decimal(0.0)
    for k in range(0, alpha + 1):
        sum_ += Decimal(comb(alpha, k)) * Decimal(1 - sampling_rate) ** Decimal(alpha - k) * Decimal(sampling_rate ** k) \
                * Decimal(np.e) ** (Decimal(k ** 2 - k) / Decimal(2 * sigma ** 2))
    rdp = sum_.ln() / Decimal(alpha - 1)
    return float(rdp)


def _rdp2dp(rdp, bad_event, alpha):
    """
    convert RDP to DP, Ref:
    - Canonne, Clément L., Gautam Kamath, and Thomas Steinke. The discrete gaussian for differential privacy. In NeurIPS, 2020. (See Proposition 12)
    - Asoodeh, S., Liao, J., Calmon, F.P., Kosut, O. and Sankar, L., A better bound gives a hundred rounds: Enhanced privacy guarantees via f-divergences. In ISIT, 2020. (See Lemma 1)
    """
    return rdp + 1.0 / (alpha - 1) * (np.log(1.0 / bad_event) + (alpha - 1) * np.log(1 - 1.0 / alpha) - np.log(alpha))


def get_dp_params(privacy_engine):
    return privacy_engine.get_epsilon(delta=DELTA), DELTA


def get_random_index_with(targets, batch_size=None, aby=0):
    # aby=-1是异标签  aby=1是同标签
    if batch_size is None:
        batch_size = targets.size()[0]
    targets = torch.argmax(targets.int(), dim=1)
    y_uni = torch.unique(targets)
    if aby == 0 or len(y_uni) < 2:
        index = torch.randperm(batch_size).cuda() if batch_size > 2 else torch.tensor([1, 0]).cuda()
        # index = torch.randperm(batch_size).cuda()
    elif aby == -1:  # 不同标签混合
        # y_uni = torch.unique(y)
        y_dict = {}
        for clas in y_uni:  # 为每个类构建随机域，其元素值为标签所在位置
            index = torch.arange(batch_size).cuda()
            # temp = torch.ones_like(targets) * clas
            mask = targets != clas
            y_dict[clas.item()] = torch.masked_select(index, mask)
            # torch.where(y == temp, -1, index)
            # y_dict[clas.item()] = np.delete(.cpu(), -1).cuda()  # 随机域对应位置
        index = torch.empty(0).cuda()
        for idx, label in enumerate(targets):  # 为每个样本挑选随机对象
            ablabel_index = y_dict[label.item()][torch.randint(high=len(y_dict[label.item()]), size=(1,))]
            index = torch.cat((index, ablabel_index))
        index = index.int()
    else:  # 相同标签混合
        # y_uni = torch.unique(y)
        y_dict = {}
        for clas in y_uni:  # 为每个类构建随机域，其元素值为标签所在位置
            index = torch.arange(batch_size).cuda()
            # temp = torch.ones_like(targets) * clas
            mask = targets == clas  # 选出同类样本
            y_dict[clas.item()] = torch.masked_select(index, mask)  # (64,)的异类布尔表
            # y_dict[clas.item()] = np.delete(torch.where(y != temp, -1, index).cpu(), -1).cuda()  # 随机域对应位置
        index = torch.empty(0).cuda()
        for idx, label in enumerate(targets):  # 为每个样本挑选随机对象
            ablabel_index = y_dict[label.item()][torch.randint(high=len(y_dict[label.item()]), size=(1,))]
            index = torch.cat((index, ablabel_index))
        index = index.int()
    return index


def get_two_random_index_with(targets, batch_size=None, aby=0):
    index1 = get_random_index_with(targets=targets, batch_size=batch_size, aby=aby)
    index2 = get_random_index_with(targets=targets, batch_size=batch_size, aby=aby)
    index = torch.stack((index1, index2), dim=1)
    return index


#
# def patchup_data(x, targets, save_mask, alpha=2.0, beta=2.0, aby_mix=0, indices=None):
#     if alpha > 0 and beta > 0:
#         distribut = torch.distributions.beta.Beta(alpha, beta)
#         lam = distribut.sample((targets.size()[0], 1)).cuda()
#         # lam = torch.random.beta(alpha, beta, targets.size()[0])
#         # lam = np.random.beta(alpha, beta)
#         # for rate in lam:
#         #     if rate < 0.5:
#         #         rate = 1 - rate
#         lam[lam < 0.5] = 1 - lam[lam < 0.5]
#     else:
#         lam = 1  # cross-entropy
#     if indices is None:
#         indices1 = get_random_index_with(targets=targets, aby=aby_mix)
#         indices2 = get_random_index_with(targets=targets, aby=0)
#
#     # if isinstance(save_mask, int):
#     #     feature_dis = -torch.abs(x - x[indices])
#     #     if 0 < save_mask < 1:
#     #         save_mask *= 100
#     #     select_num = int(save_mask / 100 * x.size(1))
#     #     value, index = torch.topk(feature_dis, select_num, dim=1)
#     #     lam = 1 - ((1 - lam) * torch.sum(value) / torch.sum(feature_dis))
#     #     threshold = value[:, -1]
#     #     # 根据阈值生成掩码
#     #     save_mask = torch.ge(feature_dis, threshold.unsqueeze(1)).float()
#
#     # mask代表要留下来的部分
#     unchanged = save_mask * x
#     change = 1 - save_mask
#     # for batch
#     total_feats = change.numel()
#     total_changed_pixels = change.sum()
#     total_changed_portion = total_changed_pixels / total_feats
#     total_unchanged_portion = (total_feats - total_changed_pixels) / total_feats
#     target_shuffled_onehot = 0.5 * (targets[indices1] + targets[indices2])
#     # target_reweighted这里，提高了targets的比例会使精度更高，为何？
#     # total_unchanged_portion = 1  total_changed_portion = 0
#     target_reweighted = total_unchanged_portion * targets + lam * total_changed_portion * targets + \
#                         target_shuffled_onehot * (1 - lam) * total_changed_portion
#
#     patches = change * x
#     patches = patches * lam + 0.5 * (patches[indices1] * (1 - lam) + patches[indices2] * (1 - lam))
#     target_b = lam * targets + (1 - lam) * target_shuffled_onehot
#     x = unchanged + change * patches
#     # return x, targets.long(), target_b.long(), target_reweighted.long(), total_unchanged_portion, save_mask
#     return x, targets, target_b, target_reweighted, total_unchanged_portion, save_mask

def patchup_data(x, targets, save_mask, alpha=2.0, beta=2.0, aby_mix=0, indices=None):
    if alpha > 0 and beta > 0:
        distribut = torch.distributions.beta.Beta(alpha, beta)
        lam = distribut.sample((targets.size()[0], 1)).cuda()
        lam = torch.clamp(lam, 0.3, 0.7)
        # lam = torch.random.beta(alpha, beta, targets.size()[0])
        # lam = np.random.beta(alpha, beta)
        # for rate in lam:
        #     if rate < 0.5:
        #         rate = 1 - rate
        # lam[lam < 0.5] = 1 - lam[lam < 0.5]
    else:
        lam = 1  # cross-entropy
    if indices is None:
        indices = get_random_index_with(targets=targets, aby=aby_mix)

    # if isinstance(save_mask, int):
    #     feature_dis = -torch.abs(x - x[indices])
    #     if 0 < save_mask < 1:
    #         save_mask *= 100
    #     select_num = int(save_mask / 100 * x.size(1))
    #     value, index = torch.topk(feature_dis, select_num, dim=1)
    #     lam = 1 - ((1 - lam) * torch.sum(value) / torch.sum(feature_dis))
    #     threshold = value[:, -1]
    #     # 根据阈值生成掩码
    #     save_mask = torch.ge(feature_dis, threshold.unsqueeze(1)).float()

    # mask代表要留下来的部分
    unchanged = save_mask * x
    change = 1 - save_mask
    # for batch
    total_feats = change.numel()
    total_changed_pixels = change.sum()
    total_changed_portion = total_changed_pixels / total_feats
    total_unchanged_portion = (total_feats - total_changed_pixels) / total_feats
    target_shuffled_onehot = targets[indices]
    # target_reweighted这里，提高了targets的比例会使精度更高，为何？
    # total_unchanged_portion = 1  total_changed_portion = 0
    target_reweighted = total_unchanged_portion * targets + lam * total_changed_portion * targets + \
                        target_shuffled_onehot * (1 - lam) * total_changed_portion  # (64, 10)
    patches = change * x
    patches = patches * lam + patches[indices] * (1 - lam)
    target_b = lam * targets + (1 - lam) * target_shuffled_onehot
    x = unchanged + change * patches
    # return x, targets.long(), target_b.long(), target_reweighted.long(), total_unchanged_portion, save_mask
    return x, targets, target_b, target_reweighted, total_unchanged_portion, save_mask, indices



def patchup_batch_data(x, targets, save_mask, alpha=2.0, beta=2.0, aby_mix=0, indices=None):
    if alpha > 0 and beta > 0:
        distribut = torch.distributions.beta.Beta(alpha, beta)
        lam = distribut.sample((targets.size()[0], 1)).cuda()
        lam = torch.clamp(lam, 0.3, 0.7)
    else:
        lam = 1  # cross-entropy
    if indices is None:
        indices = get_random_index_with(targets=targets, aby=aby_mix)
    batch_size = x.size()[0]
    if save_mask.ndim == 1:
        save_mask = torch.unsqueeze(save_mask, 0).expand(batch_size, -1)
    # mask代表要留下来的部分
    #     save_mask = torch.ge(feature_dis, threshold.unsqueeze(1)).float()
    unchanged = save_mask * x
    change = 1 - save_mask
    # for batch
    total_feats = change[0].numel()
    total_changed_pixels = change.sum(dim=1)
    total_changed_portion = (total_changed_pixels / total_feats).unsqueeze(dim=1)
    total_unchanged_portion = ((total_feats - total_changed_pixels) / total_feats).unsqueeze(dim=1)
    target_shuffled_onehot = targets[indices]
    target_reweighted = total_unchanged_portion * targets + lam * total_changed_portion * targets + \
                        target_shuffled_onehot * (1 - lam) * total_changed_portion  # (64, 10)
    patches = change * x
    patches = patches * lam + patches[indices] * (1 - lam)
    target_b = lam * targets + (1 - lam) * target_shuffled_onehot
    x = unchanged + change * patches
    return x, targets, target_b, target_reweighted, total_unchanged_portion, save_mask, indices


def patchup_data_hard(x, targets, save_mask, alpha=2.0, beta=2.0, aby_mix=0, indices=None, patchup_type="hard"):
    if alpha > 0 and beta > 0:
        distribut = torch.distributions.beta.Beta(alpha, beta)
        lam = distribut.sample((targets.size()[0], 1)).cuda()
        # lam = torch.random.beta(alpha, beta, targets.size()[0])
        # lam = np.random.beta(alpha, beta)
        # for rate in lam:
        #     if rate < 0.5:
        #         rate = 1 - rate
        lam[lam < 0.5] = 1 - lam[lam < 0.5]
    else:
        lam = 1  # cross-entropy
    if indices is None:
        indices = get_random_index_with(targets=targets, aby=aby_mix)

    # if isinstance(save_mask, int):
    #     feature_dis = -torch.abs(x - x[indices])
    #     if 0 < save_mask < 1:
    #         save_mask *= 100
    #     select_num = int(save_mask / 100 * x.size(1))
    #     value, index = torch.topk(feature_dis, select_num, dim=1)
    #     lam = 1 - ((1 - lam) * torch.sum(value) / torch.sum(feature_dis))
    #     threshold = value[:, -1]
    #     # 根据阈值生成掩码
    #     save_mask = torch.ge(feature_dis, threshold.unsqueeze(1)).float()

    # mask代表要留下来的部分
    unchanged = save_mask * x
    change = 1 - save_mask
    # for batch
    total_feats = change.numel()
    total_changed_pixels = change.sum()
    total_changed_portion = total_changed_pixels / total_feats
    total_unchanged_portion = (total_feats - total_changed_pixels) / total_feats
    target_shuffled_onehot = targets[indices]
    # target_reweighted这里，提高了targets的比例会使精度更高，为何？
    # total_unchanged_portion = 1  total_changed_portion = 0
    if patchup_type == "soft":  # 以soft为例
        # apply Soft PatchUp combining operation for the selected continues blocks.
        target_reweighted = total_unchanged_portion * targets + lam * total_changed_portion * targets + \
                            target_shuffled_onehot * (1 - lam) * total_changed_portion
        # W(yi, yj) = soft(gxi, gxj) = (mask .* y_a) + mix_lam(y_a, y_b) * 要改变的特征比例
        patches = change * x  # patches = mask_mix * feature 要混合的那些特征
        patches = patches * lam + patches[indices] * (1 - lam)  # 要混合的那些进行混合
        target_b = lam * targets + (1 - lam) * target_shuffled_onehot  # mix_y = mix_lam(y, y_b)
    elif patchup_type == "hard":
        # apply Hard PatchUp combining operation for the selected continues blocks.
        target_reweighted = total_unchanged_portion * targets + total_changed_portion * target_shuffled_onehot
        patches = change * x
        patches = patches[indices]
        target_b = targets[indices]
    # target_reweighted = total_unchanged_portion * targets + lam * total_changed_portion * targets + \
    #                     target_shuffled_onehot * (1 - lam) * total_changed_portion  # (64, 10)
    # patches = change * x
    # patches = patches * lam + patches[indices] * (1 - lam)
    # target_b = lam * targets + (1 - lam) * target_shuffled_onehot
    x = unchanged + change * patches
    # return x, targets.long(), target_b.long(), target_reweighted.long(), total_unchanged_portion, save_mask
    return x, targets, target_b, target_reweighted, total_unchanged_portion, save_mask


def mypatch_sqrt(x, targets, fc, save_mask, alpha=2.0, beta=2.0, aby_mix=0):
    # TODO 没改呢，想按距离做归一化然后当lam用试试看
    if alpha > 0 and beta > 0:
        lam = np.random.beta(alpha, beta)
        if lam < 0.5:
            lam = 1 - lam
    else:
        lam = 1  # cross-entropy
    indices = get_random_index_with(targets=targets, aby=aby_mix)

    if isinstance(save_mask, int):
        feature_dis = -torch.abs(x - x[indices])
        if 0 < save_mask < 1:
            save_mask *= 100
        select_num = int(save_mask / 100 * x.size(1))
        value, index = torch.topk(feature_dis, select_num, dim=1)
        lam = 1 - ((1 - lam) * torch.sum(value) / torch.sum(feature_dis))
        threshold = value[:, -1]
        # 根据阈值生成掩码
        save_mask = torch.ge(feature_dis, threshold.unsqueeze(1)).float()

    # mask代表要留下来的部分
    unchanged = save_mask * x
    change = 1 - save_mask
    # for batch
    total_feats = change.numel()
    total_changed_pixels = change.sum()
    total_feats_sum = torch.sum(x, dim=1)
    total_changed_portion_a = torch.sum(x * change, dim=1) / total_feats_sum  # 求混合部分特征和相对于全特征和的占比
    total_changed_portion_b = torch.sum(x[indices] * change, dim=1) / total_feats_sum
    # total_changed_portion_b = total_changed_portion_a[indices]

    ### 我的改动
    with torch.no_grad():
        total_unchanged_portion = torch.sum(unchanged, dim=1) / total_feats_sum
    ###

    target_shuffled_onehot = targets[indices]
    # target_reweighted = total_unchanged_portion * targets + lam * total_changed_portion * targets + \
    #                     target_shuffled_onehot * (1 - lam) * total_changed_portion
    patches = change * x
    patches = patches * lam + patches[indices] * (1 - lam)
    target_b = lam * targets + (1 - lam) * target_shuffled_onehot
    new_feature = unchanged + change * patches
    # new_lam = adjust_lam(x=x, new_feature=new_feature, targets=targets, index=indices, fc=fc, lam=lam)
    target_reweighted = total_unchanged_portion.unsqueeze(dim=1) * targets + lam * total_changed_portion_a.unsqueeze(
        dim=1) * targets + \
                        target_shuffled_onehot * (1 - lam) * total_changed_portion_b.unsqueeze(dim=1)
    # target_reweighted = total_unchanged_portion * (1 - new_lam[0]) + new_lam[:, 0:1] * targets + \
    #                     target_shuffled_onehot * new_lam[:, 1:2]
    # return x, targets.long(), target_b.long(), target_reweighted.long(), total_unchanged_portion, save_mask
    return new_feature, targets, target_b, target_reweighted, total_unchanged_portion, save_mask


def mymix_patch(x, targets, save_mask, fc, alpha=2.0, beta=2.0, aby_mix=0, indices=None):
    # TODO 没改呢，想按距离做归一化然后当lam用试试看  是上面的备份
    if alpha > 0 and beta > 0:
        lam = np.random.beta(alpha, beta)
        if lam < 0.5:
            lam = 1 - lam
    else:
        lam = 1  # cross-entropy
    if indices is None:
        indices = get_random_index_with(targets=targets, aby=aby_mix)

    if isinstance(save_mask, int):
        feature_dis = -torch.abs(x - x[indices])
        if 0 < save_mask < 1:
            save_mask *= 100
        select_num = int(save_mask / 100 * x.size(1))
        value, index = torch.topk(feature_dis, select_num, dim=1)
        lam = 1 - ((1 - lam) * torch.sum(value) / torch.sum(feature_dis))
        threshold = value[:, -1]
        # 根据阈值生成掩码
        save_mask = torch.ge(feature_dis, threshold.unsqueeze(1)).float()

    # mask代表要留下来的部分
    unchanged = save_mask * x
    change = 1 - save_mask
    # for batch
    total_feats = change.numel()
    total_changed_pixels = change.sum()
    total_changed_portion = total_changed_pixels / total_feats

    ### 我的改动
    with torch.no_grad():
        total_unchanged_portion = torch.sum(unchanged) / torch.sum(x)
    ###

    target_shuffled_onehot = targets[indices]
    # target_reweighted = total_unchanged_portion * targets + lam * total_changed_portion * targets + \
    #                     target_shuffled_onehot * (1 - lam) * total_changed_portion
    patches = change * x
    patches = patches * lam + patches[indices] * (1 - lam)
    target_b = lam * targets + (1 - lam) * target_shuffled_onehot
    new_feature = unchanged + change * patches
    # new_lam = adjust_lam(x=x, new_feature=new_feature, targets=targets, index=indices, fc=fc, lam=lam)
    target_reweighted = total_unchanged_portion * targets + lam * total_changed_portion * targets + \
                        target_shuffled_onehot * (1 - lam) * total_changed_portion
    # target_reweighted = total_unchanged_portion * (1 - new_lam[0]) + new_lam[:, 0:1] * targets + \
    #                     target_shuffled_onehot * new_lam[:, 1:2]
    # return x, targets.long(), target_b.long(), target_reweighted.long(), total_unchanged_portion, save_mask
    return new_feature, targets, target_b, target_reweighted, total_unchanged_portion, save_mask


# def mymix_patch(x, targets, save_mask, fc, alpha=2.0, beta=2.0, aby_mix=0):
#     # TODO 没改呢，想按距离做归一化然后当lam用试试看  是上面的备份
#     if alpha > 0 and beta > 0:
#         lam = np.random.beta(alpha, beta)
#         if lam < 0.5:
#             lam = 1 - lam
#     else:
#         lam = 1  # cross-entropy
#     indices = get_random_index_with(targets=targets, aby=aby_mix)
#
#     if isinstance(save_mask, int):
#         feature_dis = -torch.abs(x - x[indices])
#         if 0 < save_mask < 1:
#             save_mask *= 100
#         select_num = int(save_mask / 100 * x.size(1))
#         value, index = torch.topk(feature_dis, select_num, dim=1)
#         lam = 1 - ((1 - lam) * torch.sum(value) / torch.sum(feature_dis))
#         threshold = value[:, -1]
#         # 根据阈值生成掩码
#         save_mask = torch.ge(feature_dis, threshold.unsqueeze(1)).float()
#
#     # mask代表要留下来的部分
#     unchanged = save_mask * x
#     change = 1 - save_mask
#     # for batch
#     total_feats = change.numel()
#     total_changed_pixels = change.sum()
#     total_changed_portion = total_changed_pixels / total_feats
#
#     ### 我的改动
#     with torch.no_grad():
#         total_unchanged_portion = torch.sum(unchanged) / torch.sum(x)
#     ###
#
#     target_shuffled_onehot = targets[indices]
#     # target_reweighted = total_unchanged_portion * targets + lam * total_changed_portion * targets + \
#     #                     target_shuffled_onehot * (1 - lam) * total_changed_portion
#     patches = change * x
#     patches = patches * lam + patches[indices] * (1 - lam)
#     target_b = lam * targets + (1 - lam) * target_shuffled_onehot
#     new_feature = unchanged + patches
#     # new_lam = adjust_lam(x=x, new_feature=new_feature, targets=targets, index=indices, fc=fc, lam=lam)
#     target_reweighted = total_unchanged_portion * targets + lam * total_changed_portion * targets + \
#                         target_shuffled_onehot * (1 - lam) * total_changed_portion
#     # target_reweighted = total_unchanged_portion * (1 - new_lam[0]) + new_lam[:, 0:1] * targets + \
#     #                     target_shuffled_onehot * new_lam[:, 1:2]
#     # return x, targets.long(), target_b.long(), target_reweighted.long(), total_unchanged_portion, save_mask
#     return new_feature, targets, target_b, target_reweighted, total_unchanged_portion, save_mask
def mymix_data(x, targets, save_mask, alpha=2.0, beta=2.0, aby_mix=0):
    device = 'cuda:0' if x.get_device() == 0 else None
    if alpha > 0 and beta > 0:
        lam = np.random.beta(alpha, beta)
    else:
        lam = 1  # cross-entropy

    batch_size = x.size()[0]
    index = get_random_index_with(targets=targets, batch_size=batch_size, aby=aby_mix)
    # unchanged = save_mask * x
    # change = 1 - save_mask

    # patch = change * x
    feat = (x * lam) + (x[index] * (1 - lam))

    # feat = unchanged + patch
    y_a, y_b = targets, targets[index]
    y_b = lam * targets + (1 - lam) * targets[index]
    # RuntimeError: "nll_loss_forward_no_reduce_cuda_kernel_index" not implemented for 'Float'
    if device is not None:
        y_a.to(device)
        y_b.to(device)
    return feat, y_a, y_b, lam, save_mask


def mixup_data(x, targets, save_mask=None, alpha=2.0, beta=2.0, aby_mix=0):
    device = 'cuda:0' if x.get_device() == 0 else None
    if alpha > 0 and beta > 0:
        lam = np.random.beta(alpha, beta)
    else:
        lam = 1  # cross-entropy

    batch_size = x.size()[0]
    index = get_random_index_with(targets=targets, batch_size=batch_size, aby=aby_mix)
    # y_uni = torch.unique(y)
    #
    # # TODO实现选择非同标签样本 debug时间 done 如果只有单类样本就直接乱混合吧
    # if aby_mix == "0" or len(y_uni) < 2:
    #     index = torch.randperm(batch_size).cuda() if batch_size > 2 else torch.tensor([1, 0]).cuda()
    #     # index = torch.randperm(batch_size).cuda()
    # elif aby_mix == "-1":  # 不同标签混合
    #     # y_uni = torch.unique(y)
    #     y_dict = {}
    #     for clas in y_uni:  # 为每个类构建随机域，其元素值为标签所在位置
    #         index = torch.arange(batch_size).cuda()
    #         temp = torch.ones_like(y) * clas
    #         mask = y == temp
    #         y_dict[clas.item()] = torch.masked_select(index, mask)
    #         # torch.where(y == temp, -1, index)
    #         # y_dict[clas.item()] = np.delete(.cpu(), -1).cuda()  # 随机域对应位置
    #     index = torch.empty(0).cuda()
    #     for idx, label in enumerate(y):  # 为每个样本挑选随机对象
    #         ablabel_index = y_dict[label.item()][torch.randint(high=len(y_dict[label.item()]), size=(1,))]
    #         index = torch.cat((index, ablabel_index))
    #     index = index.int()
    # else:  # 相同标签混合
    #     # y_uni = torch.unique(y)
    #     y_dict = {}
    #     for clas in y_uni:  # 为每个类构建随机域，其元素值为标签所在位置
    #         index = torch.arange(batch_size).cuda()
    #         temp = torch.ones_like(y) * clas
    #         mask = y != temp
    #         y_dict[clas.item()] = torch.masked_select(index, mask)
    #         # y_dict[clas.item()] = np.delete(torch.where(y != temp, -1, index).cpu(), -1).cuda()  # 随机域对应位置
    #     index = torch.empty(0).cuda()
    #     for idx, label in enumerate(y):  # 为每个样本挑选随机对象
    #         ablabel_index = y_dict[label.item()][torch.randint(high=len(y_dict[label.item()]), size=(1,))]
    #         index = torch.cat((index, ablabel_index))
    #     index = index.int()
    y_a, y_b = targets, targets[index]
    mixed_x = lam * x + (1 - lam) * x[index]
    y_b = lam * targets + (1 - lam) * targets[index]
    if device is not None:
        y_a.to(device)
        y_b.to(device)
    return mixed_x, y_a, y_b, lam, save_mask


def get_xishu_mask(feature):
    mask = (feature > 0).int()
    return mask


def get_allxishu_mask(feature, targets):
    mask = (feature > 0).int()
    batch_size = feature.size()[0]
    index = get_random_index_with(targets=targets, batch_size=batch_size, aby=-1)
    mask = mask | mask[index]
    return mask, index


def get_ab_xishu_mask(feature, targets, random=False):
    mask = (feature > 0).int()
    if random:
        batch_size = feature.size()[0]
        index = get_random_index_with(targets=targets, batch_size=batch_size, aby=-1)
        mask = mask[index]
    return mask, index


def get_xishupatch_mask(feature, targets, gamma_adj, xishu=True, andor=True):
    higher = get_xishu_mask(feature=feature)  # 价值高的部分的掩码
    patches = get_batch_patch_mask(feature=feature, gamma_adj=gamma_adj)
    if not xishu:
        higher = 1 - higher
    if andor:
        higher = higher & patches.int()
    else:
        higher = higher | patches.int()

    return higher


def get_Uxishu_mask(feature, targets, index=None):
    mask = (feature > 0).int()
    batch_size = feature.size()[0]
    if index is None:  # torch.gt(feature, torch.mean(feature, dim=1, keepdim=True)).int()
        index = get_random_index_with(targets=targets, batch_size=batch_size, aby=-1)
    mask = mask & mask[index]
    return mask, index
    # 裁剪剪枝的目的是防止攻击者从梯度强行推理出特征，或是从特征空间中搜索，那么破坏哪些梯度可以防止推理搜索呢？
    # 续：在先验已经被数据增强保护住的前提下，难道要看cor？
    # 推理：先验使特征空间搜索从全空间缩小到标签空间，提高重建率。数增使特征不再局限于标签空间，破坏先验效果，减低重建率。
    # 目标：数增只是破坏先验，增大了梯度推特征难度，需要一个针对数增特点的裁剪或剪枝策略，防止梯度推特征。
    # 推理1：剪枝用于单样本梯度，裁剪用于批样本梯度。单样本能确定危险梯度（即更容易推理出特征的梯度），而批样本只能确定危险梯度数量。
    # 补充：选择剪枝区域的时候最好不要透露更多隐私信息，比如混合块和稀疏情况。
    # 猜测：特征有重复也有激活之分，

def get_all_Uxishu_mask(feature):
    mask = torch.gt(feature, torch.mean(feature, dim=1, keepdim=True)).int()
    # mask = (feature > 0).int()
    # batch_size = feature.size()[0]
    # mask = mask & mask
    return mask


def get_orxishu_mask(feature, targets):
    mask = (feature > 0).int()
    batch_size = feature.size()[0]
    index = get_random_index_with(targets=targets, batch_size=batch_size, aby=-1)
    mask = mask | mask[index]
    return mask, index


def get_patch_xishu_mask(feature, targets, gamma=0.1):
    mask = (feature > 0).int()
    batch_size = feature.size()[0]
    size = feature.size(1)
    index = get_random_index_with(targets=targets, batch_size=batch_size, aby=-1)
    mask = mask & mask[index]
    adj_rate = float(torch.mean(torch.sum(mask, dim=1) / size))
    if gamma < adj_rate:
        gamma /= adj_rate
    patch_mask = get_batch_patch_mask(feature, gamma_adj=gamma).int()
    mask = mask & patch_mask
    return mask, index


def get_Utopk_mask(feature, targets, rate=0.1):
    batch_size = feature.size()[0]
    index = get_random_index_with(targets=targets, batch_size=batch_size, aby=-1)
    mask = generate_mask(feature, pick_rate=rate).int()

    mask = mask & mask[index]
    return mask, index


def get_useful_mask(feature, targets, parameters, rate=0.1):
    batch_size = feature.size()[0]
    index = get_random_index_with(targets=targets, batch_size=batch_size, aby=-1)
    useful = torch.sum(parameters, dim=1)
    pass


def get_xor_xishu_mask(feature, targets):
    mask = (feature > 0).int()
    batch_size = feature.size()[0]
    index = get_random_index_with(targets=targets, batch_size=batch_size, aby=-1)
    mask = (mask != mask[index]).int()
    return mask, index


def get_unknown_Udomain_mask(feature, targets, rate=4):
    targets = torch.argmax(targets, dim=1)
    mask = get_domain_mask(feature=feature, rate=rate)
    if targets is None:
        # 使用 torch.any() 函数对每一行的 mask 取并集
        row_mask = torch.any(mask, dim=0)

        # 使用 torch.unsqueeze() 函数将 row_mask 扩展为 (batchsize, 512) 的大小
        output_mask = row_mask.unsqueeze(0).expand(feature.size(0), -1).int()
        return output_mask
    unique_targets = torch.unique(targets)  # 获取 targets 中的不重复元素
    size = unique_targets.size()[0]

    # 创建一个空的 mask，形状为 (64, 512)
    output_mask = torch.zeros_like(mask)
    if size == 1:
        # random_indices = torch.randperm(size) if size > 2 else torch.tensor([1, 0]).cuda()
        for index in range(size):
            # 找到 targets 中等于当前不重复元素的位置
            target_indices = torch.where(targets == unique_targets[index])[0]
            # rander_indices = torch.where(targets == unique_targets[random_indices[index]])[0]
            # 将对应位置的 mask 取并集
            output_mask[target_indices] = torch.any(mask[target_indices], dim=0).int()
        # size = unique_targets.size()[0]
        # random_indices = torch.randperm(size) if size > 2 else torch.tensor([1, 0]).cuda()
        return output_mask
    random_indices = torch.randperm(size) if size > 2 else torch.tensor([1, 0]).cuda()
    for index in range(size):
        # 找到 targets 中等于当前不重复元素的位置
        target_indices = torch.where(targets == unique_targets[index])[0]
        rander_indices = torch.where(targets == unique_targets[random_indices[index]])[0]
        # 将对应位置的 mask 取并集
        output_mask[target_indices] = torch.any(mask[rander_indices], dim=0).int()
    # size = unique_targets.size()[0]
    # random_indices = torch.randperm(size) if size > 2 else torch.tensor([1, 0]).cuda()
    return output_mask


def get_batch_patch_mask(feature, gamma_adj=0):
    block_size = 1
    if gamma_adj is None:
        gamma_adj = 0.9 * feature.shape[-1] ** 2 / (block_size ** 2 * (feature.shape[-1] - block_size + 1) ** 2)
    p = torch.ones_like(feature) * gamma_adj
    m_i_j = torch.bernoulli(p)
    return m_i_j


def get_halfxishu_rate(feature):
    mask = 1 - get_xishu_mask(feature=feature)
    round(float(torch.sum(mask.int()) / mask.numel()), 4)
    # TODO


def get_patch_mask(feature, gamma_adj=0.1):
    block_size = 1

    if gamma_adj is None:
        gamma_adj = 0.9 * feature.shape[-1] ** 2 / (block_size ** 2 * (feature.shape[-1] - block_size + 1) ** 2)
    p = torch.ones_like(feature[0]) * gamma_adj
    # For each feature in the feature map, we will sample from Bernoulli(p). If the result of this sampling
    # for feature f_{ij} is 0, then Mask_{ij} = 1. If the result of this sampling for f_{ij} is 1,
    # then the entire square region in the mask with the center Mask_{ij} and the width and height of
    # the square of block_size is set to 0.
    m_i_j = torch.bernoulli(p)
    # mask_shape = len(m_i_j.shape)
    #
    # # after creating the binary Mask. we are creating the binary Mask created for first sample as a pattern
    # # for all samples in the minibatch as the PatchUp binary Mask. to do so, we can just expnand the pattern
    # # created for the first sample.
    # m_i_j = m_i_j.expand(feature.size(0), m_i_j.size(0), m_i_j.size(1), m_i_j.size(2))
    #
    # # following line provides the continues blocks that should be altered with PatchUp denoted as holes here.
    # holes = F.max_pool2d(m_i_j, self.kernel_size, self.stride, self.padding)
    #
    # # following line gives the binary mask that contains 1 for the features that should be remain unchanged and 1
    # # for the features that lie in the continues blocks that selected for interpolation.
    # # 意思是holes的东西要混合，mask的东西要保留
    # mask = 1 - holes
    return m_i_j


def get_one_topk_mask(feature, pruning_rate=99):
    device = 'cuda:0' if feature.get_device() == 0 else None
    # thresh = np.percentile(feature.flatten().cpu().detach().numpy(), pruning_rate)
    # # 根据阈值生成mask，即小于阈值的位置取0，大于等于阈值的位置取1
    # mask = np.where(abs(feature.cpu()) < thresh, 0, 1).astype(np.float32)
    size = feature.size(1)  # 获取长度
    # if size == 1:
    #     size = feature.size(1)
    if 1 < pruning_rate < 100:
        pruning_rate /= 100
    # 要剪出来的元素个数
    pruned_num = int(size * pruning_rate)

    mask = torch.zeros_like(feature)
    _, index = torch.topk(feature, pruned_num)
    mask[:, index] = 1

    return mask.to(device)


def get_topk_mask(feature, pruning_rate=35):
    # t 是形状为 (batchsize, 512) 的 tensor
    # rate 表示前多少个大的数值被设为 1，范围为 [0,100]
    if 0 < pruning_rate < 1:
        pruning_rate = int(pruning_rate * 100)

    # 计算 t 的前 k 个元素的阈值
    k = int(pruning_rate / 100 * feature.size(1))
    topk_values, _ = torch.topk(feature, k, dim=1)
    threshold = topk_values[:, -1]

    # 根据阈值生成掩码
    mask = torch.ge(feature, threshold.unsqueeze(1)).float()
    return mask


def get_similar_mask(feature, targets, rate=0, aby_mix=-1):
    # 注意，这里返回的是前rate%最相似的部分！
    batch_size = targets.size()[0]
    device = 'cuda:0' if feature.get_device() == 0 else None
    # indices = torch.randperm(batch_size, device=device) if batch_size > 2 else torch.tensor([1, 0], device=device)
    indices = get_random_index_with(targets=targets, batch_size=batch_size, aby=aby_mix).to(torch.int64)

    dis = -torch.abs(feature - feature[indices])
    if rate == 0:
        rate = get_Uxishu_rate(feature, indices=indices)
    mask = generate_mask(tensor=dis, pick_rate=rate)
    return mask, indices


def get_Uxishu_rate(feature, indices=None, targets=None):
    mask = (feature > 0).int()
    feature_size = feature.size()[1]
    if indices is None:
        indices = get_random_index_with(targets=targets, batch_size=targets.size()[0], aby=-1).to(torch.int64)
    rate = torch.mean(torch.sum(mask & mask[indices], dim=1) / feature_size)
    return rate


def get_xishu_rate(feature):
    mask = (feature > 0).int()
    feature_size = feature.size()[1]
    rate = torch.mean(torch.sum(mask, dim=1) / feature_size)
    return rate


def get_abUxishu_mask(feature, targets):
    mask = (feature > 0).int()
    batch_size = feature.size()[0]
    index = get_random_index_with(targets=targets, batch_size=batch_size, aby=-1)
    label_intersection = torch.zeros_like(mask[0])  # 创建一个全为0的张量，形状与特征相同
    for l in range(targets.shape[1]):
        label_feature = mask[targets[:, l] == 1]  # 获取标签为l的样本的特征部分
        label_intersection = label_intersection.logical_or(label_feature.sum(dim=0) > 0)  # 更新交集

    return mask, index


def generate_mask(tensor, pick_rate=0.15):
    if pick_rate > 1:
        pruning_rate = pick_rate / 100
    device = 'cuda:0' if tensor.get_device() == 0 else None
    batch_size = tensor.size()[0]
    # 排除为0的位置
    tensor = torch.where(tensor != 0, tensor, torch.tensor(float('-inf')))
    k = int(pick_rate * tensor.size(1))

    # 计算0的数量，排除掉
    # zero_num = torch.sum((tensor == 0).int())/batch_size
    # k = int(pick_rate * tensor.size(1)) + int(zero_num)

    # 使用 torch.topk() 函数找到最大的 k 个元素的索引
    _, indices = torch.topk(tensor, k)

    # 创建一个大小为 (512,) 的全零掩码
    mask = torch.zeros(tensor.size()).cuda()

    # 将最大的 k 个元素对应的位置置为 1
    mask.scatter_(1, indices, 1)

    return mask.to(device)


def generate_batch_mask(tensor, pick_rate=0.15):
    batch_size = tensor.size()[0]
    device = 'cuda:0' if tensor.get_device() == 0 else None
    # 计算最大的 rate% 的元素个数
    k = int(pick_rate * tensor.size(0))

    # 使用 torch.topk() 函数找到最大的 k 个元素的索引
    _, indices = torch.topk(tensor, k)

    # 创建一个大小为 (512,) 的全零掩码
    mask = torch.zeros(tensor.size()).cuda()

    # 将最大的 k 个元素对应的位置置为 1
    mask.scatter_(1, indices, 1)

    return mask.to(device)


def get_sot_mask(model, x, y, main_y=None, feature=None, pruning_rate=90):
    # 与之前不同，要处理一个batch的特征混合
    # y = to_one_hot(y.clone())
    if main_y is None:
        main_y = torch.argmax(torch.sum(y, dim=0), dim=0)
    target_mask = (torch.argmax(y, dim=1) == main_y)
    x = x[target_mask]
    # 获得模型的特征向量
    device = 'cuda:0' if x.get_device() == 0 else None
    if feature is None:
        feature_fc1_graph = model.get_feature()  # (batch, feature_size)
    else:
        feature_fc1_graph = feature.clone()[target_mask]
    feature_fc1_graph = feature_fc1_graph[target_mask]
    feature_fc1_graph.requires_grad_(True)
    masks = None
    # 初始化目标特征向量和输入样本的中间变量的形状和类型
    for data, feature in zip(x, feature_fc1_graph):
        # 每次处理一个样本x(1, 3, 32, 32)和其对应feature(1, 512)，生成mask(1, 512)，与masks拼接
        # feature = feature.unsqueeze(0)
        data = data.unsqueeze(0)
        deviation_f1_target = torch.zeros_like(feature)
        deviation_f1_x_norm = torch.zeros_like(feature)
        # 对一个样本的每个特征向量进行遍历
        for f in range(feature.size(0)):
            # 目标向量对该特征向量取值为1，其他取值为0
            deviation_f1_target[f:] = 1
            # 反向传播目标向量，获得输入样本的中间变量
            feature.backward(deviation_f1_target, retain_graph=True)
            # 初始化输入样本的中间变量，即上文所说的d_i，由于求的是向量的范数，因此需要对输入样本进行展平后再求范数
            deviation_f1_x = data
            deviation_f1_x_norm[f:] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1) / (
                    feature.data[f] + 0.1)
            # 清除梯度
            model.zero_grad()
            # 目标向量记得要重置
            deviation_f1_target[f:] = 0
        # 对所有特征向量的输入样本中间变量的范数进行求和并排序，得到剪枝的阈值
        deviation_f1_x_norm_sum = feature
        thresh = np.percentile(deviation_f1_x_norm_sum.flatten().cpu().detach().numpy(), pruning_rate)
        # 根据阈值生成mask，即小于阈值的位置取0，大于等于阈值的位置取1
        mask = torch.from_numpy(np.where(abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)).to(
            device)
        if masks is not None:
            masks = torch.cat((masks, mask.unsqueeze(0)))
        else:
            masks = mask.unsqueeze(0)
        # 返回mask
    model.detach_feature()  # 清理feature
    #正常数据增强训练后，用剪枝或裁剪处理梯度？算结合吗？
    #sot目的是保护特征从而保护样本，原因是特征纠缠不足，数据增强提高纠缠，但是猜出掩码后有被还原的风险，sot同样是猜出掩码后会被还原。
    #有发现先验威胁的了，有和sot一样极大化还原差异，极小化防御前后信息的，
    return torch.prod(masks, dim=0)

def Soteria(model, x, feature=None, pruning_rate=90):
    device = 'cuda:0' if x.get_device() == 0 else None
    if feature is None:
        feature_fc1_graph = model.get_ori_feature()  # (batch, feature_size)
    else:
        feature_fc1_graph = feature.clone()
    pruning_rate = pruning_rate / 100 if pruning_rate >= 1 else pruning_rate
    masks = []
    feature_fc1_graph = feature_fc1_graph
    feature_fc1_graph.requires_grad_(True)
    deviation_f1_x_norm_sum = None
    # 初始化目标特征向量和输入样本的中间变量的形状和类型
    for data, feature in zip(x, feature_fc1_graph):  # (32,3,32,32)和(32,512)
        # 每次处理一个样本x(1, 3, 32, 32)和其对应feature(1, 512)，生成mask(1, 512)，与masks拼接
        # feature = feature.unsqueeze(0)
        data = data.unsqueeze(0)
        deviation_f1_target = torch.zeros_like(feature)
        deviation_f1_x_norm = torch.zeros_like(feature)
        for f in range(feature.size(0)):  # range(0, 2304) 2304=feature_fc1_graph.shape
            deviation_f1_target[f] = 1  # 把deviation_f1_target的第f列设置成1 （为啥
            # loss调用后向传播，所以feature_fc1_graph = loss
            feature.backward(deviation_f1_target, retain_graph=True)
            # 进行一次backward之后，各个节点的值会清除，这样进行第二次backward会报错，如果加上retain_graph=True后,可以再来一次backward。
            #
            deviation_f1_x = data  # 不知道为什么都还是0
            # deviation_f1_x是被攻击图片ground_truth的样本数据，(1, 3, 32, 32)
            deviation_f1_x_norm[f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1) / (
                    feature.data[f] + 0.1)
            # 相当于把梯度平铺了，然后计算平铺层的Frobenius范数(全元素平方和)，然后除以loss+0.1  为啥
            model.zero_grad()  # 清零模型梯度
            deviation_f1_target[f] = 0  # 归零第一行设为1的值
        # 对所有特征向量的输入样本中间变量的范数进行求和并排序，得到剪枝的阈值
        # deviation_f1_x_norm(512,1)是该样本特征的贡献度(提取样本的信息量)
        if deviation_f1_x_norm_sum is None:
            deviation_f1_x_norm_sum = deviation_f1_x_norm
        else:
            deviation_f1_x_norm_sum += deviation_f1_x_norm
        thresh = np.percentile(deviation_f1_x_norm_sum.flatten().cpu().detach().numpy(), pruning_rate)
        # 根据阈值生成mask，即小于阈值的位置取0，大于等于阈值的位置取1
        mask = torch.from_numpy(np.where(abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)).to(
            device)
        masks.append(mask)

    sums = torch.stack(masks).sum(dim=0)

    # 计算要获取的元素数量
    top_k = int(pruning_rate * sums.size(0))

    # 找到最大的前top_k个元素的位置
    _, indices = torch.topk(sums, top_k)

    # 创建位置掩码
    mask = torch.zeros_like(sums)
    mask[indices] = 1

    return mask


def Soteria4batch(model, x, feature=None, pruning_rate=90):
    # 废弃，不知道怎么改
    device = 'cuda:0' if x.get_device() == 0 else None
    if feature is None:
        feature_fc1_graph = model.get_ori_feature()  # (batch, feature_size)
    else:
        feature_fc1_graph = feature
    pruning_rate = pruning_rate / 100 if pruning_rate >= 1 else pruning_rate
    masks = []
    feature_fc1_graph.requires_grad_(True)

    deviation_f1_x_norm_sum = None
    # 初始化目标特征向量和输入样本的中间变量的形状和类型

    # 每次处理一个样本x(1, 3, 32, 32)和其对应feature(1, 512)，生成mask(1, 512)，与masks拼接
    # feature = feature.unsqueeze(0)
    deviation_f1_target = torch.zeros_like(feature)
    deviation_f1_x_norm = torch.zeros_like(feature)

    for f in range(feature.size(1)):  # range(0, 2304) 2304=feature_fc1_graph.shape
        deviation_f1_target[:, f] = 1  # 把deviation_f1_target的第f列设置成1 （为啥
        # loss调用后向传播，所以feature_fc1_graph = loss
        feature.backward(deviation_f1_target, retain_graph=True)
        # 进行一次backward之后，各个节点的值会清除，这样进行第二次backward会报错，如果加上retain_graph=True后,可以再来一次backward。
        #
        deviation_f1_x = x.grad.data  # 不知道为什么都还是0
        # deviation_f1_x是被攻击图片ground_truth的样本数据，(1, 3, 32, 32)
        deviation_f1_x_norm[:, f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1) / (
                feature_fc1_graph.data[:, f] + 0.1)
        # 相当于把梯度平铺了，然后计算平铺层的Frobenius范数(全元素平方和)，然后除以loss+0.1  为啥
        model.zero_grad()  # 清零模型梯度
        deviation_f1_target[:, f] = 0  # 归零第一行设为1的值
    # 对所有特征向量的输入样本中间变量的范数进行求和并排序，得到剪枝的阈值
    # deviation_f1_x_norm(512,1)是该样本特征的贡献度(提取样本的信息量)
    if deviation_f1_x_norm_sum is None:
        deviation_f1_x_norm_sum = deviation_f1_x_norm.sum(axis=0)
    else:
        deviation_f1_x_norm_sum += deviation_f1_x_norm
    thresh = np.percentile(deviation_f1_x_norm_sum.flatten().cpu().detach().numpy(), pruning_rate)
    # 根据阈值生成mask，即小于阈值的位置取0，大于等于阈值的位置取1
    mask = torch.from_numpy(np.where(abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)).to(
        device)

    return mask
def get_my_sot_mask(model, x, y, indice, feature=None, pruning_rate=90):
    # 与之前不同，要处理混合后的
    # y = to_one_hot(y.clone())
    # 获得模型的特征向量
    device = 'cuda:0' if x.get_device() == 0 else None
    if feature is None:
        feature_fc1_graph = model.get_ori_feature()  # (batch, feature_size)
    else:
        feature_fc1_graph = feature.clone()
    feature_fc1_graph = feature_fc1_graph
    feature_fc1_graph.requires_grad_(True)
    masks = None
    deviation_f1_x_norm_sum = None
    # 初始化目标特征向量和输入样本的中间变量的形状和类型
    for data, feature in zip(x, feature_fc1_graph):
        # 每次处理一个样本x(1, 3, 32, 32)和其对应feature(1, 512)，生成mask(1, 512)，与masks拼接
        # feature = feature.unsqueeze(0)
        data = data.unsqueeze(0)
        deviation_f1_target = torch.zeros_like(feature)
        deviation_f1_x_norm = torch.zeros_like(feature)
        # 对一个样本的每个特征向量进行遍历
        # for f in range(feature.size(0)):
        #     # 目标向量对该特征向量取值为1，其他取值为0
        #     deviation_f1_target[f:] = 1
        #     # 反向传播目标向量，获得输入样本的中间变量
        #     feature.backward(deviation_f1_target, retain_graph=True)
        #     # 初始化输入样本的中间变量，即上文所说的d_i，由于求的是向量的范数，因此需要对输入样本进行展平后再求范数
        #     deviation_f1_x = data
        #     deviation_f1_x_norm[f:] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1) / (
        #             feature.data[f] + 0.1)
        #     # 清除梯度
        #     model.zero_grad()
        #     # 目标向量记得要重置
        #     deviation_f1_target[f:] = 0 开始对每个特征进行反向传播，计算贡献度
        for f in range(feature.size(0)):  # range(0, 2304) 2304=feature_fc1_graph.shape
            deviation_f1_target[f] = 1  # 把deviation_f1_target的第f列设置成1 （为啥
            # loss调用后向传播，所以feature_fc1_graph = loss
            feature.backward(deviation_f1_target, retain_graph=True)
            # 进行一次backward之后，各个节点的值会清除，这样进行第二次backward会报错，如果加上retain_graph=True后,可以再来一次backward。
            #
            deviation_f1_x = data  # 不知道为什么都还是0
            # deviation_f1_x是被攻击图片ground_truth的样本数据，(1, 3, 32, 32)
            deviation_f1_x_norm[f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1) / (
                    feature.data[f] + 0.1)
            # 相当于把梯度平铺了，然后计算平铺层的Frobenius范数(全元素平方和)，然后除以loss+0.1  为啥
            model.zero_grad()  # 清零模型梯度
            model.zero_grad()
            deviation_f1_target[f] = 0  # 归零第一行设为1的值
        # 对所有特征向量的输入样本中间变量的范数进行求和并排序，得到剪枝的阈值
        # deviation_f1_x_norm(512,1)是该样本特征的贡献度(提取样本的信息量)
        if deviation_f1_x_norm_sum is None:
            deviation_f1_x_norm_sum = deviation_f1_x_norm
        else:
            deviation_f1_x_norm_sum += deviation_f1_x_norm
    thresh = np.percentile(deviation_f1_x_norm_sum.flatten().cpu().detach().numpy(), pruning_rate)
    # 根据阈值生成mask，即小于阈值的位置取0，大于等于阈值的位置取1
    mask = torch.from_numpy(np.where(abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)).to(
        device)
    # if masks is not None:
    #     masks = torch.cat((masks, mask.unsqueeze(0)))
    # else:
    #     masks = mask.unsqueeze(0)
        # 返回mask
    model.detach_feature()  # 清理feature
    # 正常数据增强训练后，用剪枝或裁剪处理梯度？算结合吗？
    # 可行性矛盾，因为sot必须在梯度清理后计算提取幅度，而混合要在训练中进行
    # sot目的是保护特征从而保护样本，原因是特征纠缠不足，数据增强提高纠缠，但是猜出掩码后有被还原的风险，sot同样是猜出掩码后会被还原。
    #
    # 现状：有发现先验威胁的了，有和sot一样极大化还原差异，极小化防御前后信息的，

    #直接先混合后sot试试
    # masks = masks + masks[indice]

    return torch.prod(mask, dim=0)


def get_domain_mask(feature, rate=4):
    # if distribution is None:
    #     # mean = torch.mean(feature, dim=0)
    #     # std = torch.std(feature, dim=0)
    #     select_num = int(0.35 * feature.size(1))
    #     value, index = torch.topk(feature, select_num, dim=1)
    #     threshold = value[:, -1]
    #     # 根据阈值生成掩码
    #     mask = torch.ge(feature, threshold.unsqueeze(1)).float()
    # else:
    mean = torch.mean(feature, dim=1)
    std = torch.std(feature, dim=1)
    threshold = mean + rate * std
    mask = (feature > threshold.unsqueeze(1)).int()
    # 报错，这里的domain判定条件是该样本特征的均值与标准差，所以每个样本特征的mask不同 done 每个样本自己统计自己选domain
    # if torch.sum(mask) <= 256:
    #     return get_xishu_mask(feature=feature)
    return mask


def get_ab_Udomain_mask(feature, targets, rate=4):
    batch_size = feature.size()[0]
    index = get_random_index_with(targets=targets, batch_size=batch_size, aby=-1)
    mean = torch.mean(feature, dim=1)
    std = torch.std(feature, dim=1)
    threshold = mean + rate * std
    mask = (feature > threshold.unsqueeze(1)).int()
    mask = mask & mask[index]
    # 报错，这里的domain判定条件是该样本特征的均值与标准差，所以每个样本特征的mask不同 done 每个样本自己统计自己选domain
    # if torch.sum(mask) <= 256:
    #     return get_xishu_mask(feature=feature)
    return mask, index


def get_ab_anddomain_mask(feature, targets, rate=4):
    batch_size = feature.size()[0]
    index = get_random_index_with(targets=targets, batch_size=batch_size, aby=-1)
    mean = torch.mean(feature, dim=1)
    std = torch.std(feature, dim=1)
    threshold = mean + rate * std
    mask = (feature > threshold.unsqueeze(1)).int()
    mask = mask | mask[index]
    return mask, index


def get_mix_domain_mask(feature):
    pass


def get_Udomain_mask(feature, targets=None):
    targets = torch.argmax(targets, dim=1)
    mask = get_domain_mask(feature=feature, rate=1)
    if targets is None:
        # 使用 torch.any() 函数对每一行的 mask 取并集
        row_mask = torch.any(mask, dim=0)

        # 使用 torch.unsqueeze() 函数将 row_mask 扩展为 (batchsize, 512) 的大小
        output_mask = row_mask.unsqueeze(0).expand(feature.size(0), -1).int()
        return output_mask
    unique_targets = torch.unique(targets)  # 获取 targets 中的不重复元素
    size = unique_targets.size()[0]
    # 创建一个空的 mask，形状为 (64, 512)
    output_mask = torch.zeros_like(mask)
    # random_indices = torch.randperm(size) if size > 2 else torch.tensor([1, 0]).cuda()
    for index in range(size):
        # 找到 targets 中等于当前不重复元素的位置
        target_indices = torch.where(targets == unique_targets[index])[0]
        # rander_indices = torch.where(targets == unique_targets[random_indices[index]])[0]
        # 将对应位置的 mask 取并集
        output_mask[target_indices] = torch.any(mask[target_indices], dim=0).int()
    # size = unique_targets.size()[0]
    # random_indices = torch.randperm(size) if size > 2 else torch.tensor([1, 0]).cuda()
    return output_mask


def get_bigger_mean_mask(feature):
    mean = torch.mean(feature, dim=1)
    threshold = mean
    mask = (feature > threshold.unsqueeze(1)).int()
    return mask


def get_UBM_mask(feature, targets):
    mask = get_bigger_mean_mask(feature=feature)
    unique_targets = torch.unique(targets)  # 获取 targets 中的不重复元素
    # 创建一个空的 mask，形状为 (64, 512)
    output_mask = torch.zeros_like(mask)
    for target in unique_targets:
        # 找到 targets 中等于当前不重复元素的位置
        target_indices = torch.where(targets == target)[0]
        # 将对应位置的 mask 取并集
        output_mask[target_indices] = torch.any(mask[target_indices], dim=0).int()
    return output_mask


def get_momen_domain_mask(domain_count, feature):
    if domain_count is None:
        domain_count = torch.ones_like(feature[1])
    mask = get_domain_mask(feature=feature)
    domain_count = domain_count + torch.sum(mask, dim=0)
    # domain_pred = torch.sigmoid(domain_count)  # 输入过大全是1了
    domain_mask = generate_mask(domain_count, 0.3)

    return domain_mask, domain_count


def get_waht_mask(feature):
    return torch.zeros_like(feature)


def adjust_lam(x, new_feature, targets, index, fc, lam):
    adj_lam = torch.zeros_like(targets)
    with torch.no_grad():
        rate = fc(new_feature) / fc(x)
        rate = torch.relu(rate)
    lama = rate[torch.arange(64), torch.argmax(targets, dim=1)]
    lamb = rate[torch.arange(64), torch.argmax(targets[index], dim=1)]

    lama[lama == 0], lamb[lamb == 0] = lam, (1 - lam)
    lama[lama > 1], lamb[lamb > 1] = 1, 1
    adj_lam = torch.cat((lama.unsqueeze(dim=1), lamb.unsqueeze(dim=1)), dim=1)

    return adj_lam  # (64, 2)


def is_mixlabel_leak(tensor1, tensor2):
    abs_tensor1 = torch.abs(tensor1)
    abs_tensor2 = torch.abs(tensor2)

    # 找到绝对值最大的两个元素的索引
    _, indices1 = torch.topk(abs_tensor1, 2, dim=0)
    _, indices2 = torch.topk(abs_tensor2, 2, dim=0)

    if tensor2[indices2[1]] == 0:
        pass
    # 判断两个张量绝对值最大的两个元素的位置是否相同
    # is_same_position = torch.equal(indices1[0], indices2[0]) and torch.equal(indices1[1], indices2[1])
    return torch.equal(indices1[0], indices2[0]), torch.equal(indices1[1], indices2[1])
    # print(is_same_position)


def to_one_hot(inp, num_classes=10):
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


def ref_cor(delta_W, ground, mix=2):
    # 计算每个梯度的二范数
    # norms = torch.norm(delta_W, dim=1)
    norms = torch.abs(torch.sum(delta_W, dim=1))

    # 计算二范数的均值和标准差
    mean_norm = torch.mean(norms)
    std_norm = torch.std(norms)
    # _, S = torch.topk(norms, mix, dim=0)

    # 定义异常值的阈值为均值加上 3 倍标准差
    threshold = mean_norm + 0.5 * std_norm

    # 挑选出异常值的索引
    outliers = torch.where(norms > threshold)[0]

    # 输出局部训练类集 S

    # 输出线性尺度的训练数据特征 rB
    rB = delta_W[outliers]

    double = torch.stack((torch.mean(rB, dim=0), torch.mean(ground, dim=0)), dim=1).T
    return outliers, torch.abs(torch.corrcoef(double)[1][0])


# fed -0.9271   patch1 -0.7988  patch-1 -0.6219 my_1_xishu -0.4373  my_ab_xishu -0.2804

def ref_cor_y(delta_W, ground, y, mix=2):
    if y.size().__len__() != 1:
        y = torch.argmax(y, dim=1)
    # 计算每个梯度的二范数
    norms = torch.norm(delta_W, dim=1)

    # 计算二范数的均值和标准差
    # mean_norm = torch.mean(norms)
    # std_norm = torch.std(norms)
    mean_norm = torch.mean(norms)
    std_norm = torch.std(norms)
    # _, S = torch.topk(norms, mix, dim=0)

    # 定义异常值的阈值为均值加上 3 倍标准差
    threshold = mean_norm + 0.5 * std_norm
    # S = torch.where(norms > threshold)[0]
    _, S = torch.topk(norms, mix, dim=0)

    # 他是先推标签再用梯度求特征相关   混合后也不影响推理  为什么
    #

    # 输出线性尺度的训练数据特征 rB
    rB = delta_W[S]
    result = 0
    for index in range(S.size()[0]):
        this_y = y == S[index]
        double = torch.stack((rB[index], torch.mean(ground[this_y], dim=0)), dim=1).T
        result += torch.abs(torch.corrcoef(double)[1][0])
    # double = torch.stack((torch.mean(rB, dim=0), torch.mean(ground, dim=0)), dim=1).T
    result = result / S.size()[0]
    # if torch.isnan(result):
    # print("keke")
    # pass
    #     result = 1.0
    result[torch.isnan(result)] = 0.0
    return S, result


def dp(model, norm_clip, eps0, delta):
    # loss_func = torch.nn.CrossEntropyLoss()
    # train_idx = list(train_idx)
    # ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
    # net.train()
    # optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    # # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)
    # epoch_loss = []
    # eps = args.eps0 / args.epochs
    # sensitivity = args.norm_clip
    # noise_scale = np.sqrt(2 * np.log(1.25 / args.delta)) * sensitivity / eps
    # start_net = copy.deepcopy(net)
    # for iter in range(args.local_ep):
    #     batch_loss = []
    #     for batch_idx, (images, labels) in enumerate(ldr_train):
    #         images, labels = images.to(args.device), labels.to(args.device)
    #         optimizer.zero_grad()
    #         log_probs = net(images)
    #         loss = loss_func(log_probs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         # scheduler.step()
    #         batch_loss.append(loss.item())
    #     epoch_loss.append(sum(batch_loss) / len(batch_loss))

    device = 'cuda:0'

    delta_net = torch.copy.deepcopy(model)
    w_start = start_net.state_dict()
    w_delta = delta_net.state_dict()
    for i in w_delta.keys():
        w_delta[i] -= w_start[i]
    delta_net.load_state_dict(w_delta)
    with torch.no_grad():
        delta_net = clip_parameters(norm_clip, delta_net)
    delta_net = add_noise(norm_clip, eps0=eps0, delta=delta, device=device, model=model)
    w_delta = delta_net.state_dict()
    for i in w_start.keys():
        w_start[i] += w_delta[i].to(w_start[i].dtype)

    # self.add_noise(net_noise)
    # w_delta = clip_and_add_noise_our(w_delta,noise_multiplier,args)
    # for i in w_start.keys():
    #     w_start[i] += w_delta[i].to(w_start[i].dtype)

    return w_start, sum(epoch_loss) / len(epoch_loss)


def clip_parameters(norm_clip, net):
    for k, v in net.named_parameters():
        v /= max(1, v.norm(2).item() / norm_clip)
    return net


def add_noise(norm_clip, eps0, delta, device, model, lr=0.01):
    sensitivity = cal_sensitivity_up(lr, norm_clip)
    with torch.no_grad():
        for k, v in model.named_parameters():
            noise = Gaussian_Simple(epsilon=eps0, delta=delta, sensitivity=sensitivity, size=v.shape)
            noise = torch.from_numpy(noise).to(device)
            v += noise
    return model


def cal_sensitivity_up(lr, clip):
    return 2 * lr * clip


def Gaussian_Simple(epsilon, delta, sensitivity, size):
    noise_scale = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    return np.random.normal(0, noise_scale, size=size)
