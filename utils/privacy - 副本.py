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
    if batch_size is None:
        batch_size = targets.size()[0]
    y_uni = torch.unique(targets)
    if aby == "0" or len(y_uni) < 2:
        index = torch.randperm(batch_size).cuda() if batch_size > 2 else torch.tensor([1, 0]).cuda()
        # index = torch.randperm(batch_size).cuda()
    elif aby == "-1":  # 不同标签混合
        # y_uni = torch.unique(y)
        y_dict = {}
        for clas in y_uni:  # 为每个类构建随机域，其元素值为标签所在位置
            index = torch.arange(batch_size).cuda()
            temp = torch.ones_like(targets) * clas
            mask = targets == temp
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
            temp = torch.ones_like(targets) * clas
            mask = targets != temp
            y_dict[clas.item()] = torch.masked_select(index, mask)
            # y_dict[clas.item()] = np.delete(torch.where(y != temp, -1, index).cpu(), -1).cuda()  # 随机域对应位置
        index = torch.empty(0).cuda()
        for idx, label in enumerate(targets):  # 为每个样本挑选随机对象
            ablabel_index = y_dict[label.item()][torch.randint(high=len(y_dict[label.item()]), size=(1,))]
            index = torch.cat((index, ablabel_index))
        index = index.int()
    return index


def patchup_data(x, targets, save_mask, alpha=2.0, beta=2.0, aby_mix=0):
    if alpha > 0 and beta > 0:
        lam = np.random.beta(alpha, beta)
        # if lam < 0.5:
        #     lam = 1 - lam
    else:
        lam = 1  # cross-entropy
    indices = get_random_index_with(targets=targets, aby=aby_mix)

    if isinstance(save_mask, int):
        save_mask = get_dis_mask(feature=x, y_b=indices, save_rate=save_mask)
        # feature_dis = -torch.abs(x - x[indices])
        # if 0 < save_mask < 1:
        #     save_mask *= 100
        # select_num = int(save_mask / 100 * x.size(1))
        # value, index = torch.topk(feature_dis, select_num, dim=1)
        # lam = 1 - ((1 - lam) * torch.sum(value) / torch.sum(feature_dis))
        # threshold = value[:, -1]
        # # 根据阈值生成掩码
        # save_mask = torch.ge(feature_dis, threshold.unsqueeze(1)).float()

    # mask代表要留下来的部分
    unchanged = save_mask * x
    change = 1 - save_mask
    # for batch
    total_feats = x.size(1)
    total_changed_pixels = change[0].sum()
    total_changed_portion = total_changed_pixels / total_feats
    total_unchanged_portion = (total_feats - total_changed_pixels) / total_feats
    target_shuffled_onehot = targets[indices]
    target_reweighted = total_unchanged_portion * targets + lam * total_changed_portion * targets + \
                        target_shuffled_onehot * (1 - lam) * total_changed_portion
    patches = change * x
    patches = patches * lam + patches[indices] * (1 - lam)
    target_b = lam * targets + (1 - lam) * target_shuffled_onehot
    # x = unchanged + patches
    x = unchanged + patches * change
    return x, targets.long(), target_b.long(), target_reweighted.long(), total_changed_portion, save_mask


def mymix_patch(x, targets, save_mask, parameters, alpha=2.0, beta=2.0, aby_mix=0):
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
        change_mask = torch.ge(feature_dis, threshold.unsqueeze(1)).float()

    unchanged = (1 - change_mask) * x
    change = change_mask
    # for batch
    # a1x = torch.mm(change, torch.transpose(torch.gather(parameters.data, 1, targets.unsqueeze(1)).squeeze(1), 0, 1))
    # b1x = torch.mm(unchanged, torch.transpose(torch.gather(parameters.data, 1, targets.unsqueeze(1)).squeeze(1), 0, 1))
    # a1x = torch.sum(change * parameters.data[targets], dim=1, keepdim=True)
    # b1x = torch.sum(unchanged * parameters.data[targets], dim=1, keepdim=True)
    # rate = b1x/a1x
    total_feats = change[0].numel()
    total_changed_pixels = change[0].sum()
    total_changed_portion = total_changed_pixels / total_feats
    total_unchanged_portion = (total_feats - total_changed_pixels) / total_feats
    target_mean = 4.5
    target_shuffled_onehot = targets[indices]
    target_reweighted = total_unchanged_portion * targets + lam * total_changed_portion * targets + \
                        target_mean * (1 - lam) * total_changed_portion
    patch = change * x
    patches = patch * lam + aby_mix * (1 - lam)
    # target_b = lam * targets + (1 - lam) * target_mean
    x = unchanged + patches

    # target recon
    target_b = lam * targets + (1 - lam) * target_shuffled_onehot
    rate = torch.sum((patches - patch) * torch.abs(parameters.data[targets]), dim=1) / \
           torch.sum(x * torch.abs(parameters.data[targets]), dim=1)
    lam *= rate  # lam(1,) rate(64, 1)
    # target_b = rate.detach()
    # target是对应类型(64)，如target[1] = 5， 而混合后只是比例，无法成为对应类型
    return x, targets.long(), target_b, target_reweighted, total_unchanged_portion, 1 - change_mask


def mymix_param(x, param, targets, saves=None, alpha=2.0, beta=2.0, aby_mix=0):
    device = 'cuda:0' if x.get_device() == 0 else None
    if alpha > 0 and beta > 0:
        lam = np.random.beta(alpha, beta)
        if lam < 0.5:
            lam = 1 - lam
    else:
        lam = 1  # cross-entropy

    batch_size = x.size()[0]
    index = get_random_index_with(targets=targets, batch_size=batch_size, aby=aby_mix)
    fc = param.clone()

    change = get_dis_mask(x=x, y_b=index, save_rate=30) if saves is None else saves
    y_a, y_b = targets, targets[index]
    if device is not None:
        y_a.to(device)
        y_b.to(device)

    saves = 1 - change
    xshangfc_a = torch.sum(x * change * fc[y_a], dim=1)
    xxiafc_a = torch.sum(x * saves * fc[y_a], dim=1)
    yijianp_a = (1 - lam) * (xshangfc_a + xxiafc_a) / xshangfc_a

    p_a = (-yijianp_a + 1).unsqueeze(1)

    xshangfc_b = torch.sum(x * change * fc[y_b], dim=1)
    xxiafc_b = torch.sum(x * saves * fc[y_b], dim=1)
    yijianp_b = (1 - lam) * (xshangfc_b + xxiafc_b) / xshangfc_b

    p_b = (-yijianp_b + 1).unsqueeze(1)

    with torch.no_grad():
        changed = x * p_a + x[y_b] * p_b
        # mask = (mask < (x.clone() * link)).int()

    unchanged = x * saves

    # changed = x * lam + x[index] * (1 - lam)
    feat = unchanged + changed

    return feat, y_a, y_b, lam, saves


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
    #
    # patch = change * x
    feat = (x * lam) + (x[index] * (1 - lam))

    # feat = unchanged + change
    y_a, y_b = targets, targets[index]
    # y_b = y_a * lam + y_b * (1 - lam)
    if device is not None:
        y_a.to(device)
        y_b.to(device)
    return feat, y_a, y_b, lam, save_mask


def mixup_data(x, y, alpha=2.0, beta=2.0, aby_mix=0):
    device = 'cuda:0' if x.get_device() == 0 else None
    if alpha > 0 and beta > 0:
        lam = np.random.beta(alpha, beta)
    else:
        lam = 1  # cross-entropy

    batch_size = x.size()[0]
    index = get_random_index_with(targets=y, batch_size=batch_size, aby=aby_mix)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    if device is not None:
        y_a.to(device)
        y_b.to(device)
    return mixed_x, y_a, y_b, lam


def get_param_mask(x, y, fc, alpha=2.0, beta=2.0, aby_mix=0):
    """

    Args:
        x: 输入的待混合数据
        y: 待混合数据的标签
        fc: 用于处理待混合数据的全连接层参数
        alpha: 选择混合比lam的参数1
        beta: 选择混合比lam的参数2
        aby_mix: 是否选择同/异标签样本混合

    Returns:混合后的掩码mask

    """
    device = 'cuda:0' if x.get_device() == 0 else None
    if alpha > 0 and beta > 0:
        lam = np.random.beta(alpha, beta)
    else:
        lam = 1  # cross-entropy

    batch_size = x.size()[0]
    index = get_random_index_with(targets=y, batch_size=batch_size, aby=aby_mix)

    mask = torch.rand_like(x)
    link = torch.mm(fc1, fc2.view(-1, 1))
    f = feature.clone()
    mask = mask < torch.mm(f, link)
    return mask


def get_param_mask(x, y, fc):
    """

    Args:
        x: 输入的待混合数据
        y: 待混合数据的标签
        fc: 用于处理待混合数据的全连接层参数
        alpha: 选择混合比lam的参数1
        beta: 选择混合比lam的参数2
        aby_mix: 是否选择同/异标签样本混合

    Returns:混合后的掩码mask

    """
    device = 'cuda:0' if x.get_device() == 0 else None
    if alpha > 0 and beta > 0:
        lam = np.random.beta(alpha, beta)
    else:
        lam = 1  # cross-entropy

    batch_size = x.size()[0]
    index = get_random_index_with(targets=y, batch_size=batch_size, aby=aby_mix)

    mask = torch.rand_like(x)
    link = torch.mm(fc1, fc2.view(-1, 1))
    f = feature.clone()
    mask = mask < torch.mm(f, link)
    return mask


def get_xishu_mask(feature):
    mask = (feature > 0).int()
    return mask


def get_orxishu_mask(feature):
    mask = (torch.sum((feature > 0).int(), dim=0) > 0).int()
    return mask


def get_dis_mask(feature, y_b, save_rate=65):
    feature_dis = -torch.abs(feature - feature[y_b])
    if 0 < save_rate < 1:
        save_rate *= 100
    select_num = int(save_rate / 100 * feature.size(1))
    value, index = torch.topk(feature_dis, select_num, dim=1)
    threshold = value[:, -1]
    # 根据阈值生成掩码
    change_mask = torch.ge(feature_dis, threshold.unsqueeze(1)).float()
    # return change_mask
    return 1 - change_mask  # TODO 感觉这样才对？



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


def get_sot_mask(model, x, feature=None, pruning_rate=10):
    # 与之前不同，要处理一个batch的特征混合

    # 获得模型的特征向量
    device = 'cuda:0' if x.get_device() == 0 else None
    if feature is None:
        feature_fc1_graph = model.get_feature()  # (batch, feature_size)
    else:
        feature_fc1_graph = feature.clone()
    masks = None
    # 初始化目标特征向量和输入样本的中间变量的形状和类型
    for data, feature in zip(x, feature_fc1_graph):
        # 每次处理一个样本x(1, 3, 32, 32)和其对应feature(1, 512)，生成mask(1, 512)，与masks拼接
        feature = feature.unsqueeze(0)
        data = data.unsqueeze(0)
        deviation_f1_target = torch.zeros_like(feature)
        deviation_f1_x_norm = torch.zeros_like(feature)
        # 对一个样本的每个特征向量进行遍历
        for f in range(feature.size(1)):
            # 目标向量对该特征向量取值为1，其他取值为0
            deviation_f1_target[:, f] = 1
            # 反向传播目标向量，获得输入样本的中间变量
            feature.backward(deviation_f1_target, retain_graph=True)
            # 初始化输入样本的中间变量，即上文所说的d_i，由于求的是向量的范数，因此需要对输入样本进行展平后再求范数
            deviation_f1_x = data
            deviation_f1_x_norm[:, f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1) / (
                    feature.data[:, f] + 0.1)
            # 清除梯度
            model.zero_grad()
            # 目标向量记得要重置
            deviation_f1_target[:, f] = 0
        # 对所有特征向量的输入样本中间变量的范数进行求和并排序，得到剪枝的阈值
        deviation_f1_x_norm_sum = feature.sum(axis=0)
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
    return masks


def get_domain_mask(feature, distribution=None):
    if distribution is None:
        # mean = torch.mean(feature, dim=0)
        # std = torch.std(feature, dim=0)
        select_num = int(0.35 * feature.size(1))
        value, index = torch.topk(feature, select_num, dim=1)
        threshold = value[:, -1]
        # 根据阈值生成掩码
        mask = torch.ge(feature, threshold.unsqueeze(1)).float()
        # TODO 看看效果？做个动量？
    else:
        mean = torch.mean(distribution, dim=1)
        std = torch.std(distribution, dim=1)
        mask = (feature > (mean + 4 * std)).int()
    # TODO 报错，这里的domain判定条件是该样本特征的均值与标准差，所以每个样本特征的mask不同
    # if torch.sum(mask) <= 256:
    #     return get_xishu_mask(feature=feature)
    return mask
