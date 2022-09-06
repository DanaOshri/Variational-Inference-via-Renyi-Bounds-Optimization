import torch
from torch import nn
from torch.distributions.multinomial import Multinomial


def elbo(model, x, z, mu, logstd, gamma=1):
    # decoded
    x_hat = model.decode(z)

    BCE = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = 0.5 * torch.sum(torch.exp(logstd) - logstd - 1 + mu.pow(2))
    loss = BCE + gamma * KLD

    return loss


# Variational Renyi

def renyi_bound(method, model, x, z, mu, logstd, alpha, K, testing_mode=False):
    log_q = model.compute_log_probabitility_gaussian(z, mu, logstd)

    log_p_z = model.compute_log_probabitility_gaussian(z, torch.zeros_like(z, requires_grad=False),
                                                       torch.zeros_like(z, requires_grad=False))
    x_hat = model.decode(z)

    log_p = model.compute_log_probabitility_bernoulli(x_hat, x)
    # log_p = model.compute_log_probabitility_gaussian(x_hat, x, torch.zeros_like(x_hat))

    log_w_matrix = (log_p_z + log_p - log_q).view(-1, K) * (1 - alpha)

    loss = 0
    if alpha == 1:
        loss = elbo(model, x, z, mu, logstd)
    if method == 'vr_pos' or method == 'vr_neg':
        loss = compute_MC_approximation(log_w_matrix, alpha, testing_mode)
    elif method == 'vr_ub':
        loss = compute_approximation_for_negative_alpha(log_w_matrix, alpha)
    else:
        print("Invalid value of alpha")

    return loss


def compute_MC_approximation(log_w_matrix, alpha, testing_mode=False):
    log_w_minus_max = log_w_matrix - torch.max(log_w_matrix, 1, keepdim=True)[0]
    ws_matrix = torch.exp(log_w_minus_max)
    ws_norm = ws_matrix / torch.sum(ws_matrix, 1, keepdim=True)

    if not testing_mode:
        sample_dist = Multinomial(1, ws_norm)
        ws_sum_per_datapoint = log_w_matrix.gather(1, sample_dist.sample().argmax(1, keepdim=True))
    else:
        ws_sum_per_datapoint = torch.sum(log_w_matrix * ws_norm, 1)

    if alpha == 1:
        print("Invalid value of alpha")
        return

    ws_sum_per_datapoint /= (1 - alpha)

    loss = -torch.sum(ws_sum_per_datapoint)
    return loss


def compute_approximation_for_negative_alpha(log_w_matrix, alpha):
    norm_log_w_matrix = log_w_matrix.view(log_w_matrix.size(0), -1)

    min_val = norm_log_w_matrix.min(1, keepdim=True)[0]
    max_val = norm_log_w_matrix.max(1, keepdim=True)[0]

    norm_log_w_matrix -= min_val
    norm_log_w_matrix /= max_val
    norm_w_matrix = torch.exp(norm_log_w_matrix)

    approx = norm_w_matrix - 1
    approx *= max_val
    approx += min_val

    ws_norm = approx / torch.sum(approx, 1, keepdim=True)
    ws_sum_per_datapoint = torch.sum(approx * ws_norm, 1)

    ws_sum_per_datapoint /= (1 - alpha)

    loss = -torch.sum(ws_sum_per_datapoint)
    return loss


# Variational Renyi - average approximation for positive and negative α

def renyi_bound_sandwich(model, x, z, mu, logstd, alpha_pos, alpha_neg, K, testing_mode=False):
    loss_pos = renyi_bound('vr_pos', model, x, z, mu, logstd, alpha_pos, K, testing_mode)
    loss_neg = renyi_bound('vr_ub', model, x, z, mu, logstd, alpha_neg, K, testing_mode)

    loss = (loss_neg + loss_pos) / 2
    return loss