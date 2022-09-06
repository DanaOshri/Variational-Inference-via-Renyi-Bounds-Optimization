import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from matplotlib.pyplot import figure


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def RenyiDiv(p, q, alpha=1):
    if (alpha == 1):
        return kl_divergence(p, q)

    if (alpha == 0):
        return -np.log(np.sum(np.where(p != 0, q, 0)))

    return 1.0 / (alpha - 1) * np.log(np.sum(np.where(p != 0, p * np.power(p / q, alpha - 1), 0)))


def plotRenyiDiv():
    plt.rcParams["font.family"] = "Times New Roman"

    x = np.arange(-10, 10, 0.001)
    p = norm.pdf(x, 0, 2)
    p = p / p.sum()
    q = norm.pdf(x, 3, 2)
    q = q / q.sum()
    fig = figure(figsize=(7, 7), dpi=80)

    alphas = np.arange(-20, 20, 0.01)
    RenyiDivs = [RenyiDiv(p, q, alpha) for alpha in alphas]
    plt.plot(alphas, RenyiDivs, c='red', label=r'$D_\alpha(p||q)$', linewidth=4)

    expRenyiDivs = [np.exp(RenyiDiv(p, q, alpha)) for alpha in alphas]
    plt.plot(alphas, expRenyiDivs, c='orange', label=r'$d_\alpha(p||q)$', linewidth=4)

    kl = kl_divergence(p, q)
    plt.plot(alphas, np.repeat(kl, len(alphas)), c='black', label=r'$D_{KL}(p||q)$', linewidth=4)

    plt.xlabel(r'$\alpha$', fontsize=36)
    plt.ylabel("Divergence", fontsize=36)
    plt.ylim(-10, 10)

    ax = plt.subplot(111)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0,
                     box.width, box.height * 1.3])
    fig.tight_layout()

    # Put a legend below current axis
    plt.legend(fontsize=36, loc='upper center', bbox_to_anchor=(0.5, 1.25),
               ncol=3, fancybox=True, shadow=True)
    plt.grid()
    plt.show()


plotRenyiDiv()