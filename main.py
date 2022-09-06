import torch
import torch.utils.data
from torch import optim
import matplotlib.pyplot as plt
import datetime
import os
import numpy as np
from model import vr_model
from data import get_data

seed = 1
torch.manual_seed(seed)

log_interval = 100
testing_frequency = 20
K = 50
learning_rate = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device = ", device)


def train(model, optimizer, epoch, train_loader, model_type, losses, recon_losses, log_p_vals):
    model.train()
    train_loss = 0
    train_recon_loss_MSE = 0
    train_recon_loss_CE = 0
    train_log_p = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        # (B, 1, F1, F2) (e.g.
        data = data.to(device)
        optimizer.zero_grad()

        loss, recon_loss_MSE, recon_loss_CE, log_p = model.compute_loss_for_batch(data, model, model_type)
        loss.backward()
        train_loss += loss.item()
        train_recon_loss_MSE += recon_loss_MSE.item()
        train_recon_loss_CE += recon_loss_CE.item()
        train_log_p += log_p.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t, log_p: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data), log_p.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}, log_p: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset), train_log_p / len(train_loader.dataset)))

    losses.append(train_loss / len(train_loader.dataset))
    recon_losses.append(
        (train_recon_loss_MSE / len(train_loader.dataset), train_recon_loss_CE / len(train_loader.dataset)))
    log_p_vals.append(train_log_p / len(train_loader.dataset))

    return losses, recon_losses, log_p_vals


def test(model, test_loader, model_type, losses, recon_losses, log_p_vals):
    model.eval()
    test_loss = 0
    test_recon_loss_MSE = 0
    test_recon_loss_CE = 0
    test_log_p = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, recon_loss_MSE, recon_loss_CE, log_p = model.compute_loss_for_batch(data, model, model_type, 50,
                                                                                      testing_mode=True)
            test_loss += loss.item()
            test_recon_loss_MSE += recon_loss_MSE.item()
            test_recon_loss_CE += recon_loss_CE.item()
            test_log_p += log_p.item()

            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(recon_batch.shape[0], 1, 28, 28)[:n]])

                f, axarr = plt.subplots(2, n)
                for j in range(n):
                    axarr[0, j].imshow(comparison.cpu()[j, 0], interpolation='nearest', cmap='viridis')
                for j in range(n, n*2):
                    axarr[1, j - n].imshow(comparison.cpu()[j, 0], interpolation='nearest', cmap='viridis')
                plt.show()

    test_loss /= len(test_loader.dataset)
    test_recon_loss_MSE /= len(test_loader.dataset)
    test_recon_loss_CE /= len(test_loader.dataset)
    test_log_p /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    losses.append(test_loss)
    recon_losses.append((test_recon_loss_MSE, test_recon_loss_CE))
    log_p_vals.append(test_log_p)
    return losses, recon_losses, log_p_vals


# Run

def run(model_type, alpha_pos=1, alpha_neg=-1, data_name='SVHN', num_datapoints=None):
    train_losses, test_losses = [], []
    train_recon_losses, test_recon_losses = [], []
    train_log_p_vals, test_log_p_vals = [], []

    model = vr_model(alpha_pos, alpha_neg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, test_loader = get_data(data_name, num_datapoints=num_datapoints)

    os.makedirs('results', exist_ok=True)
    results_path = "./results"
    print(datetime.datetime.now())
    eps = 1e-4
    epoch = 0
    testing_cnt = 0
    while True:
        # stop learning condition 1
        if testing_cnt >= 3 and test_losses[-1] >= test_losses[-2] and test_losses[-2] >= test_losses[-3]:
            break

        # stop learning condition 2
        if len(train_losses) >= 3 and np.abs(train_losses[-1] - train_losses[-2]) <= eps \
                and np.abs(train_losses[-2] - train_losses[-3]) <= eps:
            break

        train_losses, train_recon_losses, train_log_p_vals = train(model, optimizer, epoch, train_loader,
                                                                   model_type, train_losses, train_recon_losses,
                                                                   train_log_p_vals)
        if epoch % testing_frequency == 1:
            test_losses, test_recon_losses, test_log_p_vals = test(model, test_loader,
                                                                   model_type, test_losses, test_recon_losses,
                                                                   test_log_p_vals)
            testing_cnt += 1

        epoch += 1

    print(datetime.datetime.now())
    print("Training finished")

    torch.save(train_losses,
               results_path + "/{}_{}_{}_{}_train_losses.pt".format(model_type, alpha_pos, alpha_neg, data_name))
    torch.save(train_recon_losses,
               results_path + "/{}_{}_{}_{}_train_recon_losses.pt".format(model_type, alpha_pos, alpha_neg, data_name))
    torch.save(train_log_p_vals,
               results_path + "/{}_{}_{}_{}_train_log_p_vals.pt".format(model_type, alpha_pos, alpha_neg, data_name))
    torch.save(test_losses,
               results_path + "/{}_{}_{}_{}_test_losses.pt".format(model_type, alpha_pos, alpha_neg, data_name))
    torch.save(test_recon_losses,
               results_path + "/{}_{}_{}_{}_test_recon_losses.pt".format(model_type, alpha_pos, alpha_neg, data_name))
    torch.save(test_log_p_vals,
               results_path + "/{}_{}_{}_{}_test_log_p_vals.pt".format(model_type, alpha_pos, alpha_neg, data_name))
    torch.save(model.state_dict(),
               results_path + "/{}_{}_{}_{}_model.pt".format(model_type, alpha_pos, alpha_neg, data_name))


def main():
    run('vr_sandwich', alpha_pos=2, alpha_neg=-2, data_name='MNIST', num_datapoints=1000)

if __name__ == "__main__":
    main()