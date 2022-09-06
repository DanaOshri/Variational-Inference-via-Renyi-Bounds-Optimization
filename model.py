from torch.distributions.normal import Normal
from renyi_methods import *

K = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class vr_model(nn.Module):
    def __init__(self, alpha_pos, alpha_neg):
        super(vr_model, self).__init__()

        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc31 = nn.Linear(200, 50)
        self.fc32 = nn.Linear(200, 50)

        self.fc4 = nn.Linear(50, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, 784)

        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg

    def encode(self, x):
        # h1 = F.relu(self.fc1(x))
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # h3 = F.relu(self.fc3(z))
        h3 = torch.tanh(self.fc4(z))
        h4 = torch.tanh(self.fc5(h3))
        return torch.sigmoid(self.fc6(h4))

    def forward(self, x):
        mu, logstd = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logstd)
        return self.decode(z), mu, logstd

    def gaussian_likelihood(self, x_hat, x):
        scale = torch.exp(nn.Parameter(torch.Tensor([0.0])))
        scale = scale.to(device)
        mean = x_hat

        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum()

    def MSE_reconstruction_error(self, x_hat, x):
        return torch.sum(torch.mean(torch.pow(x - x_hat, 2), axis=1))

    def CE_reconstruction_error(self, x_hat, x):
        loss = -torch.sum(x * torch.log(x_hat))
        return loss / x_hat.size(dim=0)

    def compute_log_probabitility_gaussian(self, obs, mu, logstd, axis=1):
        std = torch.exp(logstd)
        n = Normal(mu, std)
        res = torch.mean(n.log_prob(obs), axis)
        return res

    def compute_log_probabitility_bernoulli(self, obs, p, axis=1):
        return torch.sum(p * torch.log(obs) + (1 - p) * torch.log(1 - obs), axis)

    def compute_loss_for_batch(self, data, model, model_type, K=K, testing_mode=False):
        # data = (B, 1, H, W)
        B, _, H, W = data.shape
        x = data.repeat((1, K, 1, 1)).view(-1, H * W)
        mu, logstd = model.encode(x)
        z = model.reparameterize(mu, logstd)

        if model_type == "vae":
            loss = elbo(model, x, z, mu, logstd)
        elif model_type == "vr_pos":
            loss = renyi_bound("vr_pos", model, x, z, mu, logstd, model.alpha_pos, K, testing_mode)
        elif model_type == "vr_neg":
            loss = renyi_bound("vr_neg", model, x, z, mu, logstd, model.alpha_neg, K, testing_mode)
        elif model_type == "vr_ub":
            loss = renyi_bound("vr_ub", model, x, z, mu, logstd, model.alpha_neg,K, testing_mode)
        elif model_type == "vr_sandwich":
            loss = renyi_bound_sandwich(model, x, z, mu, logstd, model.alpha_pos, model.alpha_neg, K, testing_mode)

        # reconstruction loss
        x_hat = model.decode(z)

        recon_loss_MSE = model.MSE_reconstruction_error(x_hat, x)
        recon_loss_CE = model.CE_reconstruction_error(x_hat, x)

        log_p = model.compute_log_probabitility_bernoulli(x_hat, x)
        log_p = torch.sum(torch.mean(log_p.view(B,K), 1))

        return loss, recon_loss_MSE, recon_loss_CE, log_p
