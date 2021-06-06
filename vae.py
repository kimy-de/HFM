import torch
import torch.nn as nn
import numpy as np
from mat_to_numpy import load_data


class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.linear1 = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3),nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(3, 10), nn.BatchNorm1d(10), nn.ReLU(), nn.Linear(10, 4))

    def encode(self, x):
        mu = self.linear1(x)
        log_var = self.linear2(x)
        return mu, log_var

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        x = self.reparametrize(mu, logvar)
        out = self.decoder(x)
        return mu, logvar, out


def loss_function(recon_x, x, mu, logvar):
    MSE = reconstruction_function(recon_x, x)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD

def vae(data, c_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = torch.cat([data, c_data], 1).to(device)
    batch_size = 10000
    learning_rate = 1e-4
    min_loss = 100
    model = VariationalAutoEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    M, _ = torch.max(data, dim=0)
    m, _ = torch.min(data, dim=0)
    print("Training VAE")
    for i in range(100000):

        idx = np.random.choice(len(data), batch_size)
        batch_data = data[idx]

        scaled_data = batch_data  # (batch_data - m) / (M - m)
        mu, log_var, outputs = model(scaled_data)
        # outputs = outputs*(M - m) + m
        loss = loss_function(outputs, batch_data, mu, log_var)
        loss.backward()
        optimizer.step()

        if loss.item() < min_loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), './vae.pth')

        if i % 1000 == 0:
            print("[Iteration: %d] training loss: %.3f" % (i, loss.item()))


def data_augmentation(data, c_data, num_samples):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datacat = torch.cat([data, c_data], 1).to(device)

    augmented_samples = num_samples
    idx = np.random.choice(len(data), augmented_samples, replace=False)
    sampled_data = data[idx]
    model = VariationalAutoEncoder().to(device)
    model.load_state_dict(torch.load('./vae.pth'))

    with torch.no_grad():
        #M, _ = torch.max(data, dim=0)
        #m, _ = torch.min(data, dim=0)
        #scaled_data = (sampled_data - m) / (M - m)
        latent_variables = torch.FloatTensor(sampled_data.size()).normal_().to(device)
        augmented_data = model.decoder(latent_variables)
        #augmented_data = (M - m) * augmented_data + m

    data = torch.cat([datacat, augmented_data], 0)
    variables = data[:, :3]
    c_data = data[:, 3:]
    print(f"Generated {num_samples} data")
    return variables.to(device), c_data.to(device)


if __name__ == "__main__":
    data, c_data, _, _, _, _, _, _, _, _ = load_data('./data/Cylinder2D_flower.mat', 157879)
    reconstruction_function = nn.MSELoss()
    vae(data, c_data)

    #num_samples = 10
    #v, c = data_augmentation(data, c_data, num_samples)
