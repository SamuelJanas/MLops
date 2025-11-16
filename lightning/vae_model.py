import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class VAE(pl.LightningModule):
    def __init__(
        self,
        input_channels=3,
        latent_dim=128,
        hidden_dims=[32, 64, 128, 256],
        learning_rate=1e-3,
        kl_weight=0.00025,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate

        # Store original hidden_dims (make a copy to avoid mutation issues)
        encoder_dims = hidden_dims.copy()
        decoder_dims = hidden_dims[::-1]  # Reversed copy for decoder

        # Encoder
        modules = []
        in_channels = input_channels
        for h_dim in encoder_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(encoder_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(encoder_dims[-1] * 4, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, decoder_dims[0] * 4)

        modules = []
        for i in range(len(decoder_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        decoder_dims[i],
                        decoder_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(decoder_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_dims[-1],
                decoder_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(decoder_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(decoder_dims[-1], input_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z):
        result = self.decoder_input(z)
        # Get the first decoder dimension from saved hyperparameters
        decoder_first_dim = self.hparams.hidden_dims[-1]  # This is 256 for default
        result = result.view(-1, decoder_first_dim, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def loss_function(self, recon_x, x, mu, log_var):
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + self.kl_weight * kld_loss, recon_loss, kld_loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon_x, mu, log_var = self(x)
        loss, recon_loss, kld_loss = self.loss_function(recon_x, x, mu, log_var)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kld_loss", kld_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        recon_x, mu, log_var = self(x)
        loss, recon_loss, kld_loss = self.loss_function(recon_x, x, mu, log_var)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kld_loss", kld_loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
