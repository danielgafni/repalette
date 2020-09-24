import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch


class PaletteExctractorNet(pl.LightningModule):
    """
    Don't use this. Bad batchnorms. TODO: improve
    """
    def __init__(self, hparams):
        super(PaletteExctractorNet, self).__init__()

        # hyperparameters format for `pytorch-lightning` to load the model from checkpoints later!
        self.hparams = hparams
        dropout = self.hparams["dropout"]

        # self.c1 = nn.Conv2d(3, 3 * 16, 1, 2, 1)
        # self.batchnorm1 = nn.BatchNorm2d(3 * 16)
        # self.act1 = nn.ReLU()
        # self.c2 = nn.Conv2d(3 * 8, 3 * 8 * 8, 1, 2, 1, 3)
        # self.batchnorm2 = nn.BatchNorm2d(3 * 8 * 8)
        # self.act2 = nn.ReLU()
        # self.c3 = nn.Conv2d(3 * 8 * 8, 3 * 8 * 8 * 8, 1, 2, 1, 3)
        # self.batchnorm3 = nn.BatchNorm2d(3 * 8 * 8 * 8)
        # self.act3 = nn.ReLU()
        # self.final_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.act2 = nn.ReLU()
        self.linear_features = nn.Linear(128 * 128 * 3, 128*3)
        self.act3 = nn.ReLU()
        self.palette_mapper = nn.Linear(128*3, 18)
        self.sigmoid = nn.Sigmoid()

        self.loss_fn = F.mse_loss

    def forward(self, x):
        # x = self.c1(x)
        # x = self.act1(x)
        # x = self.final_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.act2(x)
        x = self.linear_features(x)
        x = self.act3(x)
        x = self.palette_mapper(x).view(-1, 3, 1, 6)
        x = self.sigmoid(x)

        return x

    def training_step(self, batch, batch_nb):
        x, y_true = batch
        y_true = y_true.view(-1, 3, 1, 6)
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y_true)
        result = pl.TrainResult(minimize=loss)
        result.log("train_loss", loss, prog_bar=True)
        return result

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y_true = batch
        y_true = y_true.view(-1, 3, 1, 6)
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y_true)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, prog_bar=True)
        return result

    def validation_end(self, outputs):
        # OPTIONAL
        avg_val_loss = outputs["val_loss"].mean()
        result = pl.EvalResult(checkpoint_on=avg_val_loss)
        result.log('avg_val_loss', avg_val_loss, prog_bar=True)
        return result

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y_true = batch
        y_true = y_true.view(-1, 3, 1, 6)
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y_true)
        result = pl.EvalResult()
        result.log('test_loss', loss, prog_bar=True)
        return result

    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = outputs["test_loss"].mean().item()
        result = pl.EvalResult()
        result.log('test_loss', avg_loss, prog_bar=True)
        return result

    def configure_optimizers(self):
        # REQUIRED
        return torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])


class PaletteExctractorConvNet(pl.LightningModule):
    """
    Don't use this. Bad batchnorms. TODO: improve
    """
    def __init__(self, hparams):
        super(PaletteExctractorNet, self).__init__()

        # hyperparameters format for `pytorch-lightning` to load the model from checkpoints later!
        self.hparams = hparams
        dropout = self.hparams["dropout"]

        self.c1 = nn.Conv2d(3, 3 * 8, 1, 2, 1)
        self.batchnorm1 = nn.BatchNorm2d(3 * 8)
        self.act1 = nn.ReLU()
        self.c2 = nn.Conv2d(3 * 8, 3 * 8 * 8, 1, 2, 1, 3)
        self.batchnorm2 = nn.BatchNorm2d(3 * 8 * 8)
        self.act2 = nn.ReLU()
        self.c3 = nn.Conv2d(3 * 8 * 8, 3 * 8 * 8 * 8, 1, 2, 1, 3)
        self.batchnorm3 = nn.BatchNorm2d(3 * 8 * 8 * 8)
        self.act3 = nn.ReLU()
        self.final_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.act4 = nn.ReLU()
        self.linear_features = nn.Linear(3 * 8 * 8 * 8, 100)
        self.palette_mapper = nn.Linear(100, 18)
        self.sigmoid = nn.Sigmoid()

        self.loss_fn = F.mse_loss

    def forward(self, x):
        x = self.c1(x)
        x = self.batchnorm1(x)
        x = self.act1(x)
        x = self.c2(x)
        x = self.batchnorm2(x)
        x = self.act2(x)
        x = self.c3(x)
        x = self.batchnorm3(x)
        x = self.act3(x)
        x = self.final_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.act4(x)
        x = self.linear_features(x)
        x = self.palette_mapper(x).view(-1, 3, 1, 6)
        x = self.sigmoid(x)

        return x

    def training_step(self, batch, batch_nb):
        x, y_true = batch
        y_true = y_true.view(-1, 3, 1, 6)
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y_true)
        result = pl.TrainResult(minimize=loss)
        result.log("train_loss", loss, prog_bar=True)
        return result

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y_true = batch
        y_true = y_true.view(-1, 3, 1, 6)
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y_true)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, prog_bar=True)
        return result

    def validation_end(self, outputs):
        # OPTIONAL
        avg_val_loss = outputs["val_loss"].mean()
        result = pl.EvalResult(checkpoint_on=avg_val_loss)
        result.log('avg_val_loss', avg_val_loss, prog_bar=True)
        return result

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y_true = batch
        y_true = y_true.view(-1, 3, 1, 6)
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y_true)
        result = pl.EvalResult()
        result.log('test_loss', loss, prog_bar=True)
        return result

    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = outputs["test_loss"].mean().item()
        result = pl.EvalResult()
        result.log('test_loss', avg_loss, prog_bar=True)
        return result

    def configure_optimizers(self):
        # REQUIRED
        return torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])

# AttributeError: module 'pytorch_lightning' has no attribute 'data_loader' ???

#     @pl.data_loader
#     def train_dataloader(self):
#         return DataLoader(train_dataset, batch_size=32)

#     @pl.data_loader
#     def val_dataloader(self):
#         # OPTIONAL
#         # can also return a list of val dataloaders
#         return DataLoader(val_dataset, batch_size=32)

#     @pl.data_loader
#     def test_dataloader(self):
#         # OPTIONAL
#         # can also return a list of test dataloaders
#         return DataLoader(test_dataset, batch_size=32)

