import lightning as L
from Cifar10.LightningDataModule import LightningDataModule
from Cifar10.LightningModule import LightningModule
from lightning.pytorch.loggers import WandbLogger
import wandb
from Cifar10.Constants import MAX_EPOCHS

model = LightningModule()
dm = LightningDataModule()

wandb_logger = WandbLogger(project='wandb-cifar10', job_type='train')

trainer = L.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="auto",
    devices=1,
    logger=wandb_logger
)

trainer.fit(model, dm)

trainer.test()

wandb.finish()
