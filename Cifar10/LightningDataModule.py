import lightning as L
from Cifar10.Constants import PATH_DATASETS, BATCH_SIZE, NUM_CLASSES
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split


class LightningDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = PATH_DATASETS):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.dims = (3, 32, 32)
        self.num_classes = NUM_CLASSES

    def prepare_data(self):
        # download
        CIFAR10(root=self.data_dir, train=True,
                download=True, transform=self.transform)
        CIFAR10(root=self.data_dir, train=False,
                download=True, transform=self.transform)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            trainset = CIFAR10(root=self.data_dir, train=True,
                               download=False, transform=self.transform)
            self.data_train, self.data_val = random_split(trainset, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.data_test = CIFAR10(root=self.data_dir, train=False,
                                     download=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2)
