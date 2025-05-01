'''
train script for a periodic u net with 2 loss terms in the loss function
best model used BCE_MSE_loss with LOSS_WEIGHT_ONE = 1.0 and LOSS_WEIGHT_TWO = 10.0
'''

from models.unet_pbc_prelu import UNetPBCPrelu
from models.model import CombinedDiceMSELoss, BCE_MSE_loss
from models.dataloader import create_zipstore_lists
from torch.utils.data import DataLoader
from models.dataset import CustomZarrDataset
import albumentations as A
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.plugins.environments import SLURMEnvironment #for slurm requeueing and signal handling
import signal #slurm signal
import torch
import os

#for CUDA devices with tensor cores
torch.set_float32_matmul_precision('medium') #as opposed to "high"

NUM_GPUS = torch.cuda.device_count()
NUM_WORKERS = int(os.getenv("SLURM_CPUS_PER_TASK", "1"))  # Default to 1 if not set
NUM_NODES = int(os.getenv("SLURM_NNODES", "1"))  # Default to 1 if not set
LEARN_RATE = float(os.getenv("LEARN_RATE", "0.0001"))  # Default to 0.0001 if not set
DETECT_ANOMALY = bool(os.getenv("DETECT_ANOMALY", "False"))  # Default to False if not set
MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", "100"))  # Default to 100 if not set

LOSS_WEIGHT_ONE = float(os.getenv("LOSS_WEIGHT_ONE", "1.0"))
LOSS_WEIGHT_TWO = float(os.getenv("LOSS_WEIGHT_TWO", "1.0"))


class LitModelPeriodic(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNetPBCPrelu()
        # self.criterion = CombinedDiceMSELoss(dice_weight=LOSS_WEIGHT_ONE, mse_weight=LOSS_WEIGHT_TWO)
        self.criterion = BCE_MSE_loss(bce_weight=LOSS_WEIGHT_ONE, mse_weight=LOSS_WEIGHT_TWO) #best model used BCE_MSE_loss with LOSS_WEIGHT_ONE = 1.0 and LOSS_WEIGHT_TWO = 10.0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch["input"], batch["target"]

        y_hat = self.model(inputs)

        loss, cell_loss, vegf_loss = self.criterion(y_hat, targets)
        self.log('train_loss', loss, sync_dist=True) 
        self.log('train_cell_loss', cell_loss, sync_dist=True)
        self.log('train_MSE_loss', vegf_loss, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch["input"], batch["target"]
        y_hat = self.model(inputs)
        loss, cell_loss, vegf_loss = self.criterion(y_hat, targets)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_cell_loss', cell_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_MSE_loss', vegf_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('cell/loss', cell_loss/loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('mse/loss', vegf_loss/loss, on_step=False, on_epoch=True, sync_dist=True)
        
        # Only log for the first batch and on the main process in DDP
        if batch_idx == 0 and self.trainer.is_global_zero and self.current_epoch % 40 == 0:
            # Apply sigmoid to the first channel of y_hat (predictions) for visualization
            y_hat[:, 0, :, :] = (torch.sigmoid(y_hat[:, 0, :, :])>0.5).int()

            # Extract the first sample's input and output for logging
            input_sample = inputs[0]  # Shape: [2, H, W]
            output_sample = y_hat[0]  # Shape: [2, H, W], with the first channel sigmoided

            for channel in range(input_sample.shape[0]):
                # Prepare the input channel image
                input_channel = input_sample[channel].unsqueeze(0)  # Add a channel dimension: [1, H, W]
                tag_input = f'Epoch_{self.current_epoch}_Channel_{channel}_Input'
                self.logger.experiment.add_image(tag_input, input_channel, self.current_epoch)

                # Prepare the output channel image
                output_channel = output_sample[channel].unsqueeze(0)  # Add a channel dimension: [1, H, W]
                tag_output = f'Epoch_{self.current_epoch}_Channel_{channel}_Output'
                self.logger.experiment.add_image(tag_output, output_channel, self.current_epoch)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LEARN_RATE)
    
class LitDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, allocated_cores: int, timestep_to_predict: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.allocated_cores = allocated_cores
        self.timestep_to_predict = timestep_to_predict
        self.zipstore_list_train = None
        self.zipstore_list_val = None


    def setup(self, stage=None):
        # runs once per process, needs to be here for each GPU process
        if self.zipstore_list_train is None or self.zipstore_list_val is None:
            self.zipstore_list_train, self.zipstore_list_val = create_zipstore_lists(self.data_dir, train_size=0.8)
            print(f'there are {len(self.zipstore_list_train)} zipstore files in the training set')
            print(f'there are {len(self.zipstore_list_val)} zipstore files in the validation set')

        self.transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ],
            additional_targets={"mask": "image"},
        )

    def train_dataloader(self):
        # DataLoader return for training
        train_dataset = CustomZarrDataset(zipstore_list=self.zipstore_list_train, transform=self.transform, timestep_ahead=self.timestep_to_predict)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.allocated_cores, persistent_workers=True, prefetch_factor=10, pin_memory=True)
        return train_loader
    
    def val_dataloader(self):
        # DataLoader return for validation
        val_dataset = CustomZarrDataset(zipstore_list=self.zipstore_list_train, transform=self.transform, timestep_ahead=self.timestep_to_predict)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.allocated_cores, persistent_workers=True, prefetch_factor=10, pin_memory=True)
        return val_loader

def main():
    # For dataset
    timestep_to_predict = int(os.environ['TIMESTEP_TO_PREDICT']) # number of timesteps ahead to predict

    # For Rivanna
    zipstore_data_directory = os.environ['ZIPSTORE_DATA_DIRECTORY']
    SummaryPath = os.environ['SUMMARY_PATH']
    data_batch_size = int(os.environ['DATA_BATCH_SIZE']) #batch size for training and validation dataloadersdataloaders

    #data batch size 365 for 80gb of a100 gpu memory

    seed_everything(1, workers=True)

    #for debugging

    print('loss function weight one:', LOSS_WEIGHT_ONE)
    print('loss function weight two:', LOSS_WEIGHT_TWO)
    print('learning rate:', LEARN_RATE)

    print('num GPU devices visible = ', torch.cuda.device_count())
    print('batch size:', data_batch_size)
    print('max epochs:', MAX_EPOCHS)
    MASTER_ADDR = os.getenv("MASTER_ADDR")
    MASTER_PORT = os.getenv("MASTER_PORT")
    WORLD_SIZE = os.getenv("WORLD_SIZE")
    RANK = os.getenv("RANK")
    
    print(f"MASTER_ADDR: {MASTER_ADDR}")
    print(f"MASTER_PORT: {MASTER_PORT}")
    print(f"WORLD_SIZE: {WORLD_SIZE}")
    print(f"RANK: {RANK}")
    print(f"workers per task:{NUM_WORKERS}")
    print(f"num nodes requested:{NUM_NODES}")

    data_dir = zipstore_data_directory
    batch_size = data_batch_size
    allocated_workers = NUM_WORKERS
    rivanna_nodes = NUM_NODES
    timestep_ahead = timestep_to_predict

    model = LitModelPeriodic()
    data_module = LitDataModule(data_dir, batch_size, allocated_workers, timestep_ahead)

    # Create a TensorBoard logger
    logger = TensorBoardLogger(SummaryPath, name="model")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',    # Metric to monitor
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,          # Number of best models to save
        mode='min',            # Save models with minimum val_loss
    )

    # Trainer initialization with additional parameters
    trainer = Trainer(
        accelerator='gpu',     # Use GPU
        devices=-1,  # Use all GPU devices
        strategy='ddp',         # Use Data Parallelism
        num_nodes=rivanna_nodes,  # Use multiple nodes
        max_epochs=MAX_EPOCHS,        # Set the max number of epochs
        precision='16-mixed',          # Use FP16 for training
        callbacks=[checkpoint_callback],  # Add the ModelCheckpoint callback
        logger=logger,
        profiler="simple", #for profiling
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)], #for slurm requeueing and signal handling
    )

    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()