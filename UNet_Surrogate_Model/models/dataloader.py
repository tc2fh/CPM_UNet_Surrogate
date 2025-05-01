from torch.utils.data import DataLoader
from models.dataset import CustomZarrDataset
import albumentations as A
import os
import glob


def create_zipstore_lists(zipstore_directory,train_size=0.8):
    '''
    from the zipstore_directory, create a list of all the zipstore files and 
    divide them into train and validation sets
    '''

    zipstore_list = glob.glob(os.path.join(zipstore_directory, '*.zarr.zip'))

    #approximately 80% of the data is used for training, 20% for validation
    train_size = int(train_size * len(zipstore_list))
    val_size = len(zipstore_list) - train_size
    zipstore_list_train = zipstore_list[:train_size]
    zipstore_list_val = zipstore_list[train_size:]
    return zipstore_list_train, zipstore_list_val


#not using these get_dataloader functions anymore, setting them manually in the lightning module

def get_dataloaders(zipstore_list_train, zipstore_list_val, transform, timestep_to_predict, batch_size=32, shuffle=True, num_workers=16):
    train_dataset = CustomZarrDataset(zipstore_list=zipstore_list_train, transform=transform, timestep_ahead=timestep_to_predict)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16)

    val_dataset = CustomZarrDataset(zipstore_list=zipstore_list_val, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=16)

    return train_loader, val_loader

def get_train_loader(zipstore_list_train, transform, timestep_to_predict, batch_size=32, shuffle=True, num_workers=16):
    '''creates and returns just the train loader from the zipstore_list_train and transform for pytorch lightning module'''
    train_dataset = CustomZarrDataset(zipstore_list=zipstore_list_train, transform=transform, timestep_ahead=timestep_to_predict)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16)
    return train_loader

def get_val_loader(zipstore_list_val, transform, timestep_to_predict, batch_size=32, shuffle=False, num_workers=16):
    '''creates and returns just the validation loader from the zipstore_list_val and transform for pytorch lightning module'''
    val_dataset = CustomZarrDataset(zipstore_list=zipstore_list_val, transform=transform, timestep_ahead=timestep_to_predict)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=16)
    return val_loader

