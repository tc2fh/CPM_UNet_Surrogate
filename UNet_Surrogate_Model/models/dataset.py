import torch
from torch.utils.data import Dataset
import zarr
import numpy as np

class CustomZarrDataset(Dataset):
    '''
    we've excluded the first 200 timesteps because to allow the CC3D model to relax into a "vascular" state. 
    Images from this timestep onwards are more vascular-like as intended in the biological model
    timestep_ahead is the number of timesteps ahead we want to predict, provides the number of timesteps to skip for input and target pairs
    '''
    def __init__(self, zipstore_list, transform=None, timestep_ahead=100):
        self.zipstore_list = zipstore_list
        self.transform = transform #albumentations transform
        self.timestep_ahead = timestep_ahead
        self.length = len(self.zipstore_list) * (20000-self.timestep_ahead-200) #

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        zipstore_idx = idx // (20000-self.timestep_ahead-200) # Adjusted to exclude first 200 timesteps
        timestep = idx % (20000-self.timestep_ahead-200) + 200 # Adjusted to exclude first 200 timesteps

        zipstore_name = self.zipstore_list[zipstore_idx]
        store = zarr.storage.ZipStore(zipstore_name, mode="r")
        root = zarr.group(store=store, overwrite=False)

        fgbg = np.array(root["fgbg"][timestep])
        vegf = np.array(root["vegf"][timestep])
        fgbg_next = np.array(root["fgbg"][timestep + self.timestep_ahead])
        vegf_next = np.array(root["vegf"][timestep + self.timestep_ahead])

        # stack arrays to produce (Height, Width, Channels)
        input_stack = np.stack([fgbg, vegf], axis=-1)  # shape becomes (256, 256, 2)
        target_stack = np.stack(
            [fgbg_next, vegf_next], axis=-1
        )  # shape becomes (256, 256, 2)

        sample = {"input": input_stack, "target": target_stack}

        store.close()
        if self.transform:
            augmented = self.transform(image=sample["input"], mask=sample["target"])
            sample["input"] = torch.from_numpy(augmented["image"].transpose(2, 0, 1).astype(np.float32))
            sample["target"] = torch.from_numpy(augmented["mask"].transpose(2, 0, 1).astype(np.float32))

        else:
            sample["input"] = torch.from_numpy(sample["input"].transpose(2, 0, 1).astype(np.float32))
            sample["target"] = torch.from_numpy(sample["target"].transpose(2, 0, 1).astype(np.float32))

        return sample
    
class RecursiveZarrDataset(Dataset):
    def __init__(self, zipstore_list, transform=None, timestep_ahead=100, num_steps=5):
        self.zipstore_list = zipstore_list
        self.transform = transform  # albumentations transform
        self.timestep_ahead = timestep_ahead
        self.num_steps = num_steps
        self.length = len(self.zipstore_list) * (20000 - self.timestep_ahead * self.num_steps - 200)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        zipstore_idx = idx // (20000 - self.timestep_ahead * self.num_steps - 200)
        timestep = idx % (20000 - self.timestep_ahead * self.num_steps - 200) + 200

        zipstore_name = self.zipstore_list[zipstore_idx]
        store = zarr.storage.ZipStore(zipstore_name, mode="r")
        root = zarr.group(store=store, overwrite=False)

        fgbg = np.array(root["fgbg"][timestep]) # shape (H, W)
        vegf = np.array(root["vegf"][timestep]) # shape (H, W)
        input_stack = np.stack([fgbg, vegf], axis=-1) # shape becomes (H, W, 2)

        targets = []
        for i in range(1, self.num_steps + 1):
            fgbg_next = np.array(root["fgbg"][timestep + i * self.timestep_ahead]) # shape (H, W)
            vegf_next = np.array(root["vegf"][timestep + i * self.timestep_ahead]) # shape (H, W)
            target_stack = np.stack([fgbg_next, vegf_next], axis=-1) # shape becomes (H, W, 2)
            targets.append(target_stack) # list of shape (H, W, 2)

        targets = np.concatenate(targets, axis=-1)  # shape becomes (H, W, 2 * num_steps)

        store.close()

        if self.transform:
            augmented = self.transform(image=input_stack, mask=targets)
            input_stack = torch.from_numpy(augmented["image"].transpose(2, 0, 1).astype(np.float32))
            targets = torch.from_numpy(augmented["mask"].transpose(2, 0, 1).astype(np.float32))
        else:
            input_stack = torch.from_numpy(input_stack.transpose(2, 0, 1).astype(np.float32))
            targets = torch.from_numpy(targets.transpose(2, 0, 1).astype(np.float32))

        return {"input": input_stack, "target": targets}