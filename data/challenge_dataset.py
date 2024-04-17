import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


'''
    .. Additional data augmentation can be performed in data_transform_train to improve the model performance.
        Consider using random rotation, random cropping, random color jitter, random gaussian blur, etc.
    
    .. To balance the dataset positive weights is used (get_pos_weight function). Also, consider using sampling 
        techniques to balance the dataset, e.g. oversampling.
'''
class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        # pandas dataframe
        self.data = data
        # 'train' or 'val'
        self.mode = mode
        # mean and std values for the dataset Normalization
        self.train_mean = [0.59685254, 0.59685254, 0.59685254]
        self.train_std = [0.16043035, 0.16043035, 0.16043035]

        # Data transformations for training dataset
        self.data_transform_train = transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(self.train_mean, self.train_std)])

        # Data transformations for training dataset
        self.data_transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.train_mean, self.train_std)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = torch.tensor(self.data.iloc[idx, 1:].astype(float).tolist(), dtype=torch.float32)

        if self.mode == "train":
            image = self.data_transform_train(image)
        elif self.mode == "val":
            image = self.data_transform_val(image)

        return image, label

    @staticmethod
    def get_pos_weight(df):
        num_cracks = 0
        num_inactives = 0
        total_len = len(df)
        # To avoid division by zero
        epsilon = 1e-15
        # Calculate the number of cracks and inactives
        for row in df.values.tolist():
            num_cracks += int(row[1])
            num_inactives += int(row[2])
        weight_crack = (total_len - num_cracks) / (num_cracks + epsilon)
        weight_inactive = (total_len - num_inactives) / (num_inactives + epsilon)
        pos_weight = torch.zeros(2)
        pos_weight[0] = torch.tensor(weight_crack)
        pos_weight[1] = torch.tensor(weight_inactive)
        return pos_weight

    @staticmethod
    def set_loaders(df, batch_size=32, split_ratio=0.2, random_state=None):

        # Split the dataset into train and validation sets
        if random_state is not None:
            train_df, val_df = train_test_split(df, test_size=split_ratio, random_state=random_state)
        else:
            train_df, val_df = train_test_split(df, test_size=split_ratio)

        # Create the train and validation datasets
        train_dataset = ChallengeDataset(data=train_df, mode="train")
        val_dataset = ChallengeDataset(data=val_df, mode="val")

        # Create the train and validation dataloaders (be careful with the num_workers parameter, safe choice is 0)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        pos_weight = train_dataset.get_pos_weight(train_df)

        return train_loader, val_loader, pos_weight