import torch
import torch.optim as optim
import pandas as pd
from torch.optim.lr_scheduler import LambdaLR

from models.custom_models import *
from data.challenge_dataset import ChallengeDataset
from trainer.trainer_both import TrainerBoth
from trainer.trainer_single import TrainerSingle
from exporter.exporter import Exporter


'''
    .. Despite provided possibility to train both models simultaneously by using train_both_models(), it is recommended 
        to train them separately using train_single_model() method. Possibility of separate training is the main benefit 
        of presented approach. It allows to train each model with different parameters, e.g. train for different number 
        of epochs.
    
    .. Important parameters to tune when training a model: 
        -batch_size, 
        -num_epochs, 
        -loader_random_state, 
        -split_ratio, 
        -Learning Rate: [base_lr_crack, base_lr_inactive, lr_schedule_crack, lr_schedule_inactive], 
        -optimizer_crack, 
        -optimizer_inactive
        
        Finding the best combination of these parameters is a matter of trial and error. Good combination may improve 
        the model's performance significantly. 
            Note: batch_size is limited by the available GPU memory.
       
    .. LambdaLR is used as a learning rate scheduler. It is defined by a function lr_lambda which computes a 
        multiplicative factor given an integer parameter epoch, which is the current epoch number. In this case, 
        learning rate is a product of base_lr and value from lr_schedule_crack that is chosen based on the current
        epoch number. The same applies to lr_schedule_inactive. 
        lr_schedule_crack/lr_schedule_inactive may vary up, down or be constant throughout the training process.
'''

# Setup model parameters
csv_path = "data/data.csv"
batch_size = 32
# Number of epochs to train for
num_epochs = 100
# How often to save the model state, e.g. every 3rd epoch: save_interval = 3. Can be set to 1 to disable.
save_interval = 100
# Epoch number when to unfreeze the model. Can be set to num_epochs to disable. Note: default models are already
# unfrozen, but you can change that behaviour by calling self.freeze() in the constructor of the model.
unfreeze_epoch = 100
# Save model if F1 score is greater than f1_save_threshold despite falling in save_interval. Can be set to 0 to disable.
f1_save_threshold = 0.85
# Seed for splitting the dataset into train and validation sets
loader_random_state = 32
# Split ratio for train and validation sets, e.g. 80% train and 20% validation: split_ratio = 0.2
split_ratio = 0.2
# Use cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CSV file and split into train and validation sets
df = pd.read_csv(csv_path, delimiter=';')
train_loader, val_loader, pos_weight = ChallengeDataset.set_loaders(df, batch_size=batch_size, split_ratio=split_ratio,
                                                                    random_state=loader_random_state)

# Define submodels
num_classes = 1 # default value since in ensemble model each submodel represents a single class
model_crack = CustomResNet18Model(num_classes).to(device)
model_inactive = CustomResNet50Model(num_classes).to(device)

# Define loss function, optimizer, and learning rate scheduler
criterion_crack = nn.BCEWithLogitsLoss(pos_weight=pos_weight[0])
criterion_inactive = nn.BCEWithLogitsLoss(pos_weight=pos_weight[1])

base_lr_crack = 1
base_lr_inactive = 1
lr_schedule_crack = {(1, 5): 1e-3,
                     (6, 20): 1e-4,
                     (21, 70): 1e-5,
                     (71, 1000): 1e-6}
lr_schedule_inactive = {(1, 5): 1e-3,
                        (6, 20): 1e-4,
                        (21, 70): 1e-5,
                        (71, 1000): 1e-6}

# Define optimizers. Default choice is Adam, but other optimizers can be used as well.
optimizer_crack = optim.Adam(model_crack.parameters(), lr=base_lr_crack)
optimizer_inactive = optim.Adam(model_inactive.parameters(), lr=base_lr_inactive)

# Return the learning rate based on the current epoch
def get_lr(scheduler_epoch, lr_schedule):
    for (start, end), lr_value in lr_schedule.items():
        if start <= scheduler_epoch + 1 <= end:
            return lr_value
    # If epoch is not in any of the ranges, return 1e-8
    return 1e-8

# Create a learning rate scheduler
scheduler_crack = LambdaLR(
    optimizer_crack, lr_lambda=lambda scheduler_epoch: get_lr(scheduler_epoch, lr_schedule_crack))
scheduler_inactive = LambdaLR(
    optimizer_inactive, lr_lambda=lambda scheduler_epoch: get_lr(scheduler_epoch, lr_schedule_inactive))


def print_parameters():
    print("____________________________________________________________\n"
          "PARAMETERS:\n"
          f"\tcsv_path: {csv_path}\n"
          f"\tbatch_size: {batch_size}\n"
          f"\tnum_epochs: {num_epochs}\n"
          f"\tsave_interval: {save_interval}\n"
          f"\tunfreeze_epoch: {unfreeze_epoch}\n"
          f"\tf1_save_threshold: {f1_save_threshold}\n"
          f"\tloader_random_state: {loader_random_state}\n"
          f"\tdevice: {device}\n"
          f"\tmodel_crack: {type(model_crack).__name__}\n"
          f"\t\tnum_classes {model_crack.num_classes}\n"
          f"\tmodel_inactive: {type(model_inactive).__name__}\n"
          f"\t\tnum_classes {model_inactive.num_classes}\n"
          f"\tcriterion_crack: {type(criterion_crack).__name__}\n"
          f"\t\tpos_weight_crack {pos_weight[0]}\n"
          f"\tcriterion_inactive: {type(criterion_inactive).__name__}\n"
          f"\t\tpos_weight_inactive {pos_weight[1]}\n"
          f"\toptimizer_crack: {type(optimizer_crack).__name__}\n"
          f"\toptimizer_inactive: {type(optimizer_inactive).__name__}\n"
          f"\tbase_lr_crack: {base_lr_crack}\n"
          f"\tbase_lr_inactive: {base_lr_inactive}\n"
          f"\tlr_schedule_crack: {lr_schedule_crack}\n"
          f"\tlr_schedule_inactive: {lr_schedule_inactive}\n"
          f"\tsingle_model_prefix: {single_model_prefix}\n")


if __name__ == '__main__':
    '''
        .. Even thought TrainerSingle receives two models, it trains only one of them. The prefix parameter defines
            which model will be trained. The other one will be simply ignored. Same applies to other prefix-dependent
            parameters.
            
        .. On the other hand, in TrainerBoth both models are trained independent of the prefix value.
        
        .. Exporter exports the model to ONNX format. Specify the names of the files from checkpoints dir to be exported
            and the name of the exported model.
    '''

    # Specify the operation you want to perform: "train_single", "train_both" (not recommended), "export"
    operation = "train_both"

    # In case of both models being trained simultaneously, the prefix does not have any effect. Defines single model
    # mode. Can be 'crack' or 'inactive'
    single_model_prefix = 'inactive'




    # Train single model
    if operation == "train_single":
        print_parameters()
        print("Training single model\n")
        trainer_single = TrainerSingle(model_crack, model_inactive, criterion_crack, criterion_inactive,
                                       optimizer_crack,
                                       optimizer_inactive, scheduler_crack, scheduler_inactive, num_epochs, device,
                                       train_loader,
                                       val_loader, save_interval, unfreeze_epoch, f1_save_threshold,
                                       single_model_prefix)
        trainer_single.train_single_model()

    # Or train two models simultaneously (not recommended)
    elif operation == "train_both":
        print_parameters()
        print("Training both models\n")
        trainer_both = TrainerBoth(model_crack, model_inactive, criterion_crack, criterion_inactive, optimizer_crack,
                                   optimizer_inactive, scheduler_crack, scheduler_inactive, num_epochs, device,
                                   train_loader,
                                   val_loader, save_interval, unfreeze_epoch, f1_save_threshold, single_model_prefix)
        trainer_both.train_both_models()

    # Export model
    elif operation == "export":
        print("Exporting combined model\n")
        exporter = Exporter(model_crack, model_inactive)
        # Specify the names of the files from checkpoints dir to be exported and the name of the exported model
        exporter.export("crack_epoch5_f1=0.8211.pth", "inactive_epoch9_f1=0.8065.pth",
                        'combined_model.onnx')
        print("Export completed\n")
