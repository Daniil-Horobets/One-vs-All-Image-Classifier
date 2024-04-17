import csv

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from trainer.base_trainer import BaseTrainer
from tqdm import tqdm



class TrainerSingle(BaseTrainer):
    def __init__(self, model_crack, model_inactive, criterion_crack, criterion_inactive, optimizer_crack,
                 optimizer_inactive, scheduler_crack, scheduler_inactive, num_epochs, device, train_loader, val_loader,
                 save_interval, unfreeze_epoch, f1_save_threshold, prefix='crack'):
        super().__init__(model_crack, model_inactive, criterion_crack, criterion_inactive, optimizer_crack,
                         optimizer_inactive, scheduler_crack, scheduler_inactive, num_epochs, device, train_loader,
                         val_loader, save_interval, unfreeze_epoch, f1_save_threshold, prefix)

    def train_single_model(self):
        print("=========================================\n"
              "================= START =================\n"
              "=========================================")

        with open(f'log_file_{self.prefix}.csv', mode='w', newline='') as file:
            writer = csv.DictWriter(file, delimiter=';', fieldnames=[
                'epoch', f'train_loss_{self.prefix}', f'val_loss_{self.prefix}',
                f'train_TN_{self.prefix}', f'train_FP_{self.prefix}', f'train_FN_{self.prefix}', f'train_TP_{self.prefix}',
                f'val_TN_{self.prefix}', f'val_FP_{self.prefix}', f'val_FN_{self.prefix}', f'val_TP_{self.prefix}'
            ])
            writer.writeheader()

        for epoch in range(self.num_epochs):

            # Training
            cm_train, train_loss_epoch = self.train_epoch_single_model(epoch)

            # Validation
            cm_val, val_loss_epoch = self.validate_epoch_single_model()

            # Compute F1 score
            f1_score = self.f1_score(cm_val)

            # Log and save
            self.log_single(cm_train, cm_val, epoch, f1_score, train_loss_epoch, val_loss_epoch)
            self.save_single(epoch, f1_score)

        print("========================================\n"
              "================ FINISH ================\n"
              "========================================")

    def train_epoch_single_model(self, epoch):

        # Unfreeze the model after unfreeze_epoch
        if self.model.freeze_state_model == True and epoch >= self.unfreeze_epoch:
            self.model.unfreeze()

        # Set model to training mode
        self.model.train()

        # Initialize confusion matrix and loss
        cm_train = np.zeros((2, 2), dtype=int)  # 2 classes: actual and predicted
        train_loss_epoch = 0

        for images, labels in tqdm(self.train_loader, desc="Training", bar_format="{desc}: {percentage:3.0f}%|{bar}|"):

            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)

            # Calculate loss
            if self.prefix == 'crack':
                train_loss = self.criterion(outputs, labels[:, 0].unsqueeze(1))
            else:
                train_loss = self.criterion(outputs, labels[:, 1].unsqueeze(1))

            train_loss_epoch += train_loss.item()

            # Backward pass
            train_loss.backward()

            # Update weights
            self.optimizer.step()

            # Compute training confusion matrix
            predictions = torch.sigmoid(outputs) > 0.5
            if self.prefix == 'crack':
                cm_train += confusion_matrix(labels[:, 0].cpu().numpy(), predictions.cpu().numpy())
            else:
                cm_train += confusion_matrix(labels[:, 1].cpu().numpy(), predictions.cpu().numpy())

            # Update learning rate
        self.scheduler.step()
        train_loss_epoch /= len(self.train_loader)
        return cm_train, train_loss_epoch

    def validate_epoch_single_model(self):
        # Set model to evaluation mode
        self.model.eval()

        # Initialize confusion matrix and loss
        cm_val = np.zeros((2, 2), dtype=int)
        val_loss_epoch = 0
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating",
                                       bar_format="{desc}: {percentage:3.0f}%|{bar}|"):
                images, labels = images.to(self.device), labels.to(self.device)

                # Make predictions
                outputs = self.model(images)

                # Calculate loss
                if self.prefix == 'crack':
                    val_loss = self.criterion(outputs, labels[:, 0].unsqueeze(1))
                else:
                    val_loss = self.criterion(outputs, labels[:, 1].unsqueeze(1))

                val_loss_epoch += val_loss.item()

                # Compute validation confusion matrix
                predictions = torch.sigmoid(outputs) > 0.5
                if self.prefix == 'crack':
                    cm_val += confusion_matrix(labels[:, 0].cpu().numpy(), predictions.cpu().numpy())
                else:
                    cm_val += confusion_matrix(labels[:, 1].cpu().numpy(), predictions.cpu().numpy())

            val_loss_epoch /= len(self.val_loader)
        return cm_val, val_loss_epoch

    def save_single(self, epoch, f1_score):
        # Save the model if it has good F1 score
        if f1_score > self.f1_save_threshold:
            torch.save(self.model.state_dict(),
                       f"checkpoints/{self.prefix}_epoch{epoch + 1}_f1={f1_score:.4f}.pth")
        # Save the model every save_interval epoch
        if (epoch + 1) % self.save_interval == 0:
            torch.save(self.model.state_dict(), f"checkpoints/{self.prefix}_epoch{epoch + 1}.pth")

    def log_single(self, cm_train, cm_val, epoch, f1_score, train_loss_epoch, val_loss_epoch):
        print(
            f"Epoch {epoch + 1}/{self.num_epochs} | Trin_Loss_{self.prefix}: {train_loss_epoch:.5f}, Val_Loss_{self.prefix}: {val_loss_epoch:.5f}, "
            f"LR_{self.prefix}: {self.scheduler.get_last_lr()[0]:.1e}, F1 Score: {f1_score:.3f}")
        # Print header and aligned matrices
        print(f"\t\t\t CM (Train) - {self.prefix}|  CM (Valid) - {self.prefix}")
        for line in zip(self.format_matrix(cm_train).split('\n'), self.format_matrix(cm_val).split('\n')):
            print(f"\t\t\t{line[0]} | {line[1]}")
        print("\n")
        log_row = [epoch,
                   train_loss_epoch,
                   val_loss_epoch,

                   cm_train[0][0], cm_train[0][1],
                   cm_train[1][0], cm_train[1][1],

                   cm_val[0][0], cm_val[0][1],
                   cm_val[1][0], cm_val[1][1]]
        self.write_to_csv(log_row, f'log_file_{self.prefix}.csv')
