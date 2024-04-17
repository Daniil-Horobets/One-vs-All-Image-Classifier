import csv

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from trainer.base_trainer import BaseTrainer
from tqdm import tqdm


class TrainerBoth(BaseTrainer):
    def __init__(self, model_crack, model_inactive, criterion_crack, criterion_inactive, optimizer_crack,
                 optimizer_inactive, scheduler_crack, scheduler_inactive, num_epochs, device, train_loader, val_loader,
                 save_interval, unfreeze_epoch, f1_save_threshold, prefix='crack'):
        super().__init__(model_crack, model_inactive, criterion_crack, criterion_inactive, optimizer_crack,
                         optimizer_inactive, scheduler_crack, scheduler_inactive, num_epochs, device, train_loader,
                         val_loader, save_interval, unfreeze_epoch, f1_save_threshold, prefix)


    def train_both_models(self):
        print("=========================================\n"
              "================= START =================\n"
              "=========================================")

        with open(self.log_both_file_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, delimiter=';', fieldnames=[
                'epoch', 'train_loss_crack', 'train_loss_inactive', 'val_loss_crack', 'val_loss_inactive',
                'train_TN_crack', 'train_FP_crack', 'train_FN_crack', 'train_TP_crack',
                'train_TN_inactive', 'train_FP_inactive', 'train_FN_inactive', 'train_TP_inactive',
                'val_TN_crack', 'val_FP_crack', 'val_FN_crack', 'val_TP_crack',
                'val_TN_inactive', 'val_FP_inactive', 'val_FN_inactive', 'val_TP_inactive'
            ])
            writer.writeheader()

        for epoch in range(self.num_epochs):

            # Training
            cm_train_crack, cm_train_inactive, train_loss_crack, train_loss_inactive = self.train_epoch_both_models(epoch)

            # Validation
            cm_val_crack, cm_val_inactive, val_loss_crack, val_loss_inactive = self.validate_epoch_both_models()

            # Compute F1 score
            f1_score_crack = self.f1_score(cm_val_crack)
            f1_score_inactive = self.f1_score(cm_val_inactive)

            # Log and save
            self.log_both(cm_train_crack, cm_train_inactive, cm_val_crack, cm_val_inactive, epoch, train_loss_crack,
                          train_loss_inactive, val_loss_crack, val_loss_inactive, f1_score_crack, f1_score_inactive)
            self.save_both(epoch, f1_score_crack, f1_score_inactive)

        print("========================================\n"
              "================ FINISH ================\n"
              "========================================")

    def train_epoch_both_models(self, epoch):

        # Unfreeze the model after unfreeze_epoch
        if self.model_crack.freeze_state_model == True and epoch >= self.unfreeze_epoch:
            self.model_crack.unfreeze()
        if self.model_inactive.freeze_state_model == True and epoch >= self.unfreeze_epoch:
            self.model_inactive.unfreeze()

        # Set models to training mode
        self.model_crack.train()
        self.model_inactive.train()

        # Initialize confusion matrices and loss
        cm_train_crack = np.zeros((2, 2), dtype=int)  # 2 classes: actual and predicted
        cm_train_inactive = np.zeros((2, 2), dtype=int)
        train_loss_crack_epoch = 0
        train_loss_inactive_epoch = 0

        for images, labels in tqdm(self.train_loader, desc="Training", bar_format="{desc}: {percentage:3.0f}%|{bar}|"):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer_crack.zero_grad()
            self.optimizer_inactive.zero_grad()

            # Forward pass
            outputs_crack = self.model_crack(images)
            outputs_inactive = self.model_inactive(images)

            # Calculate loss
            loss_crack = self.criterion_crack(outputs_crack, labels[:, 0].unsqueeze(1))  # 'crack'
            loss_inactive = self.criterion_inactive(outputs_inactive, labels[:, 1].unsqueeze(1))  # 'inactive'

            train_loss_crack_epoch += loss_crack.item()
            train_loss_inactive_epoch += loss_inactive.item()

            # Backward pass
            loss_crack.backward()
            loss_inactive.backward()

            # Update weights
            self.optimizer_crack.step()
            self.optimizer_inactive.step()

            # Compute confusion matrices for training
            predictions_crack = torch.sigmoid(outputs_crack) > 0.5
            predictions_inactive = torch.sigmoid(outputs_inactive) > 0.5
            cm_train_crack += confusion_matrix(labels[:, 0].cpu().numpy(), predictions_crack.cpu().numpy())
            cm_train_inactive += confusion_matrix(labels[:, 1].cpu().numpy(), predictions_inactive.cpu().numpy())

        # Update learning rate
        self.scheduler_crack.step()
        self.scheduler_inactive.step()
        return cm_train_crack, cm_train_inactive, train_loss_inactive_epoch, train_loss_inactive_epoch

    def validate_epoch_both_models(self):
        # Set models to evaluation mode
        self.model_crack.eval()
        self.model_inactive.eval()

        # Initialize confusion matrices and loss
        cm_val_crack = np.zeros((2, 2), dtype=int)
        cm_val_inactive = np.zeros((2, 2), dtype=int)
        val_loss_crack_epoch = 0
        val_loss_inactive_epoch = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating",
                                       bar_format="{desc}: {percentage:3.0f}%|{bar}|"):

                images, labels = images.to(self.device), labels.to(self.device)

                # Make predictions
                outputs_crack = self.model_crack(images)
                outputs_inactive = self.model_inactive(images)

                # Calculate loss
                val_loss_crack = self.criterion_crack(outputs_crack, labels[:, 0].unsqueeze(1))
                val_loss_inactive = self.criterion_inactive(outputs_inactive, labels[:, 1].unsqueeze(1))
                val_loss_crack_epoch += val_loss_crack.item()
                val_loss_inactive_epoch += val_loss_inactive.item()

                # Compute confusion matrices for validation
                predictions_crack = torch.sigmoid(outputs_crack) > 0.5
                predictions_inactive = torch.sigmoid(outputs_inactive) > 0.5

                # Compute validation confusion matrices
                cm_val_crack += confusion_matrix(labels[:, 0].cpu().numpy(), predictions_crack.cpu().numpy())
                cm_val_inactive += confusion_matrix(labels[:, 1].cpu().numpy(), predictions_inactive.cpu().numpy())

        return cm_val_crack, cm_val_inactive, val_loss_crack, val_loss_inactive

    def save_both(self, epoch, f1_score_crack, f1_score_inactive):
        # Save the model if it has good F1 score
        if f1_score_crack > self.f1_save_threshold:
            torch.save(self.model_crack.state_dict(),
                       f"checkpoints/model_crack_epoch{epoch + 1}_f1={f1_score_crack:.4f}.pth")
        if f1_score_inactive >  self.f1_save_threshold:
            torch.save(self.model_inactive.state_dict(),
                       f"checkpoints/model_crack_epoch{epoch + 1}_f1={f1_score_inactive:.4f}.pth")
        # Save the models every save_interval epoch
        if (epoch + 1) % self.save_interval == 0:
            torch.save(self.model_crack.state_dict(), f"checkpoints/model_crack_epoch{epoch + 1}.pth")
            torch.save(self.model_inactive.state_dict(), f"checkpoints/model_inactive_epoch{epoch + 1}.pth")

    def log_both(self, cm_train_crack, cm_train_inactive, cm_val_crack, cm_val_inactive, epoch, train_loss_crack,
                 train_loss_inactive, val_loss_crack, val_loss_inactive, f1_score_crack, f1_score_inactive):
        print(
            f"Epoch {epoch + 1}/{self.num_epochs} | Trin_Loss_crack: {train_loss_crack}, Loss_inactive: {train_loss_inactive}, "
            f"Val_Loss_crack: {val_loss_crack}, Val_Loss_inactive: {val_loss_inactive}, "
            f"LR_crack: {self.scheduler_crack.get_last_lr()[0]}, LR_inactive: {self.scheduler_inactive.get_last_lr()[0]}, "
            f"F1_crack: {f1_score_crack}):.3f, F1_inactive: {f1_score_inactive}):.3f")

        # Print header and matrices
        print("\t\t\t CM (Train) - Crack   |  CM (Train) - Inact   |  CM (Valid) - Crack   |  CM (Valid) - Inact")
        for line in zip(self.format_matrix(cm_train_crack).split('\n'), self.format_matrix(cm_train_inactive).split('\n'),
                        self.format_matrix(cm_val_crack).split('\n'), self.format_matrix(cm_val_inactive).split('\n')):
            print(f"\t\t\t{line[0]} | {line[1]} | {line[2]} | {line[3]}")
        print("\n")
        log_row = [epoch,
                   train_loss_crack,
                   train_loss_inactive,
                   val_loss_crack,
                   val_loss_inactive,

                   cm_train_crack[0][0], cm_train_crack[0][1],
                   cm_train_crack[1][0], cm_train_crack[1][1],

                   cm_train_inactive[0][0], cm_train_inactive[0][1],
                   cm_train_inactive[1][0], cm_train_inactive[1][1],

                   cm_val_crack[0][0], cm_val_crack[0][1],
                   cm_val_crack[1][0], cm_val_crack[1][1],

                   cm_val_inactive[0][0], cm_val_inactive[0][1],
                   cm_val_inactive[1][0], cm_val_inactive[1][1]]
        self.write_to_csv(log_row, self.log_both_file_path)