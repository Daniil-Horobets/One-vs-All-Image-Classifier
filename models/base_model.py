import torch.nn as nn


# Base model for all models
class CustomModel(nn.Module):
    def __init__(self, num_classes, model):
        super(CustomModel, self).__init__()
        # Loaded model
        self.model = model
        # Number of classes (outputs)
        self.num_classes = num_classes
        # Freeze state of the model (frozen model is not trained)
        self.freeze_state_model = False
        # Freeze state of the custom output (head) of the model (frozen head is not trained)
        self.freeze_state_head = False

    # Unfreeze all layers of the model
    def unfreeze(self):
        if self.freeze_state_model:
            for param in self.model.parameters():
                param.requires_grad = True
            self.freeze_state_model = False

    # Unfreeze the head of the model
    def unfreeze_head(self):
        if self.freeze_state_head:
            for param in self.model.fc.parameters():
                param.requires_grad = True
            self.freeze_state_head = False

    # Freeze all layers of the model
    def freeze(self):
        if not self.freeze_state_model:
            for param in self.model.parameters():
                param.requires_grad = False
            self.freeze_state_model = True

    # Freeze the head of the model
    def freeze_head(self):
        if not self.freeze_state_head:
            for param in self.model.fc.parameters():
                param.requires_grad = False
            self.freeze_state_head = True

    # Forward pass
    def forward(self, x):
        x = self.model(x)
        return x