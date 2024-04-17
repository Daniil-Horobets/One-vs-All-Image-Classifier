import csv


class BaseTrainer:
    def __init__(self, model_crack, model_inactive, criterion_crack, criterion_inactive, optimizer_crack,
                 optimizer_inactive, scheduler_crack, scheduler_inactive, num_epochs, device, train_loader, val_loader,
                 save_interval, unfreeze_epoch, f1_save_threshold, prefix='crack'):
        self.model_crack = model_crack
        self.model_inactive = model_inactive
        self.criterion_crack = criterion_crack
        self.criterion_inactive = criterion_inactive
        self.optimizer_crack = optimizer_crack
        self.optimizer_inactive = optimizer_inactive
        self.scheduler_crack = scheduler_crack
        self.scheduler_inactive = scheduler_inactive
        self.num_epochs = num_epochs
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_interval = save_interval
        self.unfreeze_epoch = unfreeze_epoch
        self.f1_save_threshold = f1_save_threshold
        self.prefix = prefix
        self.log_both_file_path = 'log_file.csv'
        self.epsilon = 1e-15
        if prefix == 'crack':
            self.model = model_crack
            self.optimizer = optimizer_crack
            self.criterion = criterion_crack
            self.scheduler = scheduler_crack
        else:
            self.model = model_inactive
            self.optimizer = optimizer_inactive
            self.criterion = criterion_inactive
            self.scheduler = scheduler_inactive

    def f1_score(self, cm):
        return 2 * cm[1][1] / (2 * cm[1][1] + cm[0][1] + cm[1][0] + self.epsilon)

    def write_to_csv(self, log_row, log_file):
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(log_row)

    def format_matrix(self, matrix):
        max_width = 9
        return '\n'.join(['[' + ' '.join(f'{cell:>{max_width}}' for cell in row) + ']' for row in matrix])