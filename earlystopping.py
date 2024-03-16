import numpy as np
import torch
import copy  # Used for deep copy operation on model weights to store the best model


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    Attributes:
        patience (int): How long to wait after the last time validation loss improved.
        verbose (bool): If True, prints a message for each validation loss improvement.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        restore_best_weights (bool): If True, the model will revert to the state with the best
                                     validation loss observed during training.
        trace_func (function): Function to use for logging progress, default is print.
    """

    def __init__(self, patience=25, delta=0, restore_best_weights=False, verbose=False, trace_func=print):
        """
        Initialises the EarlyStopping callback.

        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            verbose (bool): Enables verbose output.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            restore_best_weights (bool): Whether to restore model weights from the epoch with the best value
                                         of the monitored metric.
            trace_func (function): Function to use for logging progress, default is print.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
        self.restore_best_weights = restore_best_weights
        self.best_model = None  # Variable to store the best model's weights

    def __call__(self, val_loss, model):
        """
        Call method updates the early stopping mechanism.

        Args:
            val_loss (float): The current validation loss.
            model (torch.nn.Module): The model being trained.
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.update_best_model(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.check_early_stop(model)
        else:
            self.best_score = score
            self.update_best_model(val_loss, model)
            self.counter = 0

    def check_early_stop(self, model):
        """Checks if training should be stopped early."""
        if self.counter >= self.patience:
            self.stop = True
            if self.restore_best_weights and self.best_model is not None:
                model.load_state_dict(self.best_model)  # Restore the best model weights

    def update_best_model(self, val_loss, model):
        """Updates the best model if the current model is better."""
        if self.restore_best_weights:
            self.best_model = copy.deepcopy(model.state_dict())  # Make a deep copy of the model
        self.save_checkpoint(val_loss)

    def save_checkpoint(self, val_loss):
        """
        Updates the minimum validation loss and optionally prints a message.

        Args:
            val_loss (float): The current validation loss.
        """
        if self.verbose:
            self.trace_func(f'\tValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        self.val_loss_min = val_loss
