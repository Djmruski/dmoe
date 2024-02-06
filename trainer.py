class Trainer:
    def __init__(self, data, taskcla, clsorder, n_epochs, batch, in_features):
        self.data = data
        self.taskcla = taskcla
        self.clsorder = clsorder
        self.n_epochs = n_epochs
        self.batch = batch
        self.in_features = in_features
    
    def train_loop():
        print()