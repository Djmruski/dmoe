import numpy as np

class Rehearsal:
    def __init__(self):
        self.class_means = {}
        self.class_covariances = {}

    def add_task(self, task_data):
        features, labels = task_data['x'], task_data['y']

        classes = np.unique(labels)
        for cls in classes:
            class_features = features[labels == cls]
            self.class_means[cls] = np.mean(class_features, axis=0)
            self.class_covariances[cls] = np.cov(class_features, rowvar=False)
    
    def generate_data(self, n_samples):
        rehearsal_features = []
        rehearsal_labels = []

        for cls, mean in self.class_means.items():
            cov = self.class_covariances[cls]
            cls_samples = np.random.multivariate_normal(mean, cov, n_samples)
            # rehearsal_features

    def create_dataloader(self, new_task_data):
        return

    