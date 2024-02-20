import numpy as np

class Rehearsal:
    def __init__(self, data_set, path='saves'):
        self.save_path = '/'.join([path, data_set, 'rehearsal_data.npz'])
        self.class_means = []
        self.class_covariances = []

    def add_task(self, task_data):
        features, labels = task_data['x'], task_data['y']
        classes = np.unique(labels)
        for class_id in classes:
            class_features = features[labels == class_id]
            self.class_means.append(np.mean(class_features, axis=0))
            self.class_covariances.append(np.cov(class_features, rowvar=False))
    
    def generate_data(self, n_samples_per_class):
        rehearsal_features = []
        rehearsal_labels = []

        for class_id, mean in enumerate(self.class_means):
            cov = self.class_covariances[class_id]
            samples = np.random.multivariate_normal(mean, cov, n_samples_per_class)
            rehearsal_features.append(samples)
            rehearsal_labels.append(np.full(n_samples_per_class, class_id))
        
        return np.concatenate(rehearsal_features, dtype=np.float32), np.concatenate(rehearsal_labels, dtype=np.float32)

    def save(self):
        np.savez(
            self.save_path,
            class_means = self.class_means,
            class_covariances = self.class_covariances
        )

    def load(self):
        data = np.load(self.save_path)
        self.class_means = data['class_means']
        self.class_covariances = data['class_covariances']