import numpy as np

class Rehearsal:

    """
    A class for managing rehearsal data for the DyTox model. Rehearsal data is stored and managed 
    through a Gaussian Distribution. It stores class means and covariances for each task's data to 
    generate pseudo-rehearsal data that helps in mitigating catastrophic forgetting by simulating 
    previous tasks' data distribution.

    Attributes:
        save_path (str): Path where rehearsal data (class means and covariances) is saved.
        class_means (list): A list containing the means of features for each class.
        class_covariances (list): A list of numpy covariance matrices for the features of each class.

    Args:
        data_set (str): Name of the dataset for which rehearsal data is being prepared.
        path (str, optional): Base directory path where rehearsal data will be saved. Defaults to 'saves'.
    """

    def __init__(self, data_set, path='saves'):
        self.save_path = '/'.join([path, data_set, 'rehearsal_data.npz'])
        self.class_means = []
        self.class_covariances = []


    def add_task(self, task_data):
        """
        Processes a new task's data to compute and store the class means and covariances.

        Args:
            task_data (dict): A dictionary containing 'x' and 'y' keys for features and labels.
                              'x' should be an array of features, and 'y' should be an array of labels.
        """
        features, labels = task_data['x'], task_data['y']
        classes = np.unique(labels)
        for class_id in classes:
            class_features = features[labels == class_id]
            self.class_means.append(np.mean(class_features, axis=0))
            self.class_covariances.append(np.cov(class_features, rowvar=False))


    def generate_data(self, n_samples_per_class):
        """
        Generates pseudo-rehearsal data based on the stored class means and covariances.

        Args:
            n_samples_per_class (int): Number of synthetic samples to generate per class.

        Returns:
            tuple: A tuple containing two numpy arrays: 
                   - The first array is the concatenated synthetic features for all classes.
                   - The second array is the concatenated labels for the synthetic features.
        """
        rehearsal_features = []
        rehearsal_labels = []

        for class_id, mean in enumerate(self.class_means):
            cov = self.class_covariances[class_id]
            samples = np.random.multivariate_normal(mean, cov, n_samples_per_class)
            rehearsal_features.append(samples)
            rehearsal_labels.append(np.full(n_samples_per_class, class_id))
        
        return np.concatenate(rehearsal_features, dtype=np.float32), np.concatenate(
            rehearsal_labels, dtype=np.float32)


    def save(self):
        """
        Saves the class means and covariances to the specified save path using numpy's npz format.
        """
        np.savez(
            self.save_path,
            class_means = self.class_means,
            class_covariances = self.class_covariances
        )


    def load(self):
        """
        Loads class means and covariances from the specified save path into the Rehearsal object.
        """
        data = np.load(self.save_path)
        self.class_means = data['class_means']
        self.class_covariances = data['class_covariances']