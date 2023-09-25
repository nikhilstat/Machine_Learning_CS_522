import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class Classifier:
    """
    Base Classifier class that initializes training data.
    
    Attributes:
        data (numpy.ndarray): The training data.
        X_train (numpy.ndarray): Feature data for training.
        y_train (numpy.ndarray): Target labels for training.
        num_of_features (int): Number of features in the data.
        categories (numpy.ndarray): Unique categories in the target labels.
    """
    def __init__(self, data):
        # Validate that data has at least two columns (features and labels)
        if data.shape[1] < 2:
            raise ValueError("Input data must have at least two columns (features and labels).")
        
        self.data = data
        self.X_train = data[:, :-1]  # Extracting feature data
        self.y_train = data[:, -1]  # Extracting target data
        self.num_of_features = self.X_train.shape[1]  # Calculating number of features in the data
        self.y_train = self.y_train.astype(int)  # Ensuring labels are integers
        self.categories = np.unique(self.y_train)  # Finding unique categories
        
    # Define the function to calculate Euclidean distance
    def euclidean_distance(point1, point2):
        """
        Calculate the Euclidean distance between two points represented as NumPy arrays.

        Parameters:
        - point1, point2: NumPy arrays representing the coordinates of the points

        Returns:
        - float: Euclidean distance between the two points
        """
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def mahalanobis_distance(point, mean, covariance_matrix):
        # Ensure the point and mean are column vectors
        if point.ndim == 1:
            point = point.reshape(-1, 1)
        if mean.ndim == 1:
            mean = mean.reshape(-1, 1)

        # Check if the covariance matrix is square and invertible
        if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
            raise ValueError("Covariance matrix should be square")

        cov_inv = np.linalg.inv(covariance_matrix)

        # Calculate Mahalanobis distance
        diff = point - mean
        mahalanobis_dist_squared = np.dot(np.dot(diff.T, cov_inv), diff)

        return np.sqrt(mahalanobis_dist_squared)[0, 0]  # Return as scalar



class KNNClassifier(Classifier):
    """
    k-Nearest Neighbors Classifier.
    
    Attributes:
        k (int): Number of neighbors to consider for classification.
    """
    def __init__(self, data, k=3):
        super().__init__(data)
        
        # Validate the value of k
        if k <= 0:
            raise ValueError("k must be greater than 0.")
        
        self.k = k

    def predict(self, x):
        """
        Predict the label for a given input x.
        
        Args:
            x (numpy.ndarray): The input data point.
            
        Returns:
            int: The most common class label among k nearest neighbors.
        """
        # Compute distances between x and all examples in the training set
        distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train]
        
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return the most common class label
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common


def classification_accuracy(y_true, y_pred):
    """
    Calculate overall classification accuracy and class-wise accuracy.

    Args:
        y_true (numpy.ndarray): True binary labels (ground truth).
        y_pred (numpy.ndarray): Predicted binary labels.

    Returns:
        dict: Dictionary containing overall and class-wise accuracies.
    """
    # Convert to numpy array for consistency
    y_pred = np.array(y_pred)
    
    # Check if the input arrays have the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape.")
    
    # Calculate overall accuracy
    overall_accuracy = np.mean(y_true == y_pred)
    
    # Calculate class-wise accuracy
    categories = np.unique(y_true).astype(int)
    classwise_accuracies = {}
    for category in categories:
        category_index = y_true == category
        classwise_accuracies[category] = np.mean(y_pred[category_index] == category)
        
    final_accuracy = {'overall_accuracy': overall_accuracy}
    for category in categories:
        final_accuracy[category] = classwise_accuracies[category]

    return final_accuracy



        
class Discriminant(Classifier):
    """
    A class for discriminant analysis using Gaussian models.

    Attributes:
        data (numpy.ndarray): The input data as a 2D numpy array.
        pp (dict): Dictionary of prior probabilities for each category.
        number_of_categories (numpy.ndarray): Array of unique category labels.
        category_data (dict): Dictionary to store data for each category.
        mu_vector (dict): Dictionary to store mean vectors for each category.
        vcov (dict): Dictionary to store covariance matrices for each category.
        vcov_inv (dict): Dictionary to store inverse covariance matrices for each category.

    Methods:
        min_eucl_dist_classifier(x_vector):
            Classify an input vector using the minimum Euclidean distance classifier.

        gaussian_pdf(x_vector, mu_vector=None, vcov_matrix=None):
            Calculate the Gaussian probability density function for an input vector.

    """

    def __init__(self, data, pp=None):
        """
        Initialize the Discriminant class with input data and optional prior probabilities.

        Args:
            data (numpy.ndarray): The input data as a 2D numpy array.
            pp (dict, optional): Dictionary of prior probabilities for each category. Default is None.

        """
        super().__init__(data)
        self.category_data = {}
        self.mu_vector = {}
        self.vcov = {}
        self.vcov_inv = {}  
        self.pp = {}
        for category in self.categories:
            self.category_data[category] = self.data[self.data[:, -1] == category]
            self.mu_vector[category] = np.mean(self.category_data[category][:, :-1], axis=0).reshape(-1,1)
            self.vcov[category] = np.cov(self.category_data[category][:, :-1], rowvar=False)
            self.vcov_inv[category] = np.linalg.inv(self.vcov[category])
            if pp is None:
                self.pp[category] = 1 / len(self.categories)
            else:
                self.pp[category] = pp[category]
        self.medc_vcov,self.mahdc_vcov = self.var_mat_gen()
                
                
    def var_mat_gen(self):
        vcov_mat = np.zeros((self.num_of_features,self.num_of_features))
        for category in self.categories:
            vcov_mat = vcov_mat + self.vcov[category]
        mah_vcov_mat = vcov_mat / len(self.categories)
        euc_vcov_mat = np.mean(np.diag(vcov_mat)) * np.eye(vcov_mat.shape[0])
        return euc_vcov_mat, mah_vcov_mat

        

    def min_eucl_dist_classifier(self, x_vector):
        x_vector = x_vector.reshape(-1,1)
        """
        Classify an input vector using the minimum Euclidean distance classifier.

        Args:
            x_vector (numpy.ndarray): The input vector to classify.

        """
        g_category = {}
        for category in self.categories:
            g_category[category] = self.gaussian_pdf(x_vector, 
                                                     mu_vector=self.mu_vector[category],
                                                     vcov_matrix=self.medc_vcov) + np.log(self.pp[category])
        
        max_key = max(g_category, key=lambda k: g_category[k])
        return max_key

    def mah_dist_classifier(self, x_vector):
        x_vector = x_vector.reshape(-1,1)
        """
        Classify an input vector using the minimum Mahalanobis distance classifier.

        Args:
            x_vector (numpy.ndarray): The input vector to classify.

        """
        g_category = {}
        for category in self.categories:
            g_category[category] = self.gaussian_pdf(x_vector, 
                                                     mu_vector=self.mu_vector[category],
                                                     vcov_matrix=self.mahdc_vcov) + np.log(self.pp[category])
        
        max_key = max(g_category, key=lambda k: g_category[k])
        return max_key
    
    
    
    def arbitrary_dist_classifier(self, x_vector):
        x_vector = x_vector.reshape(-1,1)
        """
        Classify an input vector using the arbitrary classifier.

        Args:
            x_vector (numpy.ndarray): The input vector to classify.

        """
        g_category = {}
        for category in self.categories:
            g_category[category] = self.gaussian_pdf(x_vector, 
                                                     mu_vector=self.mu_vector[category],
                                                     vcov_matrix=self.vcov[category]) + np.log(self.pp[category])
        
        max_key = max(g_category, key=lambda k: g_category[k])
        return max_key
    
        
    def gaussian_pdf(x, mean=None, cov=None):
        # Check dimensions and reshape if necessary to make x and mean column vectors
        x = x.reshape(-1, 1)

        # Check if mean is a column vector; if not, reshape
        if mean.ndim == 1 or mean.shape[1] != 1:
            mean = mean.reshape(-1, 1)

        # Check if covariance matrix is 2x2
        if cov.shape != (x.ndim, x.ndim):
            raise ValueError("the dimensions of the covariance matrix and the number of features do not match")

        # x and mean should now be column vectors
        d = len(x)

        # Calculate the constant term
        constant_term = 1 / np.sqrt((2 * np.pi) ** d * np.linalg.det(cov))

        # Calculate the difference between x and mean
        diff = x - mean

        # Calculate the exponent term
        exponent_term = -0.5 * np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff)

        return constant_term * np.exp(exponent_term)


    

    
    
