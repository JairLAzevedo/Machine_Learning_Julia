import numpy as np

class LDA:
    
    def __init__(self,n_components):
        self.n_components = n_components
        self.linear_discriminants = None
        
    def fit(self,X,y):
        n_features = X.shape[1]
        class_labels = np.unique(y)
                
        mean_overall = np.mean(X,axis=0)
        S_W = np.zeros((n_features,n_features))
        S_B = np.zeros((n_features,n_features))
        
        for c in class_labels:
                bol=y == c
                X_c = X[bol[:,0],:]
                mean_c = np.mean(X_c,axis=0)
                S_W += (X_c- mean_c).T.dot(X_c-mean_c)
                
                n_samples = X.shape[0]
                mean_diff = (mean_c - mean_overall).reshape(n_features,1)
                S_B += n_samples*mean_diff.dot(mean_diff.T)
                
        A = np.linalg.inv(S_W)@S_B
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvectors = eigenvectors.T
        ind = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[ind]
        eigenvectors = eigenvectors[ind]
        self.linear_discriminants = eigenvectors[0:self.n_components]
        
    def transform(self,X):
        return np.dot(X,self.linear_discriminants.T)

