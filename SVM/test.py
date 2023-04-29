# SVM classifier
class SVM:
    def __init__(self, kernel=linear_kernel, C=1):
        self.kernel=kernel
        self.C=C
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Quadratic solver equation:
        # minimize : 1/2*alphaT*alpha*P + qT*alpha
        # subject to G*alpha <= h and A*alpha=b
        
        # In our case:
        # minimize: 1/2*alpha_T*alpha*Yi*Yj*Xi*Xj + [-1]*alpha
        # subject to alpha_i >= 0 ([-1]*alpha >= 0) and Sum of alpha_i*y_i = 0 (i.e. y*alpha = 0)
        
        # Calculating Xi*Xj
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        # Solve the dual optimization problem
        # calculating Yi*Yj*Xi*Xj
        P = matrix(np.outer(y,y) * K)
        # calculating qT*alpha => which is summation of alpha_i in our dual problem
        q = matrix(-1 * np.ones(n_samples))
        
        # matrix of size 2*n_samples x n_samples that corresponds to the constraints 0 <= alpha_i <= C and -alpha_i <= 0
        # we want to create a constraint such that the values of alpha are non-negative. 
        # This can be achieved by multiplying the alpha values by -1 and stacking an identity matrix with the same size as alpha below it.
        G = matrix(np.vstack((np.eye(n_samples)*-1,np.eye(n_samples))))
        # a vector of size 2n_samples x 1 that corresponds to the vector of constants in the inequality constraints.
        # The first n_samples elements are set to 0, which corresponds to the lower bound of the constraint 0 <= alpha_i <= C. 
        # The next n_samples elements are set to C, which corresponds to the upper bound of the constraint 0 <= alpha_i <= C.
        h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        
        # a matrix of size 1 x n_samples that corresponds to the constraint y^T*alpha = 0. (or Summation of (y_i*alpha_i) = 0)
        A = matrix(y, (1,n_samples))
        # a scalar that corresponds to the constant in the equality constraint. 
        # It is set to 0.0, since the constraint is y^T*alpha = 0.
        b = matrix(0.0)

        # Run solver
        sol = solvers.qp(P, q, G, h, A, b)
        alpha = np.array(sol['x']).reshape(n_samples)

        # Get support vectors
        sv_idx = alpha > 1e-5
        self.support_vectors = X[sv_idx]
        self.support_vector_labels = y[sv_idx]
        self.support_vector_weights = alpha[sv_idx]

        # Calculate intercept
        self.intercept = np.mean(self.support_vector_labels - np.sum(self.support_vector_weights * self.support_vector_labels * K[sv_idx], axis=1))