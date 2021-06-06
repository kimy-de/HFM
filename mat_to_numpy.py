import scipy.io
import numpy as np
import torch

def load_data(path, num_samples):
    data = scipy.io.loadmat(path)

    t_star = data['t_star']  # T x 1
    x_star = data['x_star']  # N x 1
    y_star = data['y_star']  # N x 1

    T = t_star.shape[0]
    N = x_star.shape[0]

    U_star = data['U_star']  # N x T
    V_star = data['V_star']  # N x T
    P_star = data['P_star']  # N x T
    C_star = data['C_star']  # N x T
    T_star = np.tile(t_star, (1, N)).T  # N x T
    X_star = np.tile(x_star, (1, T))  # N x T
    Y_star = np.tile(y_star, (1, T))  # N x T

    # For Training
    idx_t = np.concatenate([np.array([0]), np.random.choice(T - 2, T - 2, replace=False) + 1, np.array([T - 1])])
    idx_x = np.random.choice(N, num_samples, replace=False)
    t_data = T_star[:, idx_t][idx_x, :].flatten()[:, None] # ST x 1
    x_data = X_star[:, idx_t][idx_x, :].flatten()[:, None] # ST x 1
    y_data = Y_star[:, idx_t][idx_x, :].flatten()[:, None] # ST x 1
    c_data = C_star[:, idx_t][idx_x, :].flatten()[:, None] # ST x 1

    c_tensor_data = torch.FloatTensor(c_data)

    idx_t = np.concatenate([np.array([0]), np.random.choice(T - 2, T - 2, replace=False) + 1, np.array([T - 1])])
    idx_x = np.random.choice(N, num_samples, replace=False)
    t_eqns = T_star[:, idx_t][idx_x, :].flatten()[:, None] # ST x 1
    x_eqns = X_star[:, idx_t][idx_x, :].flatten()[:, None] # ST x 1
    y_eqns = Y_star[:, idx_t][idx_x, :].flatten()[:, None] # ST x 1

    variables = torch.FloatTensor(np.concatenate((t_data, x_data, y_data), 1)) # ST x 3
    eqns = torch.FloatTensor(np.concatenate((t_eqns, x_eqns, y_eqns), 1)) # ST x 3

    print(f"Number of Time Steps: {T}, Number of sample points: {num_samples} out of {N}")

    return variables, c_tensor_data, eqns, T_star, X_star, Y_star, C_star, U_star, V_star, P_star