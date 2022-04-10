import torch
import torch.nn as nn
import numpy as np

class LinearBlock(nn.Module):

    def __init__(self, in_nodes, out_nodes):
        super(LinearBlock, self).__init__()
        self.layer = nn.utils.weight_norm(nn.Linear(in_nodes, out_nodes), dim = 0)

    def forward(self, x):
        x = self.layer(x)
        x = x * torch.sigmoid(x) # SiLU
        return x

class PINN(nn.Module):

    def __init__(self, data, layer_list):
        super(PINN, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_layer = nn.utils.weight_norm(nn.Linear(layer_list[0], layer_list[1]), dim = 0)
        self.hidden_layers = self._make_layer(layer_list[1:-1])
        self.output_layer = nn.utils.weight_norm(nn.Linear(layer_list[-2], layer_list[-1]), dim = 0)
        self.data = data
        self.mean = self.data.mean(dim=0).to(device)
        self.sig = torch.sqrt(self.data.var(dim=0)).to(device)

    def _make_layer(self, layer_list):
        layers = []
        for i in range(len(layer_list) - 1):
            block = LinearBlock(layer_list[i], layer_list[i + 1])
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = (x - self.mean) / self.sig
        x = self.input_layer(x)
        x = x * torch.sigmoid(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)

def pinn(data, layer_list):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PINN(data, layer_list).to(device)
    model.apply(weights_init)
    print("Operation mode: ", device)
    return model

def fwd_gradients(obj, x):
    dummy = torch.ones_like(obj)
    derivative = torch.autograd.grad(obj, x, dummy, create_graph= True)[0]
    return derivative

def Navier_Stokes_2D(c, u, v, p, txy, Pec, Rey):
    c_txy = fwd_gradients(c, txy)
    u_txy = fwd_gradients(u, txy)
    v_txy = fwd_gradients(v, txy)
    p_txy = fwd_gradients(p, txy)

    c_t = c_txy[:, 0:1]
    c_x = c_txy[:, 1:2]
    c_y = c_txy[:, 2:3]
    u_t = u_txy[:, 0:1]
    u_x = u_txy[:, 1:2]
    u_y = u_txy[:, 2:3]
    v_t = v_txy[:, 0:1]
    v_x = v_txy[:, 1:2]
    v_y = v_txy[:, 2:3]
    p_x = p_txy[:, 1:2]
    p_y = p_txy[:, 2:3]

    c_xx = fwd_gradients(c_x, txy)[:, 1:2]
    c_yy = fwd_gradients(c_y, txy)[:, 2:3]
    u_xx = fwd_gradients(u_x, txy)[:, 1:2]
    u_yy = fwd_gradients(u_y, txy)[:, 2:3]
    v_xx = fwd_gradients(v_x, txy)[:, 1:2]
    v_yy = fwd_gradients(v_y, txy)[:, 2:3]

    e1 = c_t + (u * c_x + v * c_y) - (1.0 / Pec) * (c_xx + c_yy)
    e2 = u_t + (u * u_x + v * u_y) + p_x - (1.0 / Rey) * (u_xx + u_yy)
    e3 = v_t + (u * v_x + v * v_y) + p_y - (1.0 / Rey) * (v_xx + v_yy)
    e4 = u_x + v_y

    return e1, e2, e3, e4

def Gradient_Velocity_2D(u, v, txy):
    u_txy = fwd_gradients(u, txy)
    v_txy = fwd_gradients(v, txy)

    u_x = u_txy[:, 1:2]
    u_y = u_txy[:, 2:3]
    v_x = v_txy[:, 1:2]
    v_y = v_txy[:, 2:3]

    return u_x, v_x, u_y, v_y

def test_data(T_star, X_star, Y_star, C_star, U_star, V_star, P_star):
    snap = np.random.randint(0, 200)
    t_star = T_star[:, snap:snap+1]
    x_star = X_star[:, snap:snap+1]
    y_star = Y_star[:, snap:snap+1]
    c_star = C_star[:, snap:snap+1]
    u_star = U_star[:, snap:snap+1]
    v_star = V_star[:, snap:snap+1]
    p_star = P_star[:, snap:snap+1]

    variables_star = torch.FloatTensor(np.concatenate((t_star, x_star, y_star), 1))  # N x 3
    target_star = torch.FloatTensor(np.concatenate((c_star, u_star, v_star, p_star), 1))  # N x 4

    return variables_star, target_star

def relative_error(pred, target):
    return torch.sqrt(torch.mean((pred - target)**2)/torch.mean((target - torch.mean(target))**2)).cpu().numpy()

if __name__ == "__main__":
    import numpy as np
    dummy_data = torch.Tensor(np.random.normal(0,1,size=(100,3)))
    layer_list = [3] + 10*[200] + [4]
    model = pinn(dummy_data, layer_list)
    print(model)
