import argparse
from mat_to_numpy import load_data
import utils
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hidden Fluid Mechanics - Evaluation')
    parser.add_argument('--datapath', default='./data/Cylinder2D_flower.mat', type=str, help='data path')
    parser.add_argument('--modelpath', default='./hfm_0.pth', type=str, help='pretrained model path')
    args = parser.parse_args()
    print(args)

    # Data
    data, _, _, T_star, X_star, Y_star, C_star, U_star, V_star, P_star = load_data(args.datapath, 30000)

    # Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    layer_list = [3] + 10 * [200] + [4]
    model = utils.pinn(data, layer_list)
    model.load_state_dict(torch.load(args.modelpath))
    #model.load_state_dict(torch.load(args.modelpath, map_location=torch.device('cpu')))

    with torch.no_grad():
        # Prediction
        c_error_list = []
        u_error_list = []
        v_error_list = []
        p_error_list = []

        for snap in range(T_star.shape[1]):
            t_star = T_star[:, snap:snap + 1] # N x 1
            x_star = X_star[:, snap:snap + 1]
            y_star = Y_star[:, snap:snap + 1]
            c_star = C_star[:, snap:snap + 1]
            u_star = U_star[:, snap:snap + 1]
            v_star = V_star[:, snap:snap + 1]
            p_star = P_star[:, snap:snap + 1]

            variables_star = torch.FloatTensor(np.concatenate((t_star, x_star, y_star), 1))  # N x 3
            target_star = torch.FloatTensor(np.concatenate((c_star, u_star, v_star, p_star), 1))  # N x 4

            data_star_outputs = model(variables_star.to(device))
            c_star_pred = data_star_outputs[:, 0:1]
            u_star_pred = data_star_outputs[:, 1:2]
            v_star_pred = data_star_outputs[:, 2:3]
            p_star_pred = data_star_outputs[:, 3:4]

            # Target (actual values)
            c_target = target_star[:, 0:1].to(device)
            u_target = target_star[:, 1:2].to(device)
            v_target = target_star[:, 2:3].to(device)
            p_target = target_star[:, 3:4].to(device)

            c_error = utils.relative_error(c_star_pred, c_target)
            u_error = utils.relative_error(u_star_pred, u_target)
            v_error = utils.relative_error(v_star_pred, v_target)
            p_error = utils.relative_error(p_star_pred, p_target)

            c_error_list.append(c_error)
            u_error_list.append(u_error)
            v_error_list.append(v_error)
            p_error_list.append(p_error)

            print('[%d] Error: c: %.3f, u: %.3f, v: %.3f, p: %.3f' % (snap, c_error, u_error, v_error, p_error))

        fig = plt.figure(figsize=(10, 10))
        fig.suptitle('Relative L2 Error', fontsize=15)
        plt.subplot(221)
        plt.plot(c_error_list)
        plt.title('c(t,x,y)', fontsize=15)
        plt.xlabel('time')
        plt.subplot(222)
        plt.plot(u_error_list)
        plt.title('u(t,x,y)', fontsize=15)
        plt.xlabel('time')
        plt.subplot(223)
        plt.plot(v_error_list)
        plt.title('v(t,x,y)', fontsize=15)
        plt.xlabel('time')
        plt.subplot(224)
        plt.plot(p_error_list)
        plt.title('p(t,x,y)', fontsize=15)
        plt.xlabel('time')
        #plt.legend(['c', 'u', 'v', 'p'])
        plt.savefig('./results/error_graph.png')

        # Last time
        ct = c_target.cpu().numpy()
        pt = p_target.cpu().numpy()
        ut = u_target.cpu().numpy()
        vt = v_target.cpu().numpy()

        fig = plt.figure(figsize=(20, 10))
        plt.subplot(241)
        plt.scatter(x_star, y_star, c=ct, cmap=cm.jet)
        plt.clim(np.min(ct), np.max(ct))
        plt.xlim(0, 6)
        plt.title('c reference', fontsize=30)
        plt.subplot(242)
        plt.scatter(x_star, y_star, c=c_star_pred.cpu().numpy(), cmap=cm.jet)
        plt.clim(np.min(ct), np.max(ct))
        plt.xlim(0, 6)
        plt.title('c prediction', fontsize=30)
        plt.subplot(243)
        plt.scatter(x_star, y_star, c=pt, cmap=cm.jet)
        plt.clim(np.min(pt), np.max(pt))
        plt.xlim(0, 6)
        plt.title('p reference', fontsize=30)
        plt.subplot(244)
        plt.scatter(x_star, y_star, c=p_star_pred.cpu().numpy(), cmap=cm.jet)
        plt.clim(np.min(pt), np.max(pt))
        plt.xlim(0, 6)
        plt.title('p prediction', fontsize=30)
        plt.subplot(245)
        plt.scatter(x_star, y_star, c=ut, cmap=cm.jet)
        plt.clim(np.min(ut), np.max(ut))
        plt.xlim(0, 6)
        plt.title('u reference', fontsize=30)
        plt.subplot(246)
        plt.scatter(x_star, y_star, c=u_star_pred.cpu().numpy(), cmap=cm.jet)
        plt.clim(np.min(ut), np.max(ut))
        plt.xlim(0, 6)
        plt.title('u prediction', fontsize=30)
        plt.subplot(247)
        plt.scatter(x_star, y_star, c=vt, cmap=cm.jet)
        plt.clim(np.min(vt), np.max(vt))
        plt.xlim(0, 6)
        plt.title('v reference', fontsize=30)
        plt.subplot(248)
        plt.scatter(x_star, y_star, c=v_star_pred.cpu().numpy(), cmap=cm.jet)
        plt.clim(np.min(vt), np.max(vt))
        plt.xlim(0, 6)
        plt.title('v prediction', fontsize=30)
        fig.tight_layout(pad=4.0)
        plt.savefig('./results/last_time_prediction.png')
        plt.close()
