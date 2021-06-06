import argparse
from mat_to_numpy import load_data
import utils
import vae
import torch
import numpy as np
from time import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hidden Fluid Mechanics - Training')
    parser.add_argument('--version_name', default='0', type=str, help='version name')
    parser.add_argument('--datapath', default='./data/Cylinder2D_flower.mat', type=str, help='data path')
    parser.add_argument('--modelpath', default=None, type=str, help='pretrained model path')
    parser.add_argument('--num_samples', default=157879, type=int, help='number of samples: N out of 157879')
    parser.add_argument('--batch_size', default=10000, type=int, help='batch size')
    parser.add_argument('--total_time', default=40, type=int, help='number of samples')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--vae', default=0, type=int, help=' number of data augmentation data')
    args = parser.parse_args()
    print(args)

    # Data
    data, c_data, eqns, T_star, X_star, Y_star, C_star, U_star, V_star, P_star = load_data(args.datapath,
                                                                                           args.num_samples)

    if args.vae != 0:
        data, c_data = vae.data_augmentation(data, c_data, args.vae)

    # Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    layer_list = [3] + 10 * [200] + [4]
    model = utils.pinn(data, layer_list)

    if args.modelpath != None:
        model.load_state_dict(torch.load(args.modelpath))

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_time = time()
    running_time = 0
    it = 0
    min_loss = 1

    print("Start training the model..")
    while running_time < args.total_time:

        # batch data
        optimizer.zero_grad()
        idx_data = np.random.choice(args.num_samples, args.batch_size)
        idx_eqns = np.random.choice(args.num_samples, args.batch_size)
        data_batch = data[idx_data, :].to(device)
        c_data_batch = c_data[idx_data, :].to(device)
        eqns_batch = data[idx_eqns, :].to(device)
        data_batch.requires_grad = True
        c_data_batch.requires_grad = True
        eqns_batch.requires_grad = True

        # prediction
        data_outputs = model(data_batch)
        c_data_pred = data_outputs[:, 0:1]

        eqns_outputs = model(eqns_batch)
        c_eqns_pred = eqns_outputs[:, 0:1]
        u_eqns_pred = eqns_outputs[:, 1:2]
        v_eqns_pred = eqns_outputs[:, 2:3]
        p_eqns_pred = eqns_outputs[:, 3:4]

        e1, e2, e3, e4 = utils.Navier_Stokes_2D(c_eqns_pred, u_eqns_pred, v_eqns_pred, p_eqns_pred, eqns_batch, 100,
                                                100)

        # loss
        loss_c = torch.mean((c_data_pred - c_data_batch) ** 2)
        loss_e = torch.mean(e1 ** 2) + torch.mean(e2 ** 2) + torch.mean(e3 ** 2) + torch.mean(e4 ** 2)
        loss = loss_c + loss_e
        loss.backward()
        optimizer.step()

        if loss.item() < min_loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), './hfm_'+ args.version_name + '.pth')
            # print(f"It: {it} - Save the best model, loss: {loss.item()}")

        if it % 100 == 0:
            elapsed = time() - start_time
            running_time += elapsed / 3600.0
            print('Iteration: %d, Loss: %.3e, Time: %.2fs, Running Time: %.2fh' % (it, loss, elapsed, running_time))
            start_time = time()

        if (it % 1000 == 0) and (it != 0):
            # Prediction
            with torch.no_grad():
                variables_star, target_star = utils.test_data(T_star, X_star, Y_star, C_star, U_star, V_star, P_star)
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
                print('Error: c: %.3f, u: %.3f, v: %.3f, p: %.3f' % (c_error, u_error, v_error, p_error))

        it += 1
        torch.save(model.state_dict(), './hfm_'+ args.version_name + '_last.pth')





