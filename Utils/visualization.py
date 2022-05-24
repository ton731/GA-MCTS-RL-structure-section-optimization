import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join

import normalization





def visualize_ground_motion(individual_folder, ground_motion, norm_dict, gm_name):  
    # for gm_index in range(3):
    
    original_ground_motion = normalization.denormalize_ground_motion(ground_motion, norm_dict).cpu().detach().numpy()
    timeline = (np.arange(original_ground_motion.shape[0])+1) * 0.05

    plt.figure(figsize=(30, 8))
    plt.rcParams['font.size'] = '16'
    plt.plot(timeline, original_ground_motion[:, 0], color='black', linewidth=1)
    plt.xlabel("Time(sec)")
    plt.ylabel("Acceleration")
    plt.title(gm_name, fontsize=20)
    plt.grid()
    plt.savefig(os.path.join(individual_folder, f"{gm_name}.png"))
    plt.close()



def visualize_response(save_dir, x, output, norm_dict, response):  
    save_dir = join(save_dir, response)
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)
    
    if response == "Acceleration":
        # normalized scale
        pred_norm = output[:, :, 0]
        # original scale
        pred = normalization.denormalize_acc(pred_norm, norm_dict)
    elif response == "Velocity":
        # normalized scale
        pred_norm = output[:, :, 1]
        # original scale
        pred = normalization.denormalize_vel(pred_norm, norm_dict)
    elif response == "Displacement":
        # normalized scale
        pred_norm = output[:, :, 2]
        # original scale
        pred = normalization.denormalize_disp(pred_norm, norm_dict)
    elif response == "Moment_Z_Column":
        # Display momentZ on y_n.
        # normalized scale
        pred_norm = output[:, :, 5]
        My_face_norm = x[:, 19]
        # original scale
        pred = normalization.denormalize_Mz(pred_norm, norm_dict)
        My_face = normalization.denormalize_Mz(My_face_norm, norm_dict).cpu().numpy()
    elif response == "Moment_Z_Xbeam":
        # Display momentZ on x_p.
        # normalized scale
        pred_norm = output[:, :, 4]
        My_face_norm = x[:, 17]
        # original scale
        pred = normalization.denormalize_Mz(pred_norm, norm_dict)
        My_face = normalization.denormalize_Mz(My_face_norm, norm_dict).cpu().numpy()
    elif response == "Shear_Y":
        # Display shearY on y_n.
        # normalized scale
        pred_norm = output[:, :, 11]
        # original scale
        pred = normalization.denormalize_Sy(pred_norm, norm_dict)

    # num of grid(3) + grid index(3)
    original_x = normalization.denormalize_x(x[:, :6], norm_dict)
    timeline = (np.arange(output.shape[1])+1) * 0.05

    # Plot
    x_grid_num, y_grid_num, z_grid_num = original_x[0, 0:3].cpu().numpy().astype(int)  

    # Eeach story print diagonal nodes.
    for story in range(1, y_grid_num):
        save_story_dir = join(save_dir, f"{story}F")
        if os.path.exists(save_story_dir) == False:
            os.mkdir(save_story_dir)
        for x_z_coord in range(min(x_grid_num, z_grid_num)):
            node_index = None
            grid_coord = np.array([x_z_coord, story, x_z_coord])
            for i in range(original_x.shape[0]):
                # find the node whose grid index = [x_grid, story, z_grid]
                if (original_x[i, 3:6].cpu().numpy() == grid_coord).all():
                    node_index = i
                    break     
            
            # If the node is not found, continue.
            if(node_index is None): continue
            

            pred_story = pred[node_index, :]
            pred_story = pred_story.cpu().detach().numpy()

            plt.figure(figsize=(30, 8))
            # Set general font size
            plt.rcParams['font.size'] = '16'

            plt.plot(timeline, pred_story, label="pred", color="black", linewidth=1)

            if response == 'Moment_Z_Column' or response == 'Moment_Z_Xbeam':
                My = np.array([My_face[node_index] for _ in timeline])
                plt.plot(timeline, My, color='red', linewidth=1, linestyle='--', label='My')
                plt.plot(timeline, -My, color='red', linewidth=1, linestyle='--')


            plt.legend(loc="upper right")
            plt.grid()
            plt.xlabel("Time(sec)", fontsize=18)
            plt.ylabel(f"{response}", fontsize=18)
            plt.title(f"{story}F, N{node_index+1}, {response}", fontsize=20)
            plt.savefig(join(save_story_dir, f"N{node_index+1}_{response}.png"))
            plt.close()






section_info_dim = 2
My_start_index = 15
yield_factor = 0.95

def visualize_plasticHinge(save_dir, graph_x, output, norm_dict):
    save_dir = join(save_dir, "plastic_hinge")
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)


    # Plot
    original_x = normalization.denormalize_x(graph_x[:, :6], norm_dict)
    x_grid_num, y_grid_num, z_grid_num = original_x[0, 0:3].cpu().numpy().astype(int) 

    
    for z in range(z_grid_num):
        fig, axs = plt.subplots(1, 1, figsize=(14, 10))
        fig.suptitle(f"Plastic Hinge Visualization --- Z{z} Section", fontsize=19, fontweight='bold')

        for y in range(y_grid_num):
            for x in range(x_grid_num):
                grid_coord = np.array([x, y, z])                
                node_index = 0
                for i in range(graph_x.shape[0]):
                    if (original_x[i, 3:6].cpu().numpy() == grid_coord).all():
                        node_index = i
                        break


                # Plot structure skeleton -- beam (0F no beam)
                if x != x_grid_num - 1 and y != 0:
                    # Get x_p section My
                    My_x_p = graph_x[node_index, My_start_index + 1 * section_info_dim].cpu().numpy()
                    color = 1 - (My_x_p - 0.3) / 0.7 * 0.8
                    axs.plot([x, x+1], [y, y], linewidth=5, color=(color, color, color), marker='o', markerfacecolor='k', markersize=10, zorder=1)




        for x in range(x_grid_num):   
            for y in range(y_grid_num):   
                grid_coord = np.array([x, y, z])            
                node_index = 0
                for i in range(graph_x.shape[0]):
                    if (original_x[i, 3:6].cpu().numpy() == grid_coord).all():
                        node_index = i
                        break

                # Plot structure skeleton -- column
                if y != y_grid_num - 1:
                    # Get y_p section My
                    My_y_p = graph_x[node_index, My_start_index + 3 * section_info_dim].cpu().numpy()
                    color = 1 - (My_y_p - 0.3) / 0.7 * 0.8
                    axs.plot([x, x], [y, y+1], linewidth=5, color=(color, color, color), marker='o', markerfacecolor='k', markersize=10, zorder=1)
                    
                
                # Plot plastic hinge
                for i, face_index in enumerate(list(range(3, 9))):   # face_index is index for Mz(x_n, x_p, y_n, y_p, z_n, z_p)
                    pred_moment_face_i = output[node_index, :, face_index]     # [2000]
                    My_localZ_face_i = graph_x[node_index, My_start_index + i * section_info_dim]
                    if My_localZ_face_i <= 0.1:     # It means this face is not connect to any element.
                        continue
                    pred_node_plastic_hinge = (pred_moment_face_i >= yield_factor * My_localZ_face_i) + 0  # [2000]

                    if torch.max(pred_node_plastic_hinge) == 1:
                        if i == 0:
                            axs.add_artist(plt.Circle((x - 0.2, y), 0.05, fill=True, color='red'))
                        elif i == 1:
                            axs.add_artist(plt.Circle((x + 0.2, y), 0.05, fill=True, color='red'))
                        elif i == 2:
                            axs.add_artist(plt.Circle((x, y - 0.2), 0.05, fill=True, color='red'))
                        elif i == 3:
                            axs.add_artist(plt.Circle((x, y + 0.2), 0.05, fill=True, color='red'))


        axs.set_xlabel('x axis (mm)')
        axs.set_ylabel('y axis (mm)')
        axs.set_xlim((-1, x_grid_num))
        axs.set_title("PREDICTION")

        plt.savefig(join(save_dir, f"Z{z}.png"))
        plt.close()
