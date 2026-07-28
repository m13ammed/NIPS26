import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')


def _get_colorbar_limits(args):
    vmin = getattr(args, 'vis_cbar_min', None)
    vmax = getattr(args, 'vis_cbar_max', None)
    return vmin, vmax


def visual(x, y, out, args, id):
    if args.geotype == 'structured_2D':
        visual_structured_2d(x, y, out, args, id)
    elif args.geotype == 'structured_1D':
        visual_structured_1d(x, y, out, args, id)
    elif args.geotype == 'structured_3D':
        visual_structured_3d(x, y, out, args, id)
    elif args.geotype == 'unstructured':
        if x.shape[-1] == 3:
            visual_unstructured_3d(x, y, out, args, id)
        elif x.shape[-1] == 2:
            visual_unstructured_2d(x, y, out, args, id)
    else:
        raise ValueError('geotype not supported')


def visual_unstructured_2d(x, y, out, args, id):
    vmin, vmax = _get_colorbar_limits(args)

    def _scatter_and_save(values, name, use_fixed_limits=False):
        plt.axis('off')
        kwargs = {'cmap': 'coolwarm'}
        if use_fixed_limits:
            kwargs.update({'vmin': vmin, 'vmax': vmax})
        plt.scatter(x=x[0, :, 0].detach().cpu().numpy(),
                    y=x[0, :, 1].detach().cpu().numpy(),
                    c=values, **kwargs)
        plt.colorbar()
        plt.savefig(os.path.join('./results/' + args.save_name + '/',
                                 f"{name}_" + str(id) + ".pdf"),
                    bbox_inches='tight', pad_inches=0)
        plt.close()

    _scatter_and_save(y[0, :].detach().cpu().numpy(), 'gt')
    _scatter_and_save(out[0, :].detach().cpu().numpy(), 'pred')
    _scatter_and_save(((y[0, :] - out[0, :])).detach().cpu().numpy(), 'error', use_fixed_limits=True)


def visual_unstructured_3d(x, y, out, args, id):
    pass


def visual_structured_1d(x, y, out, args, id):
    # Determine visualization bounds
    if args.vis_bound is not None:
        space_x_min = args.vis_bound[0]
        space_x_max = args.vis_bound[1]
    else:
        space_x_min = 0
        space_x_max = args.shapelist[0]
    
    # Extract data and convert to numpy arrays
    x_coords = x[0, :, 0].reshape(args.shapelist[0])[space_x_min:space_x_max].detach().cpu().numpy()
    
    # If there's a second dimension in x, we'll use it as a secondary coordinate
    if x.shape[2] > 1:
        x_values = x[0, :, 1].reshape(args.shapelist[0])[space_x_min:space_x_max].detach().cpu().numpy()
    else:
        # Otherwise just use the indices
        x_values = np.arange(space_x_min, space_x_max)
    
    y_gt = y[0, :, 0].reshape(args.shapelist[0])[space_x_min:space_x_max].detach().cpu().numpy()
    y_pred = out[0, :, 0].reshape(args.shapelist[0])[space_x_min:space_x_max].detach().cpu().numpy()
    error = y_pred - y_gt
    
    # Create results directory if it doesn't exist
    os.makedirs('./results/' + args.save_name + '/', exist_ok=True)
    
    # 1. Input visualization
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, x_coords, 'k-', linewidth=1.5)
    plt.grid(linestyle='--', alpha=0.7)
    plt.title('Input Data')
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                    "input_" + str(id) + ".pdf"), bbox_inches='tight')
    plt.close()
    
    # 2. Prediction visualization
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_pred, 'r-', linewidth=1.5, label='Prediction')
    plt.grid(linestyle='--', alpha=0.7)
    plt.title('Model Prediction')
    plt.legend()
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                    "pred_" + str(id) + ".pdf"), bbox_inches='tight')
    plt.close()
    
    # 3. Ground truth visualization
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_gt, 'b-', linewidth=1.5, label='Ground Truth')
    plt.grid(linestyle='--', alpha=0.7)
    plt.title('Ground Truth')
    plt.legend()
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                    "gt_" + str(id) + ".pdf"), bbox_inches='tight')
    plt.close()
    
    # 4. Comparison visualization (prediction vs ground truth)
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_gt, 'b-', linewidth=1.5, label='Ground Truth')
    plt.plot(x_values, y_pred, 'r--', linewidth=1.5, label='Prediction')
    plt.grid(linestyle='--', alpha=0.7)
    plt.title('Prediction vs Ground Truth')
    plt.legend()
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                    "comparison_" + str(id) + ".pdf"), bbox_inches='tight')
    plt.close()
    
    # 5. Error visualization
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, error, 'g-', linewidth=1.5)
    plt.grid(linestyle='--', alpha=0.7)
    plt.title('Prediction Error')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                    "error_" + str(id) + ".pdf"), bbox_inches='tight')
    plt.close()


def visual_structured_2d(x, y, out, args, id):
    vmin, vmax = _get_colorbar_limits(args)
    if args.vis_bound is not None:
        space_x_min = args.vis_bound[0]
        space_x_max = args.vis_bound[1]
        space_y_min = args.vis_bound[2]
        space_y_max = args.vis_bound[3]
    else:
        space_x_min = 0
        space_x_max = args.shapelist[0]
        space_y_min = 0
        space_y_max = args.shapelist[1]
    plt.axis('off')
    plt.pcolormesh(x[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   x[0, :, 1].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   np.zeros([args.shapelist[0], args.shapelist[1]])[space_x_min: space_x_max, space_y_min: space_y_max],
                   shading='auto', edgecolors='black', linewidths=0.1)
    plt.colorbar()
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                     "input_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.axis('off')
    plt.pcolormesh(x[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   x[0, :, 1].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   out[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   shading='auto', cmap='coolwarm')
    plt.colorbar()
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                     "pred_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.axis('off')
    plt.pcolormesh(x[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   x[0, :, 1].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   y[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   shading='auto', cmap='coolwarm')
    plt.colorbar()
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                     "gt_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.axis('off')
    plt.pcolormesh(x[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   x[0, :, 1].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   out[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy() - \
                   y[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   shading='auto', cmap='coolwarm', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                     "error_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
    plt.close()


def visual_structured_3d(x, y, out, args, id):
    vmin, vmax = _get_colorbar_limits(args)
    # Determine visualization bounds
    if args.vis_bound is not None:
        space_x_min = args.vis_bound[0]
        space_x_max = args.vis_bound[1]
        space_y_min = args.vis_bound[2]
        space_y_max = args.vis_bound[3]
        space_z_min = args.vis_bound[4] if len(args.vis_bound) > 4 else 0
        space_z_max = args.vis_bound[5] if len(args.vis_bound) > 5 else args.shapelist[2]
    else:
        space_x_min = 0
        space_x_max = args.shapelist[0]
        space_y_min = 0
        space_y_max = args.shapelist[1]
        space_z_min = 0
        space_z_max = args.shapelist[2]
    
    # Create results directory if it doesn't exist
    os.makedirs('./results/' + args.save_name + '/', exist_ok=True)
    
    # Extract coordinates
    X = x[0, :, 0].reshape(args.shapelist)[space_x_min:space_x_max, 
                                          space_y_min:space_y_max,
                                          space_z_min:space_z_max].detach().cpu().numpy()
    
    # Extract model output and ground truth
    pred = out[0, :, 0].reshape(args.shapelist)[space_x_min:space_x_max, 
                                               space_y_min:space_y_max,
                                               space_z_min:space_z_max].detach().cpu().numpy()
    
    gt = y[0, :, 0].reshape(args.shapelist)[space_x_min:space_x_max, 
                                           space_y_min:space_y_max,
                                           space_z_min:space_z_max].detach().cpu().numpy()
    
    # Calculate error
    error = pred - gt
    
    # Create a grid for visualization
    x_grid, y_grid, z_grid = np.meshgrid(
        np.linspace(space_x_min, space_x_max, X.shape[0]),
        np.linspace(space_y_min, space_y_max, X.shape[1]),
        np.linspace(space_z_min, space_z_max, X.shape[2])
    )
    
    # 1. Visualize input data (3D points)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Create a downsampled version for better visualization
    ax.set_title('Input 3D Coordinates')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                    "input_" + str(id) + ".pdf"), bbox_inches='tight')
    plt.close()
    
    # 2. Visualize prediction (3D surface)
    # Choose a slice for better visualization
    slice_idx = pred.shape[2] // 2  # Middle z-slice
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x_grid[:, :, slice_idx], y_grid[:, :, slice_idx], 
                          pred[:, :, slice_idx], cmap='coolwarm', 
                          linewidth=0, antialiased=True, alpha=0.7)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title('Model Prediction (Z-slice at index {})'.format(slice_idx))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Value')
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                    "pred_" + str(id) + ".pdf"), bbox_inches='tight')
    plt.close()
    
    # 3. Visualize ground truth (3D surface)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x_grid[:, :, slice_idx], y_grid[:, :, slice_idx], 
                          gt[:, :, slice_idx], cmap='viridis', 
                          linewidth=0, antialiased=True, alpha=0.7)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title('Ground Truth (Z-slice at index {})'.format(slice_idx))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Value')
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                    "gt_" + str(id) + ".pdf"), bbox_inches='tight')
    plt.close()
    
    # 4. Visualize error (3D surface)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x_grid[:, :, slice_idx], y_grid[:, :, slice_idx], 
                          error[:, :, slice_idx], cmap='RdBu_r', 
                          linewidth=0, antialiased=True, alpha=0.7,
                          vmin=vmin, vmax=vmax)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title('Prediction Error (Z-slice at index {})'.format(slice_idx))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Error')
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                    "error_" + str(id) + ".pdf"), bbox_inches='tight')
    plt.close()
            
    # 5. Visualize all 6 faces of the 3D cube for prediction, ground truth, and error
    
    # Create figure layout for the 6 faces
    def plot_cube_faces(data, title, filename, use_fixed_limits=False):
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(3, 3, figure=fig)

        kwargs = {'cmap': 'coolwarm', 'origin': 'lower'}
        if use_fixed_limits:
            kwargs.update({'vmin': vmin, 'vmax': vmax})
        
        # Define the 6 faces
        # Front face (x=0)
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(data[0, :, :].T, **kwargs)
        ax1.set_title('Front Face (x=0)')
        ax1.set_xlabel('Y')
        ax1.set_ylabel('Z')
        
        # Back face (x=max)
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(data[-1, :, :].T, **kwargs)
        ax2.set_title('Back Face (x=max)')
        ax2.set_xlabel('Y')
        ax2.set_ylabel('Z')
        
        # Left face (y=0)
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(data[:, 0, :], **kwargs)
        ax3.set_title('Left Face (y=0)')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        
        # Right face (y=max)
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(data[:, -1, :], **kwargs)
        ax4.set_title('Right Face (y=max)')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Z')
        
        # Bottom face (z=0)
        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(data[:, :, 0], **kwargs)
        ax5.set_title('Bottom Face (z=0)')
        ax5.set_xlabel('X')
        ax5.set_ylabel('Y')
        
        # Top face (z=max)
        ax6 = fig.add_subplot(gs[1, 2])
        im6 = ax6.imshow(data[:, :, -1], **kwargs)
        ax6.set_title('Top Face (z=max)')
        ax6.set_xlabel('X')
        ax6.set_ylabel('Y')

        x_mid = data.shape[0] // 2
        y_mid = data.shape[1] // 2
        z_mid = data.shape[2] // 2

        ax7 = fig.add_subplot(gs[2, 0])
        im7 = ax7.imshow(data[x_mid, :, :].T, **kwargs)
        ax7.set_title(f'X-Center Section (x={x_mid})')
        ax7.set_xlabel('Y')
        ax7.set_ylabel('Z')

        ax8 = fig.add_subplot(gs[2, 1])
        im8 = ax8.imshow(data[:, y_mid, :].T, **kwargs)
        ax8.set_title(f'Y-Center Section (y={y_mid})')
        ax8.set_xlabel('X')
        ax8.set_ylabel('Z')

        ax9 = fig.add_subplot(gs[2, 2])
        im9 = ax9.imshow(data[:, :, z_mid], **kwargs)
        ax9.set_title(f'Z-Center Section (z={z_mid})')
        ax9.set_xlabel('X')
        ax9.set_ylabel('Y')

        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im6, cax=cbar_ax)
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        
        plt.savefig(
            os.path.join('./results/' + args.save_name + '/',
                        filename + "_" + str(id) + ".pdf"), bbox_inches='tight')
        plt.close()
    
    # Plot the 6 faces and 3 orthogonal center sections for prediction, ground truth, and error
    plot_cube_faces(pred, 'Model Prediction - Faces & Center Sections', 'pred_faces')
    plot_cube_faces(gt, 'Ground Truth - Faces & Center Sections', 'gt_faces')
    plot_cube_faces(error, 'Prediction Error - Faces & Center Sections', 'error_faces', use_fixed_limits=True)
