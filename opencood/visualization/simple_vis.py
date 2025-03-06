from matplotlib import pyplot as plt
import numpy as np

import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev


def visualize(pred_box_tensor, gt_tensor, pcd, pc_range, save_path, method='3d', vis_gt_box=True, vis_pred_box=True,
              left_hand=False, uncertainty=None,ego_lidar_pose=None,vis_ego_box=False,return_map=False,ego_label=None):
    """
    Visualize the prediction, ground truth with point cloud together.
    They may be flipped in y axis. Since carla is left hand coordinate, while kitti is right hand.

    Parameters
    ----------
    pred_box_tensor : torch.Tensor
        (N, 8, 3) prediction.

    gt_tensor : torch.Tensor
        (N, 8, 3) groundtruth bbx

    pcd : torch.Tensor
        PointCloud, (N, 4).

    pc_range : list
        [xmin, ymin, zmin, xmax, ymax, zmax]

    save_path : str
        Save the visualization results to given path.

    dataset : BaseDataset
        opencood dataset object.

    method: str, 'bev' or '3d'

    """
    pc_range = [int(i) for i in pc_range]
    if isinstance(pcd, list):
        pcd_np = [x.cpu().numpy() for x in pcd]
    else:
        pcd_np = pcd.cpu().numpy()

    if vis_pred_box:
        pred_box_np = pred_box_tensor.detach().cpu().numpy()
        # pred_name = ['pred'] * pred_box_np.shape[0]
        pred_name = [''] * pred_box_np.shape[0]
        if uncertainty is not None:
            uncertainty_np = uncertainty.cpu().numpy()
            uncertainty_np = np.exp(uncertainty_np)
            d_a_square = 1.6 ** 2 + 3.9 ** 2

            if uncertainty_np.shape[1] == 3:
                uncertainty_np[:, :2] *= d_a_square
                uncertainty_np = np.sqrt(uncertainty_np)
                # yaw angle is in radian, it's the same in g2o SE2's setting.

                pred_name = [
                    f'x_u:{uncertainty_np[i, 0]:.3f} y_u:{uncertainty_np[i, 1]:.3f} a_u:{uncertainty_np[i, 2]:.3f}' \
                    for i in range(uncertainty_np.shape[0])]

            elif uncertainty_np.shape[1] == 2:
                uncertainty_np[:, :2] *= d_a_square
                uncertainty_np = np.sqrt(uncertainty_np)  # yaw angle is in radian

                pred_name = [f'x_u:{uncertainty_np[i, 0]:.3f} y_u:{uncertainty_np[i, 1]:3f}' \
                             for i in range(uncertainty_np.shape[0])]

            elif uncertainty_np.shape[1] == 7:
                uncertainty_np[:, :2] *= d_a_square
                uncertainty_np = np.sqrt(uncertainty_np)  # yaw angle is in radian

                pred_name = [
                    f'x_u:{uncertainty_np[i, 0]:.3f} y_u:{uncertainty_np[i, 1]:3f} a_u:{uncertainty_np[i, 6]:3f}' \
                    for i in range(uncertainty_np.shape[0])]

    if vis_gt_box:
        gt_box_np = gt_tensor.cpu().numpy()
        # gt_name = ['gt'] * gt_box_np.shape[0]
        gt_name = [''] * gt_box_np.shape[0]
    if vis_ego_box:
        ego_lidar_pose = ego_lidar_pose.cpu().numpy()
        ego_name = [''] * ego_lidar_pose.shape[0]
        if ego_label:
            ego_label[0] = 'ego:'+ego_label[0]
            ego_name = ego_label

    box_line_thickness = 5
    text_line_thickness = 3
    if method == 'bev':
        canvas = canvas_bev.Canvas_BEV_heading_right(
            canvas_shape=((pc_range[4] - pc_range[1]) * 10, (pc_range[3] - pc_range[0]) * 10),
            canvas_x_range=(pc_range[0], pc_range[3]),
            canvas_y_range=(pc_range[1], pc_range[4]),
            left_hand=left_hand
            )

        canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)  # Get Canvas Coords
        canvas.draw_canvas_points(canvas_xy[valid_mask])
        # color_list = [(0, 206, 209),(255, 215,0)]
        # for i, pcd_np_t in enumerate(pcd_np[1:2]):
        #     canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np_t) # Get Canvas Coords
        #     canvas.draw_canvas_points(canvas_xy[valid_mask], colors=color_list[i-1]) # Only draw valid points
        if vis_gt_box:
            # canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name)
            canvas.draw_boxes(gt_box_np, colors=(0, 255, 0), texts=gt_name, box_line_thickness=box_line_thickness)

        if vis_pred_box:
            canvas.draw_boxes(pred_box_np, colors=(255, 0, 0), texts=pred_name, box_line_thickness=box_line_thickness)
        if vis_ego_box:
            print("ego_lidar_pose", ego_lidar_pose, "ego_name", ego_name)
            canvas.draw_boxes(ego_lidar_pose[[0, -1]], colors=(255, 255, 0), texts=[ego_name[0], ego_name[-1]],
                              box_line_thickness=box_line_thickness)

    elif method == '3d':

        canvas = canvas_3d.Canvas_3D(left_hand=left_hand)
        canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
        if vis_gt_box:
            canvas.draw_boxes(gt_box_np + 10, colors=(0, 255, 0), texts=gt_name)


        canvas.draw_canvas_points(canvas_xy[valid_mask])


        if vis_pred_box:
            canvas.draw_boxes(pred_box_np, colors=(255, 0, 0), texts=pred_name)
        if vis_ego_box:
            canvas.draw_boxes(ego_lidar_pose, colors=(255, 255, 0), texts=ego_name,
                              box_line_thickness=box_line_thickness)
    else:
        raise (f"Not Completed for f{method} visualization.")

    plt.axis("off")
    plt.imshow(canvas.canvas)
    plt.tight_layout()
    plt.savefig(save_path, transparent=False, dpi=400, pad_inches=0.0)
    plt.clf()
    if return_map:
        return canvas.canvas
    # print(save_path)


def ori_visualize(pred_box_tensor, gt_tensor, pcd, pc_range, save_path, method='3d', vis_gt_box=True, vis_pred_box=True,
              left_hand=False, uncertainty=None):
    """
    Visualize the prediction, ground truth with point cloud together.
    They may be flipped in y axis. Since carla is left hand coordinate, while kitti is right hand.

    Parameters
    ----------
    pred_box_tensor : torch.Tensor
        (N, 8, 3) prediction.

    gt_tensor : torch.Tensor
        (N, 8, 3) groundtruth bbx

    pcd : torch.Tensor
        PointCloud, (N, 4).

    pc_range : list
        [xmin, ymin, zmin, xmax, ymax, zmax]

    save_path : str
        Save the visualization results to given path.

    dataset : BaseDataset
        opencood dataset object.

    method: str, 'bev' or '3d'

    """

    pc_range = [int(i) for i in pc_range]
    if isinstance(pcd, list):
        pcd_np = [x.cpu().numpy() for x in pcd]
    else:
        pcd_np = pcd.cpu().numpy()

    if vis_pred_box:
        pred_box_np = pred_box_tensor.cpu().numpy()
        # pred_name = ['pred'] * pred_box_np.shape[0]
        pred_name = [''] * pred_box_np.shape[0]
        if uncertainty is not None:
            uncertainty_np = uncertainty.cpu().numpy()
            uncertainty_np = np.exp(uncertainty_np)
            d_a_square = 1.6 ** 2 + 3.9 ** 2

            if uncertainty_np.shape[1] == 3:
                uncertainty_np[:, :2] *= d_a_square
                uncertainty_np = np.sqrt(uncertainty_np)
                # yaw angle is in radian, it's the same in g2o SE2's setting.

                pred_name = [
                    f'x_u:{uncertainty_np[i, 0]:.3f} y_u:{uncertainty_np[i, 1]:.3f} a_u:{uncertainty_np[i, 2]:.3f}' \
                    for i in range(uncertainty_np.shape[0])]

            elif uncertainty_np.shape[1] == 2:
                uncertainty_np[:, :2] *= d_a_square
                uncertainty_np = np.sqrt(uncertainty_np)  # yaw angle is in radian

                pred_name = [f'x_u:{uncertainty_np[i, 0]:.3f} y_u:{uncertainty_np[i, 1]:3f}' \
                             for i in range(uncertainty_np.shape[0])]

            elif uncertainty_np.shape[1] == 7:
                uncertainty_np[:, :2] *= d_a_square
                uncertainty_np = np.sqrt(uncertainty_np)  # yaw angle is in radian

                pred_name = [
                    f'x_u:{uncertainty_np[i, 0]:.3f} y_u:{uncertainty_np[i, 1]:3f} a_u:{uncertainty_np[i, 6]:3f}' \
                    for i in range(uncertainty_np.shape[0])]

    if vis_gt_box:
        gt_box_np = gt_tensor.cpu().numpy()
        # gt_name = ['gt'] * gt_box_np.shape[0]
        gt_name = [''] * gt_box_np.shape[0]

    if method == 'bev':
        canvas = canvas_bev.Canvas_BEV_heading_right(
            canvas_shape=((pc_range[4] - pc_range[1]) * 10, (pc_range[3] - pc_range[0]) * 10),
            canvas_x_range=(pc_range[0], pc_range[3]),
            canvas_y_range=(pc_range[1], pc_range[4]),
            left_hand=left_hand
            )

        canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)  # Get Canvas Coords
        canvas.draw_canvas_points(canvas_xy[valid_mask])
        # color_list = [(0, 206, 209),(255, 215,0)]
        # for i, pcd_np_t in enumerate(pcd_np[1:2]):
        #     canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np_t) # Get Canvas Coords
        #     canvas.draw_canvas_points(canvas_xy[valid_mask], colors=color_list[i-1]) # Only draw valid points
        box_line_thickness = 5
        if vis_gt_box:
            # canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name)
            canvas.draw_boxes(gt_box_np, colors=(0, 255, 0), texts=gt_name, box_line_thickness=box_line_thickness)

        if vis_pred_box:
            canvas.draw_boxes(pred_box_np, colors=(255, 0, 0), texts=pred_name, box_line_thickness=box_line_thickness)

    elif method == '3d':
        canvas = canvas_3d.Canvas_3D(left_hand=left_hand)
        canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
        canvas.draw_canvas_points(canvas_xy[valid_mask])

        if vis_gt_box:
            canvas.draw_boxes(gt_box_np, colors=(0, 255, 0), texts=gt_name)
        if vis_pred_box:
            canvas.draw_boxes(pred_box_np, colors=(255, 0, 0), texts=pred_name)
    else:
        raise (f"Not Completed for f{method} visualization.")

    plt.axis("off")

    plt.imshow(canvas.canvas)

    plt.tight_layout()
    plt.savefig(save_path, transparent=False, dpi=400, pad_inches=0.0)
    plt.clf()