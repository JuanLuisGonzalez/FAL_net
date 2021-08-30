import os
import torch
import torch.utils.data
import torch.nn.parallel
import torch.nn.functional as F
import shutil
import numpy as np
import datetime

a = datetime.timedelta(seconds=24)


def eta_calculator(batch_time_average, epoch_size, remaining_epochs, current_batch):
    eta_seconds = (
        float(batch_time_average)
        * float(epoch_size)
        * (float(remaining_epochs) - (float(current_batch) / float(epoch_size)))
    )

    return datetime.timedelta(seconds=round(eta_seconds))


def flatten(the_lists):
    result = []
    for item in the_lists:
        if isinstance(item, list):
            result += item
        else:
            result.append(item)
    if any(isinstance(item, list) for item in result):
        result = flatten(result)
    return result


def display_config(args, save_path):
    settings = ""
    settings = (
        settings + "############################################################\n"
    )
    settings = (
        settings + "# FAL-net        -         Pytorch implementation          #\n"
    )
    settings = (
        settings + "# by Juan Luis Gonzalez   juanluisgb@kaist.ac.kr           #\n"
    )
    settings = (
        settings + "############################################################\n"
    )
    settings = settings + "-------YOUR TRAINING SETTINGS---------\n"
    for arg in vars(args):
        settings = settings + "%18s: %s\n" % (str(arg), str(getattr(args, arg)))
    print(settings)
    # Save config in txt file
    with open(os.path.join(save_path, "settings.txt"), "w+") as f:
        f.write(settings)


def save_checkpoint(state, is_best, save_path, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(save_path, filename),
            os.path.join(save_path, "model_best.pth.tar"),
        )


def disp2rgb(disp_map, max_value):
    _, h, w = disp_map.shape
    rgb_map = np.ones((3, h, w)).astype(np.float32)

    if max_value is not None:
        normalized_disp_map = disp_map / max_value
    else:
        normalized_disp_map = disp_map / (np.abs(disp_map).max())

    rgb_map[0, :, :] = normalized_disp_map
    rgb_map[1, :, :] = normalized_disp_map
    rgb_map[2, :, :] = normalized_disp_map
    return rgb_map.clip(0, 1)


class RunningAverageMeter(object):
    """Computes and stores the running average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.vals = [0]
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.vals.append(val)
        if len(self.vals) > 10:
            self.vals = self.vals[1:]
        self.sum = sum(self.vals)
        self.count = len(self.vals)
        self.avg = self.sum / self.count

    def get_avg(self):
        return self.avg

    def __repr__(self):
        return "last:{:.3f} avg:({:.3f})".format(self.vals[-1], self.avg)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_avg(self):
        return self.avg

    def __repr__(self):
        return "last:{:.3f} avg:({:.3f})".format(self.val, self.avg)


class multiAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, labels):
        self.meter_no = len(labels)
        self.labels = labels
        self.reset()

    def reset(self):
        self.val = np.zeros(self.meter_no)
        self.avg = np.zeros(self.meter_no)
        self.sum = np.zeros(self.meter_no)
        self.count = np.zeros(self.meter_no)

    def update(self, val, n=1):
        for i in range(self.meter_no):
            self.val[i] = val[i]
            self.sum[i] += val[i] * n
            self.count[i] += n
            self.avg[i] = self.sum[i] / self.count[i]

    def __repr__(self):
        top_label = ""
        bottom_val = ""
        for i in range(self.meter_no):
            top_label += "{:>10}".format(self.labels[i])
            bottom_val += "{:10.4f}".format(self.avg[i])

        reading = top_label + "\n" + bottom_val
        return reading


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def get_rmse(output_right, label_right, mean=(0.411, 0.432, 0.45)):
    # B, C, H, W = output_right.shape
    mean_shift = torch.zeros(output_right.shape).cuda()
    mean_shift[:, 0, :, :] = mean[0]
    mean_shift[:, 1, :, :] = mean[1]
    mean_shift[:, 2, :, :] = mean[2]

    output_right = (output_right + mean_shift) * 255
    output_right[output_right > 255] = 255
    output_right[output_right < 0] = 0
    label_right = (label_right + mean_shift) * 255
    rmse = (torch.mean((output_right - label_right) ** 2)) ** (1 / 2)
    return rmse


kitti_error_names = ["abs_rel", "sq_rel", "rms", "log_rms", "a1", "a2", "a3"]

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351
width_to_focal[1226] = 707.0912
width_to_focal[1280] = 738.2355  # focal lenght upscaled

width_to_baseline = dict()
width_to_baseline[1242] = 0.9982 * 0.54
width_to_baseline[1241] = 0.9848 * 0.54
width_to_baseline[1224] = 1.0144 * 0.54
width_to_baseline[1238] = 0.9847 * 0.54
width_to_baseline[1226] = 0.9765 * 0.54
width_to_baseline[1280] = 0.54

sum_cnt = 0
sum_scale = 0


def compute_kitti_errors(gt, pred, use_median=False, min_d=1.0, max_d=80.0):
    global sum_cnt, sum_scale
    mask = gt > 0
    gt = gt[mask]
    pred = pred[mask]

    if use_median:
        factor = np.median(gt) / np.median(pred)
        pred = factor * pred
        sum_cnt = sum_cnt + 1
        sum_scale = sum_scale + factor
        print(sum_scale / sum_cnt)

    pred[pred > max_d] = max_d
    pred[pred < min_d] = min_d
    gt[gt > max_d] = max_d
    gt[gt < min_d] = min_d

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    errors = [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]

    return errors


def disps_to_depths_kitti2015(gt_disparities, pred_disparities):
    gt_depths = []
    pred_depths = []

    for i in range(len(gt_disparities)):
        gt_disp = gt_disparities[i]
        pred_disp = pred_disparities[i]

        height, width = gt_disp.shape

        gt_mask = gt_disp > 0
        pred_mask = pred_disp > 0

        gt_depth = width_to_focal[width] * 0.54 / (gt_disp + (1.0 - gt_mask))
        pred_depth = width_to_focal[width] * 0.54 / (pred_disp + (1.0 - pred_mask))

        gt_depths.append(gt_mask * gt_depth)
        pred_depths.append(pred_depth)

    return gt_depths, pred_depths


def disps_to_depths_kitti(gt_disparities, pred_disparities):
    gt_depths = []
    pred_depths = []

    for i in range(len(gt_disparities)):
        gt_disp = gt_disparities[i]
        pred_disp = pred_disparities[i]

        height, width = gt_disp.shape
        gt_disp = gt_disp[height - 219 : height - 4, 44:1180]
        pred_disp = pred_disp[height - 219 : height - 4, 44:1180]

        gt_mask = gt_disp > 0
        pred_mask = pred_disp > 0

        gt_depth = gt_disp
        pred_depth = (
            width_to_focal[width]
            * width_to_baseline[width]
            / (pred_disp + (1.0 - pred_mask))
        )

        gt_depths.append(gt_mask * gt_depth)
        pred_depths.append(pred_depth)

    return gt_depths, pred_depths


# Obtain point cloud from estimated disparity
# Expects rgb img (0-255) and disp in pixel units
def get_point_cloud(img, disp):
    b, c, h, w = disp.shape

    # Set camera parameters
    focal = width_to_focal[w]
    cx = w / 2
    cy = h / 2
    baseline = width_to_baseline[w]

    # Get depth from disparity
    z = focal * baseline / (disp + 0.0001)

    # Make normalized grid
    i_tetha = torch.zeros(b, 2, 3).cuda()
    i_tetha[:, 0, 0] = 1
    i_tetha[:, 1, 1] = 1
    grid = F.affine_grid(i_tetha, [b, c, h, w], align_corners=False)
    grid = (grid + 1) / 2

    # Get horizontal and vertical pixel coordinates
    u = grid[:, :, :, 0].unsqueeze(1) * w
    v = grid[:, :, :, 1].unsqueeze(1) * h

    # Get X, Y world coordinates
    x = ((u - cx) / focal) * z
    y = ((v - cy) / focal) * z

    # Cap coordinates
    z[z < 0] = 0
    z[z > 200] = 200

    xyz_rgb = torch.cat([x, z, -y, img], 1)
    xyz_rgb = xyz_rgb.view(b, 6, h * w)

    return xyz_rgb


# Saves pointcloud in .ply format for visualizing
# I recommend blender for visualization
def save_point_cloud(pc, file_name):
    _, vertex_no = pc.shape

    with open(file_name, "w+") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(vertex_no))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar diffuse_red\n")
        f.write("property uchar diffuse_green\n")
        f.write("property uchar diffuse_blue\n")
        f.write("end_header\n")
        for i in range(vertex_no):
            f.write(
                "{:f} {:f} {:f} {:d} {:d} {:d}\n".format(
                    pc[0, i],
                    pc[1, i],
                    pc[2, i],
                    int(pc[3, i]),
                    int(pc[4, i]),
                    int(pc[5, i]),
                )
            )
