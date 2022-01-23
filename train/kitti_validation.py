import time

import torch
import numpy as np

from misc import utils
from misc.loss_functions import realEPE


def validate(args, val_loader, model, epoch, output_writers, device):

    test_time = utils.AverageMeter()
    RMSES = utils.AverageMeter()
    EPEs = utils.AverageMeter()
    kitti_erros = utils.multiAverageMeter(utils.kitti_error_names)

    # switch to evaluate mode
    model.eval()

    # Disable gradients to save memory
    with torch.no_grad():
        for i, input_data in enumerate(val_loader):
            input_left = input_data[0][0].to(device)
            input_right = input_data[0][1].to(device)
            target = input_data[1][0].to(device)
            max_disp = (
                torch.Tensor([args.max_disp * args.relative_baseline])
                .unsqueeze(1)
                .unsqueeze(1)
                .type(input_left.type())
            )

            # Prepare input data
            end = time.time()
            min_disp = max_disp * args.min_disp / args.max_disp
            p_im, disp, maskL, maskRL = model(
                input_left=input_left,
                min_disp=min_disp,
                max_disp=max_disp,
                ret_disp=True,
                ret_pan=True,
                ret_subocc=True,
            )
            test_time.update(time.time() - end)

            # Measure RMSE
            rmse = utils.get_rmse(p_im, input_right, device=device)
            RMSES.update(rmse)

            # record EPE
            flow2_EPE = realEPE(disp, target, sparse=True)
            EPEs.update(flow2_EPE.detach(), target.size(0))

            # Record kitti metrics
            target_depth, pred_depth = utils.disps_to_depths_kitti2015(
                target.detach().squeeze(1).cpu().numpy(),
                disp.detach().squeeze(1).cpu().numpy(),
            )
            kitti_erros.update(
                utils.compute_kitti_errors(target_depth[0], pred_depth[0]),
                target.size(0),
            )

            denormalize = np.array([0.411, 0.432, 0.45])
            denormalize = denormalize[:, np.newaxis, np.newaxis]

            if i < len(output_writers):  # log first output of first batches
                if epoch == 0:
                    output_writers[i].add_image(
                        "Input left", input_left[0].cpu().numpy() + denormalize, 0
                    )

                # Plot disp
                output_writers[i].add_image(
                    "Left disparity", utils.disp2rgb(disp[0].cpu().numpy(), None), epoch
                )

                # Plot left subocclsion mask
                output_writers[i].add_image(
                    "Left sub-occ", utils.disp2rgb(maskL[0].cpu().numpy(), None), epoch
                )

                # Plot right-from-left subocclsion mask
                output_writers[i].add_image(
                    "RightL sub-occ",
                    utils.disp2rgb(maskRL[0].cpu().numpy(), None),
                    epoch,
                )

                # Plot synthetic right (or panned) view output
                p_im = p_im[0].detach().cpu().numpy() + denormalize
                p_im[p_im > 1] = 1
                p_im[p_im < 0] = 0
                output_writers[i].add_image("Output Pan", p_im, epoch)

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t Time {2}\t RMSE {3}".format(
                        i, len(val_loader), test_time, RMSES
                    )
                )

    print("* RMSE {0}".format(RMSES.avg))
    print(" * EPE {:.3f}".format(EPEs.avg))
    print(kitti_erros)
    return RMSES.avg
