import sys
import os
from misc.download import check_kitti_availability
from misc.flags import specific_argparse
from misc.utils import print_and_save_config
from misc.save_path_handler import make_save_path


def main():
    print(" ".join(sys.argv[:]))

    args, kitti_needed, script = specific_argparse()

    os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join(
        [str(item) for item in args.gpu_indices]
    )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if kitti_needed:
        check_kitti_availability(args)

    make_save_path(args, script)
    print_and_save_config(args)

    import torch

    from testing.test_k import main as testk
    from testing.test_k_eigen_classic import main as testk_eigenclassic
    from testing.test_a import main as testa
    from train.train_stage1_k import main as train1k
    from train.train_stage2_k import main as train2k
    from predict.predict import predict
    from train.retrain_stage1_a import main as retrain1a
    from train.train_stage1_a import main as train1a
    from mean.mean_a import main as mean_a
    from mean.mean_k import main as mean_k

    device = torch.device("cuda" if args.gpu_indices else "cpu")

    def f(x):
        return {
            "predict": predict,
            "testk_eigenclassic": testk_eigenclassic,
            "testk": testk,
            "testa": testa,
            "train1k": train1k,
            "train1a": train1a,
            "retrain1a": retrain1a,
            "train2k": train2k,
            "mean_a": mean_a,
            "mean_k": mean_k,
        }[x]

    f(script)(args, device)


if __name__ == "__main__":

    main()
