import argparse


def specific_argparse():
    kitti_needed = False
    script = ""

    parser = argparse.ArgumentParser(
        description="FAL_net in pytorch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-mo",
        "--modus_operandi",
        default="test",
        help="Select the modus operandi.",
        choices=["test", "predict", "train", "mean"],
        required=True,
    )
    args, _ = parser.parse_known_args()

    if args.modus_operandi in ["test", "predict", "train", "mean"]:
        parser.add_argument(
            "-gpu",
            "--gpu_indices",
            # default=[] if args.modus_operandi in ["predict", "mean"] else [0],
            default=[],
            type=int,
            nargs="+",
            help="GPU indices to train on. Trains on CPU if none are supplied.",
        )

        parser.add_argument(
            "-sp",
            "--save_path",
            help="Path that outputs will be saved to",
            default=None,
            required=True if args.modus_operandi == "predict" else False,
        )

    if args.modus_operandi in ["test", "predict", "train"]:
        parser.add_argument("-w", "--workers", metavar="Workers", default=4, type=int)

        parser.add_argument(
            "-relbase",
            "--relative_baseline",
            default=1,
            help="Relative baseline",
        )

        parser.add_argument(
            "-mdisp", "--max_disp", default=300
        )  # of the training patch W
        parser.add_argument(
            "-mindisp", "--min_disp", default=2
        )  # of the training patch W

        parser.add_argument(
            "--no_levels",
            default=49,
            help="Number of quantization levels in MED",
        )

    if args.modus_operandi in ["test", "train", "mean"]:
        parser.add_argument(
            "-d",
            "--dataset",
            metavar="Name of the Dataset to be used.",
            choices=[
                "KITTI",
                "ASM_stereo_small_test",
                "ASM_stereo_train",
                "ASM_stereo_test",
            ],
            required=True,
        )
        args, _ = parser.parse_known_args()

        if args.dataset == "KITTI":
            script = "mean_k"
        elif "ASM" in args.dataset:
            script = "mean_a"

        parser.add_argument(
            "--data_directory",
            metavar="DIR",
            type=str,
            default="/hdd/asm/datasets/",
            help="Directory containing the datasets",
        )

    if args.modus_operandi in ["predict", "test"]:
        parser.add_argument("-si", "--save_input", action="store_true", default=False)
        parser.add_argument(
            "-mspp",
            "--ms_post_process",
            default=True,
            help="Post-processing with multi-scale input",
        )
        parser.add_argument("--model", help="Model", required=True)

    if args.modus_operandi in ["test", "train"]:
        parser.add_argument(
            "-p",
            "--print_freq",
            default=10,
            type=int,
            metavar="N",
            help="print frequency",
        )
    if args.modus_operandi in ["train", "mean"]:
        parser.add_argument(
            "-b", "--batch_size", metavar="Batch Size", default=1, type=int
        )

    if args.modus_operandi == "predict":
        script = "predict"

        parser.add_argument(
            "--input",
            dest="input",
            default="./data/test.png",
            help="path to the input image to be depth predicted",
            required=True,
        )

        parser.add_argument(
            "-pickle", "--pickle_predictions", action="store_true", default=False
        )

    if args.modus_operandi == "test":
        parser.add_argument(
            "-fpp",
            "--f_post_process",
            default=False,
            help="Post-processing with flipped input",
        )

        parser.add_argument(
            "-save_pan", "--save_pan", action="store_true", default=False
        )
        parser.add_argument("-save", "--save", action="store_true", default=False)

        if args.dataset == "KITTI":
            kitti_needed = True
            parser.add_argument(
                "-save_pc", "--save_pc", action="store_true", default=False
            )

            parser.add_argument(
                "-tesp",
                "--test_split",
                metavar="Name of the test split of the data to be loaded from the dataset.",
                choices=[
                    "eigen_test_improved",
                    "eigen_test_classic",
                ],
                required=True,
            )
            args, _ = parser.parse_known_args()
            if args.test_split == "eigen_test_classic":
                script = "testk_eigenclassic"
                parser.add_argument(
                    "--pickle_predictions",
                    action="store_true",
                    default=False,
                )
            if args.test_split == "eigen_test_improved":
                script = "testk"
        if "ASM" in args.dataset:
            script = "testa"

    if args.modus_operandi == "train":
        parser.add_argument(
            "--pretrained",
            dest="pretrained",
            default=None,
            help="path to pre-trained model",
        )

        parser.add_argument(
            "--epoch_size",
            default=0,
            type=int,
            metavar="N",
            help="manual epoch size (will match dataset size if set to 0)",
        )

        parser.add_argument(
            "--weight_decay",
            "-wd",
            default=0.0,
            type=float,
            metavar="W",
            help="weight decay",
        )

        parser.add_argument(
            "--bias_decay", default=0.0, type=float, metavar="B", help="bias decay"
        )

        parser.add_argument(
            "-perc", "--a_p", default=0.01, help="Perceptual loss weight"
        )

        parser.add_argument(
            "-cw", "--crop_width", metavar="Batch crop W Size", default=320
        )

        parser.add_argument(
            "-ch", "--crop_height", metavar="Batch crop H Size", default=320
        )

        parser.add_argument("-op", "--optimizer", metavar="Optimizer", default="adam")
        parser.add_argument(
            "--beta",
            metavar="BETA",
            type=float,
            help="Beta parameter for adam",
            default=0.999,
        )

        parser.add_argument(
            "--momentum",
            default=0.5,
            type=float,
            metavar="Momentum",
            help="Momentum for Optimizer",
        )

        parser.add_argument(
            "-s",
            "--stage",
            metavar="Which training stage to be used.",
            type=int,
            choices=[
                1,
                2,
            ],
            required=True,
        )
        args, _ = parser.parse_known_args()
        parser.add_argument(
            "--epochs",
            default=50 if args.stage == 1 else 20,
            type=int,
            metavar="N",
            help="number of total epochs to run in train stage 1",
        )
        parser.add_argument(
            "--lr",
            type=float,
            metavar="Initial Learning Rate Train1",
            default=0.0001 if args.stage == 1 else 0.00005,
        )
        parser.add_argument(
            "-sm",
            "--smooth",
            default=0.2 * 2 / 512 if args.stage == 1 else 0.4 * 2 / 512,
            help="Smoothness loss weightfor training stage 1",
        )
        parser.add_argument(
            "--milestones",
            default=[30, 40] if args.stage == 1 else [5, 10],
            metavar="N",
            nargs="*",
            help="epochs at which learning rate is divided by 2 in training stage 1",
        )
        if args.stage == 2:
            parser.add_argument(
                "-mirror_loss", "--a_mr", default=1, help="Mirror loss weight"
            )
            parser.add_argument(
                "--fix_model",
                dest="fix_model",
                default="KITTI_stage1/08-20-13_25/FAL_netB,e50es,b1,lr0.0001/checkpoint.pth.tar",
                required=True,
            )
        if args.dataset == "KITTI":
            kitti_needed = True
            parser.add_argument(
                "-trsp",
                "--train_split",
                metavar="Name of the train split of the data to be loaded from the dataset.",
                default="eigen_train",
            )
            parser.add_argument(
                "-vds",
                "--validation_dataset",
                metavar="Validation dataset Name",
                default="KITTI2015",
            )

            parser.add_argument(
                "-vasp",
                "--validation_split",
                metavar="Name of the validation split of the data to be loaded from the dataset.",
                default="bello_val",
            )
            if args.stage == 1:
                script = "train1k"
            if args.stage == 2:
                script = "train2k"
        elif "ASM" in args.dataset:
            parser.add_argument(
                "--val_freq",
                default=0,
                type=int,
                metavar="N",
                help="validation frequency",
            )
            parser.add_argument(
                "--retrain",
                default=False,
                type=bool,
                help="retrain or not",
            )
            args, _ = parser.parse_known_args()

            if args.retrain:
                script = "retrain1a"
            else:
                script = "train1a"

    args = parser.parse_args()
    return args, kitti_needed, script
