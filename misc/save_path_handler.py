import os


def make_save_path(args, script):
    save_path = None
    if args.save_path != None:
        save_path = args.save_path

    elif args.modus_operandi == "train":
        save_path = args.dataset + "_" + script
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        _, sub_directories, _ = next(os.walk(save_path))
        filtered = filter(lambda x: x.isdigit(), sorted(sub_directories))
        idx = len(list(filtered))
        save_path = os.path.join(save_path, str(idx).zfill(10))

    elif args.modus_operandi == "test":
        if not os.path.isfile(args.model):
            raise Exception(f"Could not open model at {args.model}.")
        model_dir = os.path.dirname(args.model)
        save_path = os.path.join(model_dir, "test_results", args.dataset, script)
        if args.f_post_process:
            save_path = save_path + "fpp"
        if args.ms_post_process:
            save_path = save_path + "mspp"

    elif args.modus_operandi == "mean":
        pass

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        args.save_path = save_path
