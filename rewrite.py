import torch
from models.FAL_netB import FAL_netB
import os

device = torch.device("cpu")
model = FAL_netB(no_levels=49, device=device)
checkpoint = torch.load(
    "./KITTI_stage1/08-25-11_55/FAL_netB,e50es,b16,lr0.0001/model_best.pth.tar",
    map_location=device,
)
# print((checkpoint.keys()))
model.load_state_dict(checkpoint["state_dict"])
torch.save(
    {
        "epoch": 50,
        "model_state_dict": model.module.state_dict()
        if isinstance(model, torch.nn.DataParallel)
        else model.state_dict(),
        "optimizer_state_dict": None,
        "loss": None,
    },
    os.path.join("./lel/", "model_best.pth.tar"),
)
