# avic_gatn/tasks/gatn_real/adapter.py
# avic_gatn/tasks/gatn_real/adapter.py
import os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from avic_gatn.models.circuit_controller import CircuitController
from avic_gatn.models.circuit_controller import CircuitID



class RealGATNTaskAdapter:
    def __init__(self, cfg, device="cpu"):
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cpu"))
        self.model = None
        self.val_loader = None

    def setup(self):
        # 1) 把真实 GATN repo 加到 import path（repo_root 就是你 GATN 的根目录）
        repo_root = self.cfg["gatn"]["repo_root"]
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        # 2) 复用原 repo 的 dataset / model factory（来自 train_gatn.py 的 import）
        from coco import COCO2014
        from models import gatn_resnet

        embedding = self.cfg["gatn"]["embedding"]
        embedding_length = int(self.cfg["gatn"]["embedding_length"])
        adj_file = self.cfg["gatn"]["adj_file"]
        t1 = float(self.cfg["gatn"]["t1"])
        num_classes = int(self.cfg["gatn"]["num_classes"])

        # 3) 数据集（val 或 test；原脚本里用 phase='val'）:contentReference[oaicite:10]{index=10}
        dataset_root = self.cfg["gatn"]["dataset_root"]
        phase = self.cfg["gatn"].get("phase", "val")
        val_ds = COCO2014(dataset_root, phase=phase, inp_name=embedding)

        self.val_loader = DataLoader(
            val_ds,
            batch_size=int(self.cfg["gatn"].get("batch_size", 16)),
            shuffle=False,
            num_workers=int(self.cfg["gatn"].get("workers", 4)),
            pin_memory=False,
        )

        # 4) 构建模型（与训练脚本一致）:contentReference[oaicite:11]{index=11}
        self.model = gatn_resnet(
            num_classes=num_classes,
            t1=t1,
            adj_file=adj_file,
            in_channel=embedding_length,
        ).to(self.device)

        # 5) 加载 checkpoint（你已验证格式是 dict + state_dict）
        ckpt_path = self.cfg["gatn"]["checkpoint_path"]
        obj = torch.load(ckpt_path, map_location="cpu")
        state_dict = obj["state_dict"]
        # DataParallel 兼容
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print(f"[CKPT] epoch={obj.get('epoch')} best_score={obj.get('best_score')}")
        print(f"[CKPT] missing={len(missing)} unexpected={len(unexpected)}")

        controller = CircuitController()

        # 例：假设模型里有 blocks / transformer / etc.
        # 你要把 controller 挂到每个 Attention 模块上
        for name, m in self.model.named_modules():
            if m.__class__.__name__ == "Attention":
                m._avic_controller = controller

        # 只有一个 transformerblock，所以 layer_idx=0
        blk = self.model.transformerblock
        blk.attn1._avic_controller = controller
        blk.attn2._avic_controller = controller

        blk.attn1._avic_layer_idx = 0
        blk.attn2._avic_layer_idx = 0

        blk.attn1._avic_attn_name = "attn1"
        blk.attn2._avic_attn_name = "attn2"

        self.model.eval()

        batch = next(iter(self.val_loader))
        print("[BATCH TYPE]", type(batch))
        if isinstance(batch, (list, tuple)):
            print("[BATCH LEN]", len(batch), "shapes:",
                [getattr(x, "shape", None) for x in batch])

    @torch.no_grad()
    def evaluate_clean(self):
        crit = nn.MultiLabelSoftMarginLoss()
        total_loss, n = 0.0, 0

        for batch in self.val_loader:
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 3:
                    x, y, inp = batch[0], batch[1], batch[2]
                elif len(batch) == 2:
                    x, y = batch[0], batch[1]
                    # 如果 dataset 只返回 (img, target)，那就必须从 dataset 或 cfg 里拿 inp
                    raise RuntimeError(
                        "Dataset batch has no 'inp'. Please modify COCO2014 __getitem__ to return (img, target, inp) "
                        "or provide an 'inp' tensor in adapter."
                    )
                else:
                    raise RuntimeError(f"Unexpected batch length: {len(batch)}")

                x = x.to(self.device)
                y = y.to(self.device)
                # inp 可能是 list/tuple (like inp[0])，按 models.py 里 inp=inp[0] 的逻辑处理 :contentReference[oaicite:7]{index=7}
                if isinstance(inp, (list, tuple)):
                    inp = inp[0]
                inp = inp.to(self.device)

                logits = self.model(x, inp)
                loss = crit(logits, y)

            else:
                raise RuntimeError("Unknown batch format; print the batch keys first.")

            bs = x.size(0)
            total_loss += loss.item() * bs
            n += bs

        return {"primary": -(total_loss / max(n, 1)), "loss": (total_loss / max(n, 1))}
  
    def list_circuits(self):
        blk = self.model.transformerblock
        # 动态取 head 数，避免写死 5
        H = blk.attn1.num_attention_heads
        circuits = []
        for attn_name in ["attn1", "attn2"]:
            for h in range(H):
                circuits.append(CircuitID(layer=0, attn=attn_name, head=h))
        return circuits
