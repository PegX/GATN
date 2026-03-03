# avic_gatn/tasks/gatn_real/adapter.py
# avic_gatn/tasks/gatn_real/adapter.py
import os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from avic_gatn.models.circuit_controller import CircuitController
from avic_gatn.models.circuit_controller import CircuitID
import torch
from torchvision import transforms

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class EvalResult:
    primary: float
    metrics: Dict[str, Any]

def build_coco_collate_fn(image_size: int):
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    def collate(batch):
        imgs, ys = [], []
        for sample in batch:
            img, y = sample[0], sample[1]
            if hasattr(img, "mode"):  # PIL
                img = tfm(img)
            imgs.append(img)
            ys.append(y)
        imgs = torch.stack(imgs, 0)
        ys = torch.stack(ys, 0) if torch.is_tensor(ys[0]) else torch.tensor(ys)
        return imgs, ys

    return collate

import torch
import numpy as np
from torchvision import transforms

class CocoCollator:
    def __init__(self, image_size: int, num_classes: int):
        self.num_classes = num_classes
        self.tfm = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def _is_label(self, x):
        if torch.is_tensor(x) and x.ndim == 1 and x.numel() == self.num_classes:
            return True
        if isinstance(x, np.ndarray) and x.ndim == 1 and x.size == self.num_classes:
            return True
        return False

    def _is_image_tensor(self, x):
        return torch.is_tensor(x) and x.ndim == 3 and x.shape[0] in (1,3)

    def __call__(self, batch):
        imgs, ys = [], []

        for sample in batch:
            # sample 可能是 tuple/list，里面还套 tuple/list
            flat = []
            stack = [sample]
            while stack:
                cur = stack.pop(0)
                if isinstance(cur, (list, tuple)):
                    stack = list(cur) + stack
                else:
                    flat.append(cur)

            # 找 label
            y = None
            for item in flat:
                if self._is_label(item):
                    y = torch.from_numpy(item) if isinstance(item, np.ndarray) else item
                    break
            if y is None:
                raise TypeError(f"Cannot find label vector of size {self.num_classes} in sample: types={ [type(x) for x in flat] }")

            # 找 image
            img = None
            for item in flat:
                if hasattr(item, "mode"):  # PIL
                    img = self.tfm(item)
                    break
                if self._is_image_tensor(item):
                    img = item
                    break
            if img is None:
                raise TypeError(f"Cannot find image (PIL or CHW tensor) in sample: types={ [type(x) for x in flat] }")

            imgs.append(img)
            ys.append(y)

        imgs = torch.stack(imgs, 0)
        ys = torch.stack(ys, 0)
        return imgs, ys

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

        from torchvision import transforms
        image_size = int(self.cfg["gatn"].get("image_size", 448))
        num_classes = int(self.cfg["gatn"].get("num_classes", 80))
        val_ds = COCO2014(dataset_root, phase=phase, inp_name=embedding)
        self.val_ds = val_ds

        #image_size = int(self.cfg["gatn"].get("image_size", 448))
        #collator = CocoCollator(image_size)
        collator = CocoCollator(image_size=image_size, num_classes=num_classes)

        batch_size = self.cfg["gatn"]["batch_size"]
        workers = self.cfg["gatn"]["workers"]
        self.val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,   # 先固定 0
            collate_fn=collator,
        )

        # 4) 构建模型（与训练脚本一致）:contentReference[oaicite:11]{index=11}

        t_hidden = int(self.cfg["gatn"].get("t_hidden", 20))

        self.model = gatn_resnet(
            num_classes=num_classes,
            t_hidden=t_hidden,
            t1=t1,
            adj_file=adj_file,
            in_channel=embedding_length,
        ).to(self.device)

        # 5) 加载 checkpoint（你已验证格式是 dict + state_dict）
        ckpt_path = self.cfg["task"]["checkpoint_path"]
        obj = torch.load(ckpt_path, map_location="cpu")
        state_dict = obj["state_dict"]
        # DataParallel 兼容
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        # missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        # print(f"[CKPT] epoch={obj.get('epoch')} best_score={obj.get('best_score')}")
        # print(f"[CKPT] missing={len(missing)} unexpected={len(unexpected)}")
        ret = self.model.load_state_dict(state_dict, strict=False)
        missing = ret.missing_keys
        unexpected = ret.unexpected_keys
        print(f"[CKPT] missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) < 20:
            print("[CKPT] missing sample:", missing[:10])
        if len(unexpected) < 20:
            print("[CKPT] unexpected sample:", unexpected[:10])

        self.controller = CircuitController()
        

        # 例：假设模型里有 blocks / transformer / etc.
        # 你要把 controller 挂到每个 Attention 模块上
        for name, m in self.model.named_modules():
            if m.__class__.__name__ == "Attention":
                m._avic_controller = self.controller

        # 只有一个 transformerblock，所以 layer_idx=0
        blk = self.model.transformerblock
        blk.attn1._avic_controller = self.controller
        blk.attn2._avic_controller = self.controller

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

    # @torch.no_grad()
    # def evaluate_clean(self, steps_eval):
    #     crit = nn.MultiLabelSoftMarginLoss()
    #     total_loss, n = 0.0, 0

    #     for step,batch in self.val_loader:
    #         if steps_eval > 0 and step >= steps_eval:
    #             break
    #         if isinstance(batch, (list, tuple)):
    #             if len(batch) >= 3:
    #                 x, y, inp = batch[0], batch[1], batch[2]
    #             elif len(batch) == 2:
    #                 x, y = batch[0], batch[1]
    #                 # 如果 dataset 只返回 (img, target)，那就必须从 dataset 或 cfg 里拿 inp
    #                 raise RuntimeError(
    #                     "Dataset batch has no 'inp'. Please modify COCO2014 __getitem__ to return (img, target, inp) "
    #                     "or provide an 'inp' tensor in adapter."
    #                 )
    #             else:
    #                 raise RuntimeError(f"Unexpected batch length: {len(batch)}")

    #             x = x.to(self.device)
    #             y = y.to(self.device)
    #             # inp 可能是 list/tuple (like inp[0])，按 models.py 里 inp=inp[0] 的逻辑处理 :contentReference[oaicite:7]{index=7}
    #             if isinstance(inp, (list, tuple)):
    #                 inp = inp[0]
    #             inp = inp.to(self.device)

    #             logits = self.model(x, inp)
    #             loss = crit(logits, y)

    #         else:
    #             raise RuntimeError("Unknown batch format; print the batch keys first.")

    #         bs = x.size(0)
    #         total_loss += loss.item() * bs
    #         n += bs

    #     return {"primary": -(total_loss / max(n, 1)), "loss": (total_loss / max(n, 1))}

    @torch.no_grad()
    def evaluate_clean(self, steps_eval: int = -1):
        import numpy as np
        crit = nn.MultiLabelSoftMarginLoss()
        total_loss, n = 0.0, 0

        # 从 dataset 拿 inp（class embedding）
        inp = None
        for attr in ["inp", "inps", "embeddings"]:
            if hasattr(self.val_ds, attr):
                inp = getattr(self.val_ds, attr)
                break
        if inp is None:
            raise RuntimeError("No 'inp' found in COCO2014 dataset object. Inspect coco.py to expose it.")
        if isinstance(inp, np.ndarray):
            inp = torch.from_numpy(inp)
        if isinstance(inp, (list, tuple)):
            inp = inp[0]
        inp = inp.to(self.device)

        for step, batch in enumerate(self.val_loader):
            if steps_eval > 0 and step >= steps_eval:
                break

            # 你的 batch 目前是 (x, y)
            x, y = batch[0], batch[1]
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x, inp)
            loss = crit(logits, y)

            bs = x.size(0)
            total_loss += loss.item() * bs
            n += bs

        avg_loss = total_loss / max(n, 1)

        metrics = {
            "loss": float(avg_loss)
        }

        # 因为 loss 越小越好，所以 primary 取负数（越大越好）
        primary_value = -float(avg_loss)
        
        return EvalResult(primary=primary_value, metrics=metrics)
        #return {"primary": -avg, "loss": avg}
  
    def list_circuits(self):
        blk = self.model.transformerblock
        # 动态取 head 数，避免写死 5
        H = blk.attn1.num_attention_heads
        circuits = []
        for attn_name in ["attn1", "attn2"]:
            for h in range(H):
                circuits.append(CircuitID(layer=0, attn=attn_name, head=h))
        return circuits

    def set_ablation(self, circuit):
        # circuit=None 表示清除
        if circuit is None:
            self.controller.clear_ablation()
        else:
            self.controller.set_ablation(circuit)

    def sample_batch(self):
        """
        Return one batch for attack/eval.

        For real GATN:
        - img_feat: images [B,3,H,W]
        - node_feat: class embedding inp [80,D]
        - A1/A2: None (model owns them)
        - y: multi-hot labels [B,80]
        """
        import numpy as np

        # 1) get inp (node features)
        inp = None
        for attr in ["inp", "inps", "embeddings"]:
            if hasattr(self.val_ds, attr):
                inp = getattr(self.val_ds, attr)
                break
        if inp is None:
            raise RuntimeError("No 'inp' found in dataset.")

        if isinstance(inp, np.ndarray):
            inp = torch.from_numpy(inp)
        if isinstance(inp, (list, tuple)):
            inp = inp[0]
        node_feat = inp.to(self.device).float()

        # 2) get a batch (images, labels)
        x, y = next(iter(self.val_loader))
        img_feat = x.to(self.device)
        y = y.to(self.device)

        return img_feat, node_feat, None, None, y