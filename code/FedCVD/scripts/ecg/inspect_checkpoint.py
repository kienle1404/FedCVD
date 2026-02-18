"""Quick script to inspect a FedDualAtt checkpoint."""
import torch
import sys

path = sys.argv[1] if len(sys.argv) > 1 else r"c:\Users\kl23a\Documents\Working\FedCVD\output\dual_attention_resnet1d\feddualatt\global5_local3\seed42\20260204155710\server\model.pth"

ckpt = torch.load(path, map_location='cpu', weights_only=False)
print(f"Keys: {list(ckpt.keys())}")
print(f"Round: {ckpt.get('round')}")

if 'local_attention_params' in ckpt:
    lap = ckpt['local_attention_params']
    print(f"local_attention_params: {len(lap)} clients")
    for i, d in enumerate(lap):
        print(f"  Client {i}: {len(d)} params")
        nonzero = sum(1 for v in d.values() if v.abs().sum() > 0)
        print(f"    Non-zero tensors: {nonzero}/{len(d)}")
        for k in list(d.keys())[:2]:
            print(f"    {k}: shape={d[k].shape}, norm={d[k].norm():.4f}")
else:
    print("No local_attention_params in checkpoint!")

if 'global_model' in ckpt:
    sd = ckpt['global_model']
    local_keys = [k for k in sd.keys() if 'local_att' in k or 'local_proj' in k]
    print(f"\nglobal_model local param positions ({len(local_keys)} keys):")
    for k in local_keys[:4]:
        print(f"  {k}: norm={sd[k].norm():.6f}")
    if local_keys:
        all_zero = all(sd[k].abs().sum() == 0 for k in local_keys)
        print(f"  All local positions zero in global_model? {all_zero}")
