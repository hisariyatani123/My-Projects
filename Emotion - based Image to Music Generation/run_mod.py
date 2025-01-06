from train_mod import train

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset_path', type=str, required=True)
parser.add_argument('--valid_dataset_path', type=str, required=True)
parser.add_argument('--precompute_dir', type=str, required=True)
parser.add_argument('--lr', type=float, required=False, default=1e-5)
parser.add_argument('--epochs', type=int, required=False, default=100)
parser.add_argument('--using_wandb', type=bool, required=False, default=False)
parser.add_argument('--save_file', type=str, required=True)
parser.add_argument('--weight_decay', type=float, required=False, default=0.001)
parser.add_argument('--grad_acc', type=int, required=False, default=2)
parser.add_argument('--warmup_steps', type=int, required=False, default=100)
parser.add_argument('--batch_size', type=int, required=False, default=4)
parser.add_argument('--use_cfg', type=int, required=False, default=0)
args = parser.parse_args()

train(
    train_dataset_path=args.train_dataset_path,
    valid_dataset_path=args.valid_dataset_path,
    precompute_dir=args.precompute_dir,
    lr=args.lr,
    epochs=args.epochs,
    using_wandb=args.using_wandb,
    save_file=args.save_file,
    weight_decay=args.weight_decay,
    grad_acc=args.grad_acc,
    warmup_steps=args.warmup_steps,
    batch_size=args.batch_size,
    use_cfg=args.use_cfg,
)