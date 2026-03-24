import argparse
import os
import itertools
from collections import defaultdict
import torch
import pytorch_lightning as pl

# 开启 TensorCore 加速，消除警告并提升 RTX 40/30 系显卡的训练速度
# torch.set_float32_matmul_precision('medium')

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

# 导入自定义模块
from dataloader import KnowledgeGraph, ReasoningDataset, collate_fn
from model import KGReasoningModel
from lightning import KGReasoningModule

def parse_args():
    parser = argparse.ArgumentParser(description="Inductive Logic KGC")
    
    # 指向包含 train.txt, support.txt, test.txt 的文件夹
    parser.add_argument('--data_path', type=str, required=True, help='Folder path containing generated dataset, e.g., data_generated/cn15k-10')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model Hparams
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--d_model_list', type=str, default='128')
    parser.add_argument('--n_layers_list', type=str, default='3')
    parser.add_argument('--top_k_list', type=str, default='5,10,15,20')
    parser.add_argument('--metric_name', type=str, default='test_mae')
    parser.add_argument('--plot_path', type=str, default='')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--drop_rate', type=float, default=0.2, help='DropEdge rate for dynamic graph augmentation')
    # 【修改】：消融实验开关 (Ablation Flags) - 移除了重复的 bico，加入了 former
    parser.add_argument('--disable_sfe', action='store_true', help='Ablation: remove Structure Feature Encoder')
    parser.add_argument('--disable_lre', action='store_true', help='Ablation: remove Logic Reasoning Encoder')
    parser.add_argument('--disable_conf', action='store_true', help='Ablation: remove Confidence Encoder')
    parser.add_argument('--disable_former', action='store_true', help='Ablation: remove Global Linear Attention (Former)') # 新增
    parser.add_argument('--disable_bico', action='store_true', help='Ablation: remove BiCoAttention (Compatibility)') 
    parser.add_argument('--conf_mask_prob', type=float, default=0.8, help='Keep probability for confidence masking')
    
    return parser.parse_args()

def parse_int_list(value, default_values):
    if value is None or value == '':
        return list(default_values)
    return [int(v.strip()) for v in value.split(',') if v.strip()]

def build_dataloaders(args):
    print(f"Loading Dataset from: {args.data_path}")
    print("Initializing Train KG...")
    train_kg = KnowledgeGraph(args.data_path, mode='train')
    print("Initializing Valid KG...")
    val_kg = KnowledgeGraph(
        args.data_path,
        mode='valid',
        entity_dict=train_kg.entity2id,
        relation_dict=train_kg.relation2id
    )
    print("Initializing Test KG (with Support Set)...")
    test_kg = KnowledgeGraph(
        args.data_path,
        mode='test',
        entity_dict=train_kg.entity2id,
        relation_dict=train_kg.relation2id
    )
    train_dataset = ReasoningDataset(train_kg, drop_rate=args.drop_rate)
    val_dataset = ReasoningDataset(val_kg, drop_rate=0.0)
    test_dataset = ReasoningDataset(test_kg, drop_rate=0.0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    return train_kg, train_loader, val_loader, test_loader

def to_float(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)

def run_single_experiment(args, train_kg, train_loader, val_loader, test_loader, d_model, n_layers, top_k, run_dir):
    pl.seed_everything(args.seed)
    print(f"Entities: {train_kg.n_entities}, Relations: {train_kg.n_relations}")
    model = KGReasoningModel(
        n_ents=train_kg.n_entities,
        n_rels=train_kg.n_relations * 2,
        d_model=d_model,
        n_layers=n_layers,
        top_k_evd=top_k,
        disable_sfe=args.disable_sfe,
        disable_lre=args.disable_lre,
        disable_conf=args.disable_conf,
        disable_former=args.disable_former,
        disable_bico=args.disable_bico,
        conf_mask_prob=args.conf_mask_prob
    )
    lit_model = KGReasoningModule(model, lr=args.lr)
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        filename='model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if args.gpu > 0 and torch.cuda.is_available() else 'cpu',
        devices=args.gpu if args.gpu > 0 and torch.cuda.is_available() else 'auto',
        callbacks=[checkpoint_callback, early_stop],
        log_every_n_steps=10
    )
    print("================ Starting Training ================")
    try:
        trainer.fit(lit_model, train_loader, val_loader)
    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("🛑 [Interrupted] Training manually stopped by user (Ctrl+C).")
        print("🔄 Automatically jumping to the Testing Phase...")
        print("=" * 50 + "\n")
    best_path = checkpoint_callback.best_model_path
    if best_path and os.path.exists(best_path):
        print(f"✅ Best model saved at: {best_path}")
    else:
        print("⚠️ Warning: No checkpoint saved yet. (Perhaps interrupted too early)")
    print("================ Starting Final Inductive Test ================")
    test_results = []
    try:
        if best_path and os.path.exists(best_path):
            test_results = trainer.test(model=lit_model, dataloaders=test_loader, ckpt_path='best')
        else:
            print("Running test with current in-memory model weights...")
            test_results = trainer.test(model=lit_model, dataloaders=test_loader)
    except Exception as e:
        print(f"❌ Testing failed with error: {e}")
    metrics = {
        "d_model": d_model,
        "n_layers": n_layers,
        "top_k": top_k
    }
    if test_results:
        for k, v in test_results[0].items():
            metrics[k] = to_float(v)
    val_loss = trainer.callback_metrics.get('val_loss')
    val_mae = trainer.callback_metrics.get('val_mae')
    if val_loss is not None:
        metrics['val_loss'] = to_float(val_loss)
    if val_mae is not None:
        metrics['val_mae'] = to_float(val_mae)
    return metrics

def resolve_unique_path(path):
    if not path:
        return path
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    if not ext:
        ext = ".png"
    idx = 1
    candidate = f"{base}_{idx}{ext}"
    while os.path.exists(candidate):
        idx += 1
        candidate = f"{base}_{idx}{ext}"
    return candidate

def plot_topk_bar(results, metric_name, plot_path):
    import matplotlib.pyplot as plt
    values_by_topk = defaultdict(list)
    for row in results:
        if metric_name in row and row.get("top_k") is not None:
            values_by_topk[row["top_k"]].append(row[metric_name])
    topks = sorted(values_by_topk.keys())
    if not topks:
        print("No metrics available to plot.")
        return
    means = [sum(values_by_topk[k]) / len(values_by_topk[k]) for k in topks]
    plt.figure(figsize=(8, 4))
    plt.bar([str(k) for k in topks], means)
    plt.xlabel("top_k")
    plt.ylabel(metric_name)
    plt.tight_layout()
    if plot_path:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True) if os.path.dirname(plot_path) else None
        plot_path = resolve_unique_path(plot_path)
        plt.savefig(plot_path)
        print(f"Saved bar chart to: {plot_path}")
    plt.close()

def main():
    args = parse_args()
    train_kg, train_loader, val_loader, test_loader = build_dataloaders(args)
    if not args.sweep:
        run_dir = args.save_dir
        run_single_experiment(
            args,
            train_kg,
            train_loader,
            val_loader,
            test_loader,
            args.d_model,
            args.n_layers,
            args.top_k,
            run_dir
        )
        return
    d_model_list = parse_int_list(args.d_model_list, [args.d_model])
    n_layers_list = parse_int_list(args.n_layers_list, [args.n_layers])
    top_k_list = parse_int_list(args.top_k_list, [args.top_k])
    results = []
    for d_model, n_layers, top_k in itertools.product(d_model_list, n_layers_list, top_k_list):
        run_dir = os.path.join(args.save_dir, f"d{d_model}_l{n_layers}_k{top_k}")
        metrics = run_single_experiment(
            args,
            train_kg,
            train_loader,
            val_loader,
            test_loader,
            d_model,
            n_layers,
            top_k,
            run_dir
        )
        results.append(metrics)
    metric_name = args.metric_name
    available = set()
    for row in results:
        available.update(row.keys())
    if metric_name not in available:
        if "val_mae" in available:
            metric_name = "val_mae"
        elif "val_loss" in available:
            metric_name = "val_loss"
        elif "test_loss" in available:
            metric_name = "test_loss"
    plot_path = args.plot_path or os.path.join(args.save_dir, "topk_bar.png")
    plot_topk_bar(results, metric_name, plot_path)

if __name__ == "__main__":
    main()
