"""
Federated Learning with Dual Attention Heads - ECG Classification Trainer.

This script orchestrates federated learning with dual attention personalization
for ECG classification across 4 institutions (SPH, PTB-XL, SXPH, G12EC).

Model: DualAttentionResNet1D (Hybrid ResNet1D34 + Dual Attention Transformer)
Task: Multi-label classification (20 cardiac conditions)
FL Algorithm: FedDualAtt (Global attention aggregated, Local attention personalized)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithm.ecg.feddualatt import FedDualAttServerHandler, FedDualAttSerialClientTrainer
from algorithm.pipeline import Pipeline
from fedlab.utils.functional import setup_seed
from fedlab.utils.logger import Logger
from torch.utils.data import DataLoader
from datetime import datetime
import torch.nn as nn
from model import get_model
from utils.evaluation import FedClientMultiLabelEvaluator, FedServerMultiLabelEvaluator
from utils.dataloader import get_ecg_dataset
from utils.io import guarantee_path
import json
import argparse
import wandb
import torch


parser = argparse.ArgumentParser(description='Federated Learning with Dual Attention for ECG Classification')
parser.add_argument("--input_path", type=str, default="", help="Path to preprocessed data")
parser.add_argument("--output_path", type=str, default="", help="Path to save results")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--max_epoch", type=int, default=1, help="Number of local epochs per round")
parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
parser.add_argument("--model", type=str, default="dual_attention_resnet1d", help="Model architecture")
parser.add_argument("--mode", type=str, default="feddualatt", help="FL algorithm mode")
parser.add_argument("--communication_round", type=int, default=50, help="Number of FL communication rounds")
parser.add_argument("--case_name", type=str, default="feddualatt_ecg", help="Experiment name for logging")
parser.add_argument("--num_clients", type=int, default=4, help="Number of FL clients")
parser.add_argument("--optimizer_name", type=str, default="SGD", help="Optimizer (SGD or Adam)")
parser.add_argument("--clients", type=list[str], default=["client1", "client2", "client3", "client4"],
                    help="List of client names")
parser.add_argument("--data_fraction", type=float, default=1.0, help="Fraction of data to use (0.0-1.0)")

if __name__ == "__main__":
    args = parser.parse_args()
    setup_seed(args.seed)

    # Device configuration
    if torch.cuda.is_available():
        device = None  # Auto-detect best GPU
        print("CUDA available. Using GPU for training.")
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU for training.")

    # Hyperparameters
    batch_size = args.batch_size
    lr = args.lr
    max_epoch = args.max_epoch
    communication_round = args.communication_round
    num_clients = args.num_clients
    sample_ratio = 1  # All clients participate each round (cross-silo FL)

    # Paths
    input_path = args.input_path
    output_path = args.output_path
    input_path = input_path if input_path[-1] == "/" else input_path + "/"
    output_path = output_path if output_path[-1] == "/" else output_path + "/"
    output_path = output_path + args.model + "/" + args.mode + "/" + datetime.now().strftime("%Y%m%d%H%M%S") + "/"
    clients = args.clients

    # Create datasets for each client
    print("Loading ECG datasets...")
    train_datasets = [get_ecg_dataset(
        [f"{input_path}/ECG/preprocessed/{client}/train.csv"],
        base_path=f"{input_path}/ECG/preprocessed/",
        locations=clients,
        file_name="records.h5",
        n_classes=20,
        frac=args.data_fraction
    ) for client in clients]

    test_datasets = [get_ecg_dataset(
        [f"{input_path}/ECG/preprocessed/{client}/test.csv"],
        base_path=f"{input_path}/ECG/preprocessed/",
        locations=clients,
        file_name="records.h5",
        n_classes=20,
        frac=args.data_fraction
    ) for client in clients]

    # Create data loaders
    train_loaders = [
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for train_dataset in train_datasets
    ]
    test_loaders = [
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        for test_dataset in test_datasets
    ]

    # Initialize model
    print(f"Initializing model: {args.model}")
    model = get_model(args.model)

    # Loss function for multi-label classification
    criterion = nn.BCELoss()

    # Evaluation trackers
    client_evaluators = [FedClientMultiLabelEvaluator() for _ in range(num_clients)]
    server_evaluator = FedServerMultiLabelEvaluator()

    # Create output directories
    for client in clients:
        guarantee_path(output_path + client + "/")
    guarantee_path(output_path + "server/")

    # Save experiment settings
    setting = {
        "dataset": "ECG",
        "model": args.model,
        "batch_size": batch_size,
        "client_lr": lr,
        "criterion": "BCELoss",
        "num_clients": num_clients,
        "sample_ratio": sample_ratio,
        "communication_round": communication_round,
        "max_epoch": max_epoch,
        "seed": args.seed,
        "algorithm": "FedDualAtt",
        "description": "Dual Attention Heads - 4 global (aggregated) + 4 local (personalized)"
    }
    with open(output_path + "setting.json", "w") as f:
        f.write(json.dumps(setting, indent=2))

    # Initialize Weights & Biases logging
    wandb.init(
        project="FedCVD_ECG_FL",
        name=args.case_name,
        config=setting
    )

    # Create loggers
    client_loggers = [
        Logger(log_name=client, log_file=output_path + client + "/logger.log")
        for client in clients
    ]
    server_logger = Logger(log_name="server", log_file=output_path + "server/logger.log")

    # Initialize FL trainer (client-side)
    print("Initializing Federated Dual Attention Client Trainer...")
    trainer = FedDualAttSerialClientTrainer(
        model=model,
        num_clients=num_clients,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        lr=lr,
        criterion=criterion,
        max_epoch=max_epoch,
        output_path=output_path,
        evaluators=client_evaluators,
        optimizer_name=args.optimizer_name,
        device=device,
        logger=client_loggers
    )

    # Initialize FL handler (server-side)
    print("Initializing Federated Dual Attention Server Handler...")
    handler = FedDualAttServerHandler(
        model=model,
        test_loaders=test_loaders,
        criterion=criterion,
        output_path=output_path,
        evaluator=server_evaluator,
        communication_round=communication_round,
        num_clients=num_clients,
        sample_ratio=sample_ratio,
        device=device,
        logger=server_logger
    )

    # Create and run FL pipeline
    print(f"Starting Federated Learning with Dual Attention ({communication_round} rounds)...")
    print(f"Model: {args.model}")
    print(f"Clients: {num_clients} ({clients})")
    print(f"Global attention heads: 4 (aggregated via FedAvg)")
    print(f"Local attention heads: 4 (personalized per client)")
    print("-" * 80)

    standalone = Pipeline(handler, trainer)
    standalone.main()

    print("Training complete!")
    print(f"Results saved to: {output_path}")
