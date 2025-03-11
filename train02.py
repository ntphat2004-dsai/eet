"""
Cleaned PyTorch training pipeline for an eye tracking application.
Contributors: Zuowen Wang, Chang Gao
Institute of Neuroinformatics, University of Zurich and ETH Zurich, TU Delft
Email: wangzu@ethz.ch, gaochangw@outlook.com
"""

import argparse
import json
import os
import shutil
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.BaselineEyeTrackingModel import *
from utils.training_utils import train_epoch, validate_epoch, top_k_checkpoints
from utils.metrics import weighted_MSELoss
from dataset import (
    ThreeETplus_Eyetracking,
    ScaleLabel,
    NormalizeLabel,
    LabelTemporalSubsample,
    SliceLongEventsToShort,
    EventSlicesToVoxelGrid,
    SliceByTimeEventsTargets,
    SpatialShift,
    EventCutout
)

import tonic.transforms as transforms
from tonic import SlicedDataset, DiskCachedDataset
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()

def clean_cached_dataset(cache_dir='./cached_dataset'):
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cleaned cached dataset directory: {cache_dir}")
    os.makedirs(cache_dir, exist_ok=True)

def train(model, train_loader, val_loader, criterion, optimizer, args):
    best_val_loss = float("inf")
    for epoch in range(args.num_epochs):
        # Training phase
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}', leave=True)
        model, train_loss, metrics = train_epoch(model, train_pbar, criterion, optimizer, args)
        
        # Hiển thị metrics qua table
        table_title = (
            f"Epoch {epoch+1}/{args.num_epochs} - Train & Val Metrics"
            if args.val_interval > 0 and (epoch+1) % args.val_interval == 0
            else f"Epoch {epoch+1}/{args.num_epochs} - Train Metrics"
        )
        table = Table(title=table_title)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Train Loss", f"{train_loss:.4f}")
        for name, value in metrics.get('tr_p_euc_error_all', {}).items():
            table.add_row(f"Train {name}", f"{value:.4f}")
        for name, value in metrics.get('tr_p_acc_all', {}).items():
            table.add_row(f"Train {name}", f"{value:.4f}")
        
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metrics(metrics.get('tr_p_acc_all', {}), step=epoch)
        mlflow.log_metrics(metrics.get('tr_p_euc_error_all', {}), step=epoch)

        # Validation phase
        if args.val_interval > 0 and (epoch+1) % args.val_interval == 0:
            val_pbar = tqdm(val_loader, desc='Validation', leave=True)
            val_loss, val_metrics = validate_epoch(model, val_pbar, criterion, args)
            table.add_row("Validation Loss", f"{val_loss:.4f}")
            for name, value in val_metrics.get('val_p_euc_error_all', {}).items():
                table.add_row(f"Val {name}", f"{value:.4f}")
            for name, value in val_metrics.get('val_p_acc_all', {}).items():
                table.add_row(f"Val {name}", f"{value:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(args.model_path, f"model_best_ep{epoch}_val_loss_{val_loss:.4f}.pth")
                torch.save(model.state_dict(), best_model_path)
                rprint(f"[green]Saved best model to:[/green] {best_model_path}")
                top_k_checkpoints(args, mlflow.get_artifact_uri())
            
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metrics(val_metrics.get('val_p_acc_all', {}), step=epoch)
            mlflow.log_metrics(val_metrics.get('val_p_euc_error_all', {}), step=epoch)
        
        console.print(table)
    return model

def main(args):
    clean_cached_dataset()
    
    # Load configuration từ file JSON
    if args.config_file:
        config_path = os.path.join('./configs', args.config_file)
        if not os.path.exists(config_path):
            raise ValueError(f"Configuration file {config_path} not found.")
        with open(config_path, 'r') as f:
            config = json.load(f)
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
        args = argparse.Namespace(**config)
    else:
        raise ValueError("Please provide a JSON configuration file.")

    mlflow.set_tracking_uri(args.mlflow_path)
    mlflow.set_experiment(experiment_name=args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_artifact(__file__)
        mlflow.log_params(vars(args))
        # Lưu lại cấu hình vào file args.json trong thư mục model_path
        args_file_path = os.path.join(args.model_path, "args.json")
        with open(args_file_path, 'w') as f:
            json.dump(vars(args), f)

        # Define your model, optimizer, and criterion
        model = eval(args.architecture)(args).to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4) # Decay learning rate
        if args.loss == "mse":
            criterion = nn.MSELoss()
        elif args.loss == "weighted_mse":
            criterion = weighted_MSELoss(
                weights=torch.tensor((args.sensor_width / args.sensor_height, 1)).to(args.device),
                reduction='mean'
            )
        else:
            raise ValueError("Invalid loss name")

        factor = args.spatial_factor
        temp_subsample_factor = args.temporal_subsample_factor

        # Label transformation: Scale và normalize
        label_transform = transforms.Compose([
            ScaleLabel(factor),
            NormalizeLabel(pseudo_width=640 * factor, pseudo_height=480 * factor)
        ])

        # Augmentation cho training
        train_transform = transforms.Compose([
            transforms.Downsample(spatial_factor=factor),
            SpatialShift(
                max_shift_x=10, 
                max_shift_y=10, 
                sensor_size=(args.sensor_width*factor, args.sensor_height*factor)),
            EventCutout(
                cutout_width=20, 
                cutout_height=20, 
                sensor_size=(args.sensor_width*factor, args.sensor_height*factor))
        ])
        # Validation chỉ dùng Downsample
        val_transform = transforms.Downsample(spatial_factor=factor)

        # Load dữ liệu gốc
        train_data_orig = ThreeETplus_Eyetracking(
            save_to=args.data_dir,
            split="train",
            transform=transforms.Downsample(spatial_factor=factor),
            target_transform=label_transform
        )
        val_data_orig = ThreeETplus_Eyetracking(
            save_to=args.data_dir,
            split="val",
            transform=transforms.Downsample(spatial_factor=factor),
            target_transform=label_transform
        )

        slicing_time_window = args.train_length * int(10000 / temp_subsample_factor)
        train_stride_time = int(10000 / temp_subsample_factor * args.train_stride)
        train_slicer = SliceByTimeEventsTargets(
            slicing_time_window,
            overlap=slicing_time_window - train_stride_time,
            seq_length=args.train_length,
            seq_stride=args.train_stride,
            include_incomplete=False
        )
        val_slicer = SliceByTimeEventsTargets(
            slicing_time_window,
            overlap=0,
            seq_length=args.val_length,
            seq_stride=args.val_stride,
            include_incomplete=False
        )

        post_slicer_transform = transforms.Compose([
            SliceLongEventsToShort(
                time_window=int(10000 / temp_subsample_factor),
                overlap=0,
                include_incomplete=True
            ),
            EventSlicesToVoxelGrid(
                sensor_size=(int(640 * factor), int(480 * factor), 2),
                n_time_bins=args.n_time_bins,
                per_channel_normalize=args.voxel_grid_ch_normaization
            )
        ])

        # Kết hợp transforms: dùng list các transform
        train_post_slicer_transform = transforms.Compose(
            post_slicer_transform.transforms + train_transform.transforms
        )
        val_post_slicer_transform = transforms.Compose(
            post_slicer_transform.transforms + [val_transform]
        )

        # Cắt subsequences từ dữ liệu gốc
        train_data = SlicedDataset(
            train_data_orig,
            train_slicer,
            transform=train_post_slicer_transform,
            metadata_path=f"./metadata/3et_train_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}"
        )
        val_data = SlicedDataset(
            val_data_orig,
            val_slicer,
            transform=val_post_slicer_transform,
            metadata_path=f"./metadata/3et_val_vl_{args.val_length}_vs{args.val_stride}_ch{args.n_time_bins}"
        )

        # Cache dữ liệu để tăng tốc training
        train_data = DiskCachedDataset(
            train_data,
            cache_path=f'./cached_dataset/train_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}'
        )
        val_data = DiskCachedDataset(
            val_data,
            cache_path=f'./cached_dataset/val_vl_{args.val_length}_vs{args.val_stride}_ch{args.n_time_bins}'
        )

        num_workers = max(1, (os.cpu_count() or 1) - 2)
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        # Huấn luyện mô hình
        model = train(model, train_loader, val_loader, criterion, optimizer, args)
        final_model_path = os.path.join(args.model_path, f"model_last_epoch{args.num_epochs}.pth")
        torch.save(model.state_dict(), final_model_path)
        rprint(f"[green]Final model saved at:[/green] {final_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow_path", type=str, help="Path to MLflow tracking server")
    parser.add_argument("--model_path", type=str, help="Path to save trained models")
    parser.add_argument("--experiment_name", type=str, help="Name of the experiment")
    parser.add_argument("--run_name", type=str, help="Name of the run")
    parser.add_argument("--architecture", type=str, help="Model architecture to train")
    parser.add_argument("--config_file", type=str, default=None, help="Path to JSON configuration file")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    # Các tham số khác được mong đợi từ file config hoặc dòng lệnh
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--loss", type=str, default="mse", help="Loss function: mse or weighted_mse")
    parser.add_argument("--sensor_width", type=int, help="Sensor width")
    parser.add_argument("--sensor_height", type=int, help="Sensor height")
    parser.add_argument("--spatial_factor", type=float, help="Spatial downsample factor")
    parser.add_argument("--temporal_subsample_factor", type=float, help="Temporal subsample factor")
    parser.add_argument("--data_dir", type=str, help="Directory of dataset")
    parser.add_argument("--train_length", type=int, help="Train sequence length")
    parser.add_argument("--train_stride", type=int, help="Train sequence stride")
    parser.add_argument("--val_length", type=int, help="Validation sequence length")
    parser.add_argument("--val_stride", type=int, help="Validation sequence stride")
    parser.add_argument("--n_time_bins", type=int, help="Number of time bins")
    parser.add_argument("--voxel_grid_ch_normaization", type=bool, default=False, help="Voxel grid per channel normalization")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--val_interval", type=int, default=1, help="Validation interval (epochs)")

    args = parser.parse_args()
    main(args)
