import argparse
import optuna
import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from vae_model import VAE
from data_loader import CIFAR10DataModule


def objective(trial, args):
    """Optuna objective function for hyperparameter optimization."""
    
    # Suggest hyperparameters
    latent_dim = trial.suggest_int("latent_dim", 64, 256, step=64)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    kl_weight = trial.suggest_float("kl_weight", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    
    # Initialize data module
    dm = CIFAR10DataModule(
        data_dir=args.data_dir,
        batch_size=batch_size,
        num_workers=args.num_workers,
    )
    
    # Initialize model
    model = VAE(
        input_channels=3,
        latent_dim=latent_dim,
        hidden_dims=[32, 64, 128, 256],
        learning_rate=learning_rate,
        kl_weight=kl_weight,
    )
    
    # MLflow logger
    mlflow_logger = MLFlowLogger(
        experiment_name="vae_cifar10",
        tracking_uri=args.mlflow_uri,
        run_name=f"trial_{trial.number}",
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/trial_{trial.number}",
        filename="vae-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
    )
    
    # Train
    trainer.fit(model, dm)
    
    # Return validation loss for optimization
    return trainer.callback_metrics["val_loss"].item()


def train_best_model(best_params, args):
    dm = CIFAR10DataModule(
        data_dir=args.data_dir,
        batch_size=best_params["batch_size"],
        num_workers=args.num_workers,
    )

    model = VAE(
        input_channels=3,
        latent_dim=best_params["latent_dim"],
        hidden_dims=[32, 64, 128, 256],
        learning_rate=best_params["learning_rate"],
        kl_weight=best_params["kl_weight"],
    )

    with mlflow.start_run(run_name="final_model"):
        mlflow_logger = MLFlowLogger(
            experiment_name="vae_cifar10",
            tracking_uri=args.mlflow_uri,
            run_name="final_model",
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints/final_model",
            filename="ckpt",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
        )

        trainer = pl.Trainer(
            max_epochs=args.final_epochs,
            logger=mlflow_logger,
            callbacks=[checkpoint_callback, early_stop],
            accelerator="auto",
            devices=1,
        )

        trainer.fit(model, dm)
        best_path = checkpoint_callback.best_model_path

        mlflow.log_params(best_params)
        # mlflow.pytorch.log_model(model, "model")

    return model, best_path


def main():
    parser = argparse.ArgumentParser(description="Train VAE on CIFAR-10")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--mlflow_uri", type=str, default="./mlruns", help="MLflow tracking URI")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--max_epochs", type=int, default=20, help="Max epochs per trial")
    parser.add_argument("--final_epochs", type=int, default=50, help="Epochs for final model")
    parser.add_argument("--skip_optuna", action="store_true", help="Skip Optuna optimization")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    if not args.skip_optuna:
        print("Starting Optuna hyperparameter optimization...")
        
        # Create Optuna study
        study = optuna.create_study(
            direction="minimize",
            study_name="vae_cifar10_optimization",
        )
        
        # Optimize
        study.optimize(
            lambda trial: objective(trial, args),
            n_trials=args.n_trials,
        )
        
        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best validation loss: {study.best_value:.4f}")
        print(f"Best hyperparameters: {study.best_params}")
        
        # Train final model with best parameters
        best_params = study.best_params
    else:
        print("Skipping Optuna optimization, using default parameters...")
        best_params = {
            "latent_dim": 128,
            "learning_rate": 0.001,
            "kl_weight": 0.00025,
            "batch_size": 128,
        }
    
    # Train final model
    final_model, model_path = train_best_model(best_params, args)
    
    print("Training complete!")
    print(f"Best model path: {model_path}")
    print(f"View results: mlflow ui --backend-store-uri {args.mlflow_uri}")


if __name__ == "__main__":
    main()

