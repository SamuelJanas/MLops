import argparse
from typing import Dict, Any

import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from model import MNISTClassifier
from data_loader import MNISTDataModule


def build_trainer_and_run(
    model_hparams: Dict[str, Any],
    dm_hparams: Dict[str, Any],
    max_epochs: int,
    project: str,
    run_name: str,
    trial: optuna.Trial | None = None,
) -> float:
    """Build datamodule, model, trainer, run training and return validation accuracy."""
    pl.seed_everything(42)

    # Data
    dm = MNISTDataModule(
        data_dir=dm_hparams["data_dir"],
        batch_size=dm_hparams["batch_size"],
        num_workers=dm_hparams["num_workers"],
    )

    # Model
    model = MNISTClassifier(**model_hparams)

    # Logging
    wandb_logger = WandbLogger(
        project=project,
        name=run_name,
        log_model=True,
        reinit=True,  # separate run per trial
    )
    wandb_logger.experiment.config.update(
        {**model_hparams, **dm_hparams, "max_epochs": max_epochs}
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/mnist_classifier",
        filename="epoch{epoch:02d}-val_acc{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )
    early_stop = EarlyStopping(
        monitor="val_acc",
        patience=3,
        mode="max",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop],
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        logger=wandb_logger,
    )

    trainer.fit(model, dm)

    # Evaluate on validation set to return score to Optuna
    val_metrics = trainer.validate(model, datamodule=dm, verbose=False)[0]
    val_acc = float(val_metrics.get("val_acc", 0.0))

    # Also run test at the end (not used for optimization)
    trainer.test(model, datamodule=dm, verbose=False)

    if trial is not None:
        # Report intermediate score to Optuna
        trial.report(val_acc, step=max_epochs)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_acc


def objective(trial: optuna.Trial, base_args: argparse.Namespace) -> float:
    """Optuna objective function."""
    # Hyperparameter search space
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    model_hparams = {
        "learning_rate": learning_rate,
    }

    dm_hparams = {
        "data_dir": base_args.data_dir,
        "batch_size": batch_size,
        "num_workers": base_args.num_workers,
    }

    run_name = f"mnist-optuna-trial-{trial.number}"
    val_acc = build_trainer_and_run(
        model_hparams=model_hparams,
        dm_hparams=dm_hparams,
        max_epochs=base_args.max_epochs,
        project=base_args.wandb_project,
        run_name=run_name,
        trial=trial,
    )

    return val_acc


def run_single_training(args: argparse.Namespace) -> None:
    """Train a single model with given CLI hyperparameters (no Optuna)."""
    model_hparams = {
        "learning_rate": args.learning_rate,
    }
    dm_hparams = {
        "data_dir": args.data_dir,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }

    run_name = args.run_name or "mnist-baseline"
    val_acc = build_trainer_and_run(
        model_hparams=model_hparams,
        dm_hparams=dm_hparams,
        max_epochs=args.max_epochs,
        project=args.wandb_project,
        run_name=run_name,
    )
    print(f"Final validation accuracy: {val_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train MNIST-10 classifier")
    # Data / training params
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")

    # W&B
    parser.add_argument("--wandb_project", type=str, default="mlops-mnist", help="Weights & Biases project name")
    parser.add_argument("--run_name", type=str, default=None, help="Optional W&B run name")

    # Optuna
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter optimization")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")

    args = parser.parse_args()

    if args.tune:
        study = optuna.create_study(direction="maximize", study_name="mnist_optuna")
        study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
        print("Best trial:", study.best_trial.number)
        print("  Value (val_acc):", study.best_value)
        print("  Params:", study.best_params)
    else:
        run_single_training(args)


if __name__ == "__main__":
    main()
