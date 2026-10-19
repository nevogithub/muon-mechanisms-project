import os
import json
import csv
from contextlib import nullcontext
import torch
from torch.nn.utils import clip_grad_norm_
from ..wandb_interface import init_training_run, finish_run


class Trainer:
    def __init__(self, model, optimizer, train_dataloader, test_dataloader, device, cfg, params, scheduler=None, train_eval_dataloader=None):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        # Clean eval loader for train accuracy — no augmentation
        self.train_eval_dataloader = train_eval_dataloader if train_eval_dataloader is not None else train_dataloader
        self.device = device
        self.model.to(self.device)
        self.criterion = None
        self.scheduler = scheduler
        configured_clip = cfg.get("training", {}).get("grad_clip")
        self.clip_grad_norm = (
            getattr(params, "clip_grad_norm", None)
            if getattr(params, "clip_grad_norm", None) is not None
            else configured_clip
        )
        # This trainer does not use a GradScaler. BF16 is safe without one;
        # silently using FP16 here can underflow gradients on older GPUs.
        self.autocast_enabled = (
            self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        )
        self.autocast_dtype = torch.bfloat16

        self.cfg = cfg
        self.params = params.__dict__
        self.metrics_data = []

        self.checkpoint_path = None
        self.log_file_path = None
        self.run_directory = "./"
        self.wandb_run = None

        # ------------------------------------------------------
        # WandB
        # ------------------------------------------------------
        wandb_cfg = cfg.get("wandb", {})
        self.use_wandb = wandb_cfg.get("enabled", False)

        if self.use_wandb:
            self.wandb_run = init_training_run(cfg, self.params)
            self.use_wandb = self.wandb_run is not None

    # ==========================================================
    # Training Setup
    # ==========================================================
    def train(self):
        """
        Prepare run directory and save config.
        """
        # Use per-experiment results dir if provided (for parallel runs)
        base_results_dir = self.cfg.get("results_dir", "results")
        os.makedirs(base_results_dir, exist_ok=True)

        existing_folders = [
            d for d in os.listdir(base_results_dir)
            if os.path.isdir(os.path.join(base_results_dir, d))
        ]

        run_id = len(existing_folders) + 1
        run_directory = os.path.join(base_results_dir, f"run{run_id}")
        os.makedirs(run_directory)

        self.checkpoint_path = os.path.join(run_directory, "checkpoints")
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.log_file_path = os.path.join(run_directory, "metrics_log.txt")
        self.run_directory = run_directory

        # Save full merged config
        with open(os.path.join(run_directory, "config.json"), "w") as f:
            json.dump({
                "config": self.cfg,
                "cli_args": self.params,
            }, f, indent=4)

    # ==========================================================
    # Evaluation Placeholder
    # ==========================================================
    def evaluate(self):
        pass

    # ==========================================================
    # Save Metrics
    # ==========================================================
    def save_metrics_and_params(self):

        csv_file_path = os.path.join(self.run_directory, "results.csv")

        if not self.metrics_data:
            print("No metrics data to save.")
            return

        combined_rows = [{**metrics, **self.params} for metrics in self.metrics_data]

        fieldnames = list(combined_rows[0].keys())

        with open(csv_file_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_rows)

        diagnostic_history = getattr(self.optimizer, "diagnostic_history", None)
        if diagnostic_history:
            diagnostic_path = os.path.join(self.run_directory, "optimizer_diagnostics.csv")
            with open(diagnostic_path, "w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(diagnostic_history[0].keys()))
                writer.writeheader()
                writer.writerows(diagnostic_history)

    def finalize(self):
        finish_run(self.wandb_run)

    def _autocast_context(self):
        if not self.autocast_enabled:
            return nullcontext()
        return torch.amp.autocast(
            device_type="cuda",
            dtype=self.autocast_dtype,
        )

    # ==========================================================
    # Optimization Step
    # ==========================================================
    def make_optimization_step(self, inputs, targets, first_step=False):

        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        outputs = None
        loss = None

        optimizer_name = self.optimizer.__class__.__name__

        if optimizer_name in ["Mu2SGD", "STORM"]:
            if first_step:
                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast_context():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                loss.backward()
                if self.clip_grad_norm:
                    clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
                self.optimizer.compute_estimator()
            else:
                # First pass
                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast_context():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                loss.backward()
                if self.clip_grad_norm:
                    clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
                self.optimizer.step()

                # Second pass
                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast_context():
                    outputs2 = self.model(inputs)
                    loss2 = self.criterion(outputs2, targets)
                loss2.backward()
                if self.clip_grad_norm:
                    clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
                self.optimizer.compute_estimator()

        elif optimizer_name in ["MARS"]:
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer.update_last_grad()

        elif optimizer_name in ["AdaStormW", "AdaMu2W"]:
            def closure():
                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast_context():
                    out = self.model(inputs)
                    l = self.criterion(out, targets)
                l.backward()
                if self.clip_grad_norm:
                    clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
                return l

            loss = self.optimizer.step(closure)
            with torch.no_grad():
                with self._autocast_context():
                    outputs = self.model(inputs)

        else:
            if not first_step:
                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast_context():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                loss.backward()
                if self.clip_grad_norm:
                    clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
                self.optimizer.step()

        return outputs, loss
