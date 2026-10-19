import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import os
from .trainer import Trainer
from prettytable import PrettyTable
from tqdm import tqdm
from .metrics import VRLossMomentum
from ..wandb_interface import log_metrics


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self, num_classes: int, label_smoothing: float = 0.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, targets):
        if targets.ndim == 1:
            return F.cross_entropy(
                logits,
                targets,
                label_smoothing=self.label_smoothing,
            )

        probs = targets.to(dtype=logits.dtype)
        if self.label_smoothing > 0.0:
            probs = probs * (1.0 - self.label_smoothing) + self.label_smoothing / self.num_classes

        log_probs = F.log_softmax(logits, dim=-1)
        return -(probs * log_probs).sum(dim=-1).mean()


class CLFTrainer(Trainer):

    def __init__(self, model, optimizer, train_dataloader, test_dataloader, device, cfg, params, scheduler=None, train_eval_dataloader=None):

        super().__init__(model, optimizer, train_dataloader, test_dataloader, device, cfg, params, scheduler, train_eval_dataloader)

        num_classes = cfg["model"]["num_classes"]
        training_cfg = cfg.get("training", {})
        self.num_classes = int(num_classes)
        self.mixup_alpha = float(training_cfg.get("mixup_alpha", 0.0))
        self.cutmix_alpha = float(training_cfg.get("cutmix_alpha", 0.0))
        self.mix_probability = float(training_cfg.get("mix_probability", 0.0))
        self.cutmix_switch_probability = float(training_cfg.get("cutmix_switch_probability", 0.5))
        self.criterion = SoftTargetCrossEntropy(
            num_classes=self.num_classes,
            label_smoothing=float(training_cfg.get("label_smoothing", 0.0)),
        )

        self.accuracy_metric = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
        ).to(self.device)
        self.top5_accuracy_metric = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
            top_k=min(5, num_classes),
        ).to(self.device)

        self.best_accuracy = 0.0
        self.best_epoch = None
        self.vr_loss_metric = VRLossMomentum()
        self.available_metrics = [
            "Train Loss",
            "Train Accuracy",
            "Test Loss",
            "Test Accuracy",
            "Test Top-5 Accuracy",
            "VR Loss",
        ]
        requested_metrics = cfg.get("training", {}).get("metrics_to_log")
        if requested_metrics:
            requested_set = {str(m) for m in requested_metrics}
            self.metrics_to_log = [m for m in self.available_metrics if m in requested_set]
        else:
            self.metrics_to_log = list(self.available_metrics)
        if not self.metrics_to_log:
            self.metrics_to_log = ["Train Accuracy", "Test Accuracy"]

    def _mixing_enabled(self):
        return self.mix_probability > 0.0 and (self.mixup_alpha > 0.0 or self.cutmix_alpha > 0.0)

    def _sample_beta(self, alpha: float, device):
        concentration = torch.tensor([float(alpha)], device=device)
        return torch.distributions.Beta(concentration, concentration).sample().item()

    def _one_hot(self, labels):
        return F.one_hot(labels, num_classes=self.num_classes).to(dtype=torch.float32)

    def _rand_bbox(self, width: int, height: int, lam: float, device):
        cut_ratio = (1.0 - lam) ** 0.5
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)

        cx = torch.randint(width, (1,), device=device).item()
        cy = torch.randint(height, (1,), device=device).item()

        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, width)
        y2 = min(cy + cut_h // 2, height)
        return x1, y1, x2, y2

    def _maybe_mix_batch(self, inputs, labels):
        if not self._mixing_enabled() or torch.rand((), device=inputs.device).item() > self.mix_probability:
            return inputs, labels

        batch_size = inputs.size(0)
        perm = torch.randperm(batch_size, device=inputs.device)
        labels_a = self._one_hot(labels)
        labels_b = labels_a[perm]

        use_cutmix = (
            self.cutmix_alpha > 0.0
            and (self.mixup_alpha <= 0.0 or torch.rand((), device=inputs.device).item() < self.cutmix_switch_probability)
        )

        if use_cutmix:
            lam = self._sample_beta(self.cutmix_alpha, inputs.device)
            x1, y1, x2, y2 = self._rand_bbox(inputs.size(-1), inputs.size(-2), lam, inputs.device)
            mixed = inputs.clone()
            mixed[:, :, y1:y2, x1:x2] = inputs[perm, :, y1:y2, x1:x2]
            box_area = max(0, x2 - x1) * max(0, y2 - y1)
            lam = 1.0 - box_area / float(inputs.size(-1) * inputs.size(-2))
            targets = labels_a * lam + labels_b * (1.0 - lam)
            return mixed, targets

        lam = self._sample_beta(self.mixup_alpha, inputs.device)
        mixed = inputs * lam + inputs[perm] * (1.0 - lam)
        targets = labels_a * lam + labels_b * (1.0 - lam)
        return mixed, targets

    def make_optimization_step(self, inputs, targets, first_step=False):
        if self.model.training and not first_step:
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            inputs, targets = self._maybe_mix_batch(inputs, targets)
        return super().make_optimization_step(inputs, targets, first_step=first_step)

    # ==========================================================
    # Training
    # ==========================================================
    def train(self, save_csv: bool = True):

        super().train()

        epochs = self.cfg["training"]["epochs"]
        eval_interval = self.cfg["training"]["eval_interval"]
        eval_every = self.cfg["training"].get("eval_every", 1)

        self.model.train()
        self.accuracy_metric.reset()

        # First step for special optimizers
        data, label = next(iter(self.train_dataloader))
        self.make_optimization_step(data, label, first_step=True)

        inputs = data.to(self.device, non_blocking=True)
        labels = label.to(self.device, non_blocking=True)

        with torch.no_grad():
            with self._autocast_context():
                outputs_next = self.model(inputs)
                loss_next = self.criterion(outputs_next, labels)

        metrics_table = PrettyTable()
        iter_title = "Epoch" if eval_interval is None else "Iteration"
        metrics_table.field_names = [iter_title] + self.metrics_to_log

        running_loss = 0.0
        running_correct = 0
        running_total = 0
        global_step = 0

        for epoch in range(epochs):
            if hasattr(self.optimizer, "set_epoch"):
                self.optimizer.set_epoch(epoch + 1)

            if eval_interval is not None:
                for next_inputs, next_labels in tqdm(self.train_dataloader):
                    global_step += 1
                    next_inputs = next_inputs.to(self.device, non_blocking=True)
                    next_labels = next_labels.to(self.device, non_blocking=True)

                    with torch.no_grad():
                        loss_prev = loss_next
                        with self._autocast_context():
                            outputs_next = self.model(next_inputs)
                            loss_next = self.criterion(outputs_next, next_labels)

                    outputs, loss = self.make_optimization_step(inputs, labels)
                    running_loss += loss.item()

                    # Track train accuracy inline for free
                    with torch.no_grad():
                        preds = torch.argmax(outputs, dim=1)
                        running_correct += (preds == labels).sum().item()
                        running_total += labels.size(0)

                    self.vr_loss_metric.update(loss_prev, loss)

                    inputs, labels = next_inputs, next_labels

                    if global_step % eval_interval == 0:
                        train_acc = torch.tensor(running_correct / running_total) if running_total > 0 else None
                        self._evaluation_step(
                            iteration=global_step // eval_interval,
                            total=epochs,
                            interval=eval_interval,
                            running_loss=running_loss,
                            table=metrics_table,
                            iter_title="Iteration",
                            train_accuracy=train_acc,
                        )
                        running_loss = 0.0
                        running_correct = 0
                        running_total = 0
                        self.vr_loss_metric.reset()

            else:
                running_loss = 0.0
                running_correct = 0
                running_total = 0

                for inputs, labels in tqdm(self.train_dataloader):
                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    outputs, loss = self.make_optimization_step(inputs, labels)
                    running_loss += loss.item()

                    # Track train accuracy inline for free
                    with torch.no_grad():
                        preds = torch.argmax(outputs, dim=1)
                        running_correct += (preds == labels).sum().item()
                        running_total += labels.size(0)

                is_last_epoch = (epoch == epochs - 1)
                if (epoch + 1) % eval_every == 0 or is_last_epoch:
                    train_acc = torch.tensor(running_correct / running_total) if running_total > 0 else None
                    self._evaluation_step(
                        iteration=epoch + 1,
                        total=epochs,
                        interval=len(self.train_dataloader),
                        running_loss=running_loss,
                        table=metrics_table,
                        iter_title="Epoch",
                        train_accuracy=train_acc,
                        is_last=is_last_epoch,
                    )
                    running_correct = 0
                    running_total = 0

            if self.scheduler is not None:
                self.scheduler.step()

        if save_csv:
            self.save_metrics_and_params()
        self.finalize()
        print("Finished Training")
        return self.metrics_data

    # ==========================================================
    # Evaluation Step
    # ==========================================================
    def _evaluation_step(self, iteration, total, interval, running_loss, table, iter_title, train_accuracy=None, is_last=False):

        average_loss = running_loss / interval
        vr_loss = self.vr_loss_metric.compute()

        # Only run expensive full train-set eval on the last epoch
        # Otherwise use the train_accuracy passed in from the training loop (computed for free)
        if train_accuracy is None or is_last:
            train_accuracy, _, _ = self._compute_accuracy_on_loader(self.train_eval_dataloader)

        test_accuracy, test_top5_accuracy, test_loss = self.evaluate()

        metrics_snapshot = {
            "Train Loss": average_loss,
            "Train Accuracy": train_accuracy.item(),
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy.item(),
            "Test Top-5 Accuracy": test_top5_accuracy.item(),
            "VR Loss": vr_loss.item(),
        }

        row = [f"{iteration}/{total}"]
        for metric_name in self.metrics_to_log:
            value = metrics_snapshot[metric_name]
            if "Accuracy" in metric_name:
                row.append(f"{value:.2%}")
            else:
                row.append(f"{value:.4f}")
        table.add_row(row)

        with open(self.log_file_path, "a") as log_file:
            log_file.write(f"{table.get_string()}\n\n")

        print(table)
        table.clear_rows()

        metrics_record = {iter_title: iteration}
        metrics_record.update({k: metrics_snapshot[k] for k in self.metrics_to_log})
        self.metrics_data.append(metrics_record)

        if self.use_wandb:
            log_metrics(
                self.wandb_run,
                {k: metrics_snapshot[k] for k in self.metrics_to_log},
            )

        test_accuracy_value = float(test_accuracy.item())
        if test_accuracy_value > self.best_accuracy:
            self.best_accuracy = test_accuracy_value
            self.best_epoch = int(iteration)
            torch.save(
                self.model.state_dict(),
                os.path.join(self.checkpoint_path, "best.pth"),
            )
        if is_last:
            torch.save(
                self.model.state_dict(),
                os.path.join(self.checkpoint_path, "last.pth"),
            )

        self.accuracy_metric.reset()
        self.top5_accuracy_metric.reset()

    # ==========================================================
    # Shared accuracy computation (eval mode, no_grad)
    # ==========================================================
    def _compute_accuracy_on_loader(self, loader):
        self.model.eval()
        self.accuracy_metric.reset()
        self.top5_accuracy_metric.reset()

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with self._autocast_context():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

                predictions = torch.argmax(outputs, dim=1)
                self.accuracy_metric.update(predictions, labels)
                self.top5_accuracy_metric.update(outputs, labels)

        accuracy = self.accuracy_metric.compute()
        top5_accuracy = self.top5_accuracy_metric.compute()
        avg_loss = total_loss / total_samples

        self.model.train()
        self.accuracy_metric.reset()
        self.top5_accuracy_metric.reset()

        return accuracy, top5_accuracy, avg_loss

    # ==========================================================
    # Evaluate
    # ==========================================================
    def evaluate(self):
        return self._compute_accuracy_on_loader(self.test_dataloader)
