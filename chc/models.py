"""Small PyTorch message-passing regressor and outcome-trained tabular baselines.

The graph network uses separate incoming/outgoing neighbor means so dependency
direction is not discarded. This is standard message passing, not an architectural
novelty claim. Training targets are observed paired changes in final-answer loss.
"""
from __future__ import annotations

import json
import gc
import random
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from .graph import FEATURE_NAMES, GRAPH_FEATURES, node_features
from .resources import choose_device, seed_everything
from .schema import DecisionState


class BenefitGNN(torch.nn.Module):
    """Two rounds of directed mean message passing followed by scalar regression.

    Args:
        inputs: Number of pre-action node features.
        hidden: Hidden width; 32 is suitable for tiny laptop graph batches.
    Attributes:
        encoder: Maps [N,F] raw features to [N,H] embeddings.
        rounds: Maps concatenated self/incoming/outgoing [N,3H] back to [N,H].
        head: Predicts a scalar benefit at every node; final nodes are loss-masked.
    """
    def __init__(self, inputs: int = 14, hidden: int = 32) -> None:
        """Create standard learnable linear layers without pretrained embeddings.

        Args:
            inputs: Feature dimension of each node.
            hidden: Message-passing embedding dimension.
        Returns:
            None; all parameters are initialized by the current PyTorch RNG.
        """
        super().__init__()
        self.encoder = torch.nn.Linear(inputs, hidden)
        self.rounds = torch.nn.ModuleList([torch.nn.Linear(hidden*3, hidden) for _ in range(2)])
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Propagate directed messages and return one predicted benefit per node.

        Args:
            x: Float node feature matrix [N,F], standardized using training only.
            edge_index: Long tensor [2,E] with source and target row indices.
        Returns:
            Float vector [N]; final-placeholder predictions are ignored in loss.
        Complexity:
            O(N*H^2 + E*H) per round; graphs in a batch have offset indices.
        """
        h = torch.relu(self.encoder(x))
        source, target = edge_index
        for layer in self.rounds:
            incoming, outgoing = torch.zeros_like(h), torch.zeros_like(h)
            inc_count = torch.zeros((len(h), 1), device=h.device, dtype=h.dtype)
            out_count = torch.zeros_like(inc_count)
            if edge_index.numel():
                # index_add sums messages arriving at each target. Dividing by
                # degree prevents high-degree nodes from receiving larger vectors
                # merely because more neighbors exist.
                incoming.index_add_(0, target, h[source])
                outgoing.index_add_(0, source, h[target])
                ones = torch.ones((len(source), 1), device=h.device, dtype=h.dtype)
                inc_count.index_add_(0, target, ones)
                out_count.index_add_(0, source, ones)
            joined = torch.cat([h, incoming/inc_count.clamp_min(1), outgoing/out_count.clamp_min(1)], dim=1)
            h = torch.relu(layer(joined))
        return self.head(h).squeeze(-1)


def training_graph(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Join a visible state to training targets after features have been extracted.

    Args:
        record: Training episode with pre_state, baseline, and singleton outcomes.
    Returns:
        Node features, directed edges, observed benefit labels, and candidate mask.
    Safeguard:
        Outcome labels never enter node_features. Reference answers are not inputs.
    """
    state = DecisionState.from_dict(record['pre_state'])
    x, edges, ids = node_features(state)
    baseline_error = 1-float(record['baseline']['correct'])
    y = np.zeros(len(ids), dtype=np.float32)
    mask = np.zeros(len(ids), dtype=bool)
    for i, node in enumerate(state.claims):
        if not node.is_final and node.id in record['singletons']:
            # V2 joins signed utility labels only after pre-action features exist.
            # A missing factual denominator is excluded rather than set to zero.
            if record.get('supervision_scope') == 'v2_explicit_utility':
                target=record['training_benefits'].get(node.id)
                if target is None:
                    continue
                y[i]=float(target)
            else:
                y[i] = baseline_error-(1-float(record['singletons'][node.id]['correct']))
            mask[i] = True
    return x, edges, y, mask


def fit_tabular(train: list[dict[str, Any]], validation: list[dict[str, Any]], path: Path,
                columns: list[int] | None = None, seed: int = 42) -> dict[str, Any]:
    """Fit a small regressor to measured singleton intervention benefits.

    Args:
        train: Training-question episodes only.
        validation: Separate tuning questions; used to choose regularization.
        path: Destination joblib checkpoint with feature metadata.
        columns: Feature subset for real retrained ablations; None uses all.
        seed: Regressor random seed.
    Returns:
        Selected validation error and training target distribution.
    Raises:
        ValueError: If no candidate targets exist in either partition.
    """
    columns = list(range(len(FEATURE_NAMES))) if columns is None else columns
    def flatten(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        """Concatenate candidate rows without mixing question partitions.

        Args:
            records: Episodes from exactly one predefined partition.
        Returns:
            Matrix [candidate_rows,selected_features] and benefit vector.
        """
        graphs = [training_graph(r) for r in records]
        x = np.concatenate([g[0][g[3]][:, columns] for g in graphs])
        y = np.concatenate([g[2][g[3]] for g in graphs])
        return x, y
    x, y = flatten(train)
    vx, vy = flatten(validation)
    if not len(y) or not len(vy):
        raise ValueError('No observed intervention targets')
    best: tuple[float, Any] | None = None
    for regularization in (1., 5.):
        model = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=7, min_samples_leaf=5,
                                              l2_regularization=regularization, random_state=seed, early_stopping=False)
        model.fit(x, y)
        mse = float(np.mean((model.predict(vx)-vy)**2))
        if best is None or mse < best[0]:
            best = mse, model
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': best[1], 'columns': columns, 'feature_names': FEATURE_NAMES,
                 'seed': seed, 'target': 'observed_final_error_reduction'}, path)
    return {'validation_mse': best[0], 'train_candidates': len(y),
            'target_counts': {str(v): int((y==v).sum()) for v in np.unique(y)}}


def pack_graphs(records: list[dict[str, Any]], mean: np.ndarray, scale: np.ndarray,
                device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a disconnected graph batch with correct node-index offsets.

    Args:
        records: A minibatch of episodes, never randomly mixed train/test records.
        mean: Training-only feature means, shape [F].
        scale: Training-only standard deviations clamped away from zero.
        device: One CUDA device or CPU.
    Returns:
        Features [sum N,F], edges [2,sum E], labels [sum N], mask [sum N].
    """
    xs, es, ys, masks = [], [], [], []
    offset = 0
    for record in records:
        x, edges, y, mask = training_graph(record)
        xs.append((x-mean)/scale)
        es.append(edges+offset)
        ys.append(y)
        masks.append(mask)
        offset += len(x)
    return (torch.tensor(np.concatenate(xs), device=device, dtype=torch.float32),
            torch.tensor(np.concatenate(es, axis=1), device=device, dtype=torch.long),
            torch.tensor(np.concatenate(ys), device=device),
            torch.tensor(np.concatenate(masks), device=device))


def fit_gnn(train: list[dict[str, Any]], validation: list[dict[str, Any]], path: Path,
            seed: int = 42, epochs: int = 25, batch_size: int = 16, accumulation: int = 2,
            device: str = 'auto', resume: bool = False) -> dict[str, Any]:
    """Train graph benefit regression with validation, AMP, and resumable state.

    Args:
        train: Training questions' observed paired trajectories.
        validation: Disjoint tuning questions, used for checkpoint selection.
        path: Best model checkpoint; a sibling .last.pt stores resume state.
        seed: Initialization and minibatch order seed.
        epochs: Total desired epochs including any resumed epochs.
        batch_size: Number of small graphs per forward pass.
        accumulation: Number of minibatches contributing to one optimizer update.
        device: One CUDA device or CPU.
        resume: Restore optimizer, scaler, epoch, and RNGs from .last.pt if present.
    Returns:
        Training/validation log, device, memory peak, and fallback events.
    OOM handling:
        Retry the whole deterministic training run at smaller batch size, then CPU.
    """
    seed_everything(seed)
    target_device = choose_device(device)
    all_x = np.concatenate([training_graph(r)[0] for r in train])
    mean, scale = all_x.mean(axis=0), all_x.std(axis=0).clip(min=1.)
    model = BenefitGNN().to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=target_device.type=='cuda')
    last = path.with_suffix('.last.pt')
    path.parent.mkdir(parents=True, exist_ok=True)
    start_epoch, best, logs = 0, float('inf'), []
    if resume and last.exists():
        checkpoint = torch.load(last, map_location=target_device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch, best, logs = checkpoint['epoch']+1, checkpoint['best'], checkpoint['logs']
        if not np.allclose(mean, checkpoint['mean']) or not np.allclose(scale, checkpoint['scale']):
            raise ValueError('Resume data normalization differs; use the original training split')
        random.setstate(checkpoint['python_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if target_device.type=='cuda' and checkpoint.get('cuda_rng'):
            torch.cuda.set_rng_state_all([r.cpu() for r in checkpoint['cuda_rng']])
    # Explicit tensor references make OOM cleanup safe even if allocation fails
    # before the first complete minibatch. Otherwise a recursive retry could
    # retain the failed run's autograd graph and its GPU parameter references.
    x = edges = y = mask = prediction = loss = payload = None
    try:
        for epoch in range(start_epoch, epochs):
            order = list(range(len(train)))
            random.shuffle(order)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss = 0.
            batches = [order[i:i+batch_size] for i in range(0, len(order), batch_size)]
            for bi, indices in enumerate(batches):
                x, edges, y, mask = pack_graphs([train[i] for i in indices], mean, scale, target_device)
                with torch.autocast(device_type=target_device.type, enabled=target_device.type=='cuda'):
                    prediction = model(x, edges)
                    # MSE trains expected signed benefit, so harmful interventions
                    # have negative targets rather than being silently discarded.
                    loss = torch.nn.functional.mse_loss(prediction[mask].float(), y[mask].float())
                # A partial final accumulation group divides by its actual size.
                group_size = min(accumulation, len(batches)-(bi//accumulation)*accumulation)
                scaler.scale(loss/group_size).backward()
                train_loss += float(loss.detach())
                if (bi+1)%accumulation==0 or bi+1==len(batches):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            model.eval()
            squared, count = 0., 0
            with torch.inference_mode():
                for offset in range(0, len(validation), batch_size):
                    x, edges, y, mask = pack_graphs(validation[offset:offset+batch_size], mean, scale, target_device)
                    pred = model(x, edges)
                    squared += float(((pred[mask]-y[mask])**2).sum())
                    count += int(mask.sum())
            validation_mse = squared/max(count, 1)
            logs.append({'epoch': epoch+1, 'train_batch_mean_mse': train_loss/max(len(batches), 1), 'validation_mse': validation_mse})
            payload = {'model': model.state_dict(), 'mean': mean, 'scale': scale, 'seed': seed,
                       'feature_names': FEATURE_NAMES, 'epoch': epoch, 'optimizer': optimizer.state_dict(),
                       'scaler': scaler.state_dict(), 'best': min(best, validation_mse), 'logs': logs,
                       'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                       'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if target_device.type=='cuda' else None}
            if validation_mse < best:
                best = validation_mse
                torch.save(payload, path)
            torch.save(payload, last)
    except torch.cuda.OutOfMemoryError:
        x = edges = y = mask = prediction = loss = payload = None
        del model, optimizer, scaler
        gc.collect()
        torch.cuda.empty_cache()
        return dict(fit_gnn(train, validation, path, seed, epochs, max(1, batch_size//2), accumulation,
                            'cpu' if batch_size==1 else str(target_device), resume=False), oom_fallback=True)
    result = {'best_validation_mse': best, 'device': str(target_device), 'epochs': epochs, 'seed': seed,
              'batch_size': batch_size, 'gradient_accumulation': accumulation,
              'amp': target_device.type=='cuda', 'gradient_checkpointing': 'not useful for this tiny graph network',
              'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(target_device) if target_device.type=='cuda' else 0,
              'logs': logs}
    path.with_suffix('.training.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


class GraphPolicy:
    """Load a trained directed graph regressor for label-free action scoring.

    Args:
        path: Best checkpoint selected by validation, never by test performance.
        device: Serving execution device; CPU is sufficient for tiny graphs.
    """
    def __init__(self, path: Path, device: str = 'cpu') -> None:
        """Restore trained weights and training-only normalization constants.

        Args:
            path: Locally created PyTorch checkpoint.
            device: Single execution device.
        Returns:
            None; malformed/missing checkpoint errors remain visible.
        """
        self.device = choose_device(device)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.mean, self.scale = checkpoint['mean'], checkpoint['scale']
        self.model = BenefitGNN().to(self.device).eval()
        self.model.load_state_dict(checkpoint['model'])

    def scores(self, state: DecisionState) -> dict[str, float]:
        """Return predicted singleton benefit for each nonfinal visible claim.

        Args:
            state: Pre-action graph without reference or outcome fields.
        Returns:
            Candidate-ID to signed expected-benefit mapping.
        """
        x, edges, ids = node_features(state)
        with torch.inference_mode():
            values = self.model(torch.tensor((x-self.mean)/self.scale, device=self.device),
                                torch.tensor(edges, device=self.device)).cpu().tolist()
        return {c.id: float(values[i]) for i, c in enumerate(state.claims) if not c.is_final}
