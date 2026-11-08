"""Pretrained time-series foundation model: IBM Granite TinyTimeMixer (TTM).

TTM (Ekambaram et al., 2024, "Tiny Time Mixers (TTMs): Fast Pre-trained Models
for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series") is a
compact (~1M parameter) MLP-Mixer model from IBM Research, pretrained on public
time-series corpora. Here it is used **zero-shot**: the benchmark data is
never used to train or fine-tune it, it only conditions on the history it is
given at each forecast origin. That makes it an honest comparison point
against models that are refit on every fold.

Which pretrained variant?
-------------------------
TTM-r2 ships variants for several context lengths. The shorter-context ones
(``*-ft-r2.1``) were trained with a *frequency token* that only covers
minutely..weekly data; monthly and yearly series would get the
"out-of-vocabulary" token, which is out of distribution for those weights.
So the benchmark uses the plain ``512-96-r2`` variant (revision ``main``),
which has no frequency token, for every dataset. Histories shorter than 512
are left-padded (with the window mean, i.e. zero after TTM's internal std
scaling) and flagged as unobserved via ``past_observed_mask`` - the
"zeropad" mode documented by granite-tsfm.
This choice was made from the documentation, *before* looking at test scores.

Optional dependency:  pip install -e ".[foundation]"   (torch + granite-tsfm)
Weights (~3 MB) are downloaded once from Hugging Face and cached locally.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np

from .models import FOUNDATION, Forecaster

log = logging.getLogger(__name__)

TTM_REPO = "ibm-granite/granite-timeseries-ttm-r2"
TTM_REVISION = "main"  # = the 512-96-r2 variant, no frequency token
TTM_CONTEXT = 512


class GraniteTTM(Forecaster):
    """Zero-shot IBM Granite TinyTimeMixer forecaster."""

    name, family = "TTM", FOUNDATION
    parallel_safe = False  # holds torch weights; run in the main process

    def __init__(self, context_length: int = TTM_CONTEXT, repo: str = TTM_REPO,
                 revision: str = TTM_REVISION):
        self.context_length = context_length
        self.repo = repo
        self.revision = revision
        self._models: dict[int, object] = {}

    def __getstate__(self):  # never pickle torch weights
        state = self.__dict__.copy()
        state["_models"] = {}
        return state

    def _model(self, horizon: int):
        if horizon not in self._models:
            try:
                from tsfm_public.toolkit.get_model import get_model
            except ImportError as e:  # pragma: no cover - depends on environment
                raise ImportError(
                    'TTM needs the optional dependencies: pip install -e ".[foundation]"') from e
            for name in ("tsfm_public", "transformers", "huggingface_hub", "httpx"):
                logging.getLogger(name).setLevel(logging.ERROR)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = get_model(self.repo, context_length=self.context_length,
                                  prediction_length=horizon, model_revision=self.revision)
            model.eval()
            self._models[horizon] = model
            log.info("loaded Granite TTM (%s@%s, context %d, horizon %d)",
                     self.repo, self.revision, self.context_length, horizon)
        return self._models[horizon]

    def _inputs(self, histories: list[np.ndarray]) -> dict:
        """Batch of model inputs for one or more histories (each its own window)."""
        import torch

        c = self.context_length
        windows = np.empty((len(histories), c))
        observed = np.ones((len(histories), c), dtype=bool)
        for i, y in enumerate(histories):
            w = np.asarray(y[-c:], dtype=float)
            if len(w) < c:
                # "zeropad" in *standardised* units: TTM std-scales its input
                # internally, so pad with the window mean (which scales to exactly
                # 0) and mask the padding out of the scaler statistics. Padding raw
                # zeros would feed huge outliers (0 - mean) / std into the network.
                pad = c - len(w)
                observed[i, :pad] = False
                w = np.concatenate([np.full(pad, w.mean()), w])
            windows[i] = w
        return dict(past_values=torch.tensor(windows, dtype=torch.float32).unsqueeze(-1),
                    past_observed_mask=torch.tensor(observed).unsqueeze(-1),
                    return_loss=False)

    @staticmethod
    def _call(model, inputs: dict):
        import torch

        if getattr(model.config, "resolution_prefix_tuning", False):
            n = inputs["past_values"].shape[0]
            inputs = {**inputs, "freq_token": torch.zeros(n, dtype=torch.long)}  # "oov"
        return model(**inputs)

    def _predict(self, model, train: np.ndarray, horizon: int) -> np.ndarray:
        import torch

        model.eval()
        with torch.no_grad():
            out = self._call(model, self._inputs([np.asarray(train, dtype=float)]))
        pred = out.prediction_outputs[0, :, 0].cpu().numpy().astype(float)
        if len(pred) < horizon:
            raise ValueError(f"TTM returned {len(pred)} steps, need {horizon}")
        return pred[:horizon]

    def forecast(self, train, horizon):
        return self._predict(self._model(horizon), train, horizon)


class GraniteTTMFineTuned(GraniteTTM):
    """Few-shot Granite TTM: fine-tuned on each fold's *training data only*.

    IBM's recommended recipe for small data: freeze the pretrained backbone and
    train only the decoder + forecast head (~290k of ~805k parameters). Every
    call starts again from the pretrained weights, so nothing learned in one
    fold (or one conformal calibration window) leaks into another.

    Training samples are all (history -> next ``horizon`` values) pairs that fit
    entirely inside ``train``; the loss is MAE in standardised units, so the
    learning rate means the same thing on every dataset. A fixed small budget
    (epochs, lr) is used instead of early stopping, because carving a
    validation set out of a short series would cost the most recent data.
    """

    name = "TTM-FT"
    parallel_safe = True  # each worker process loads its own copy of the weights

    def __init__(self, epochs: int = 10, lr: float = 1e-3, batch_size: int = 64,
                 stride: int = 1, min_history: int = 24, seed: int = 0, **kw):
        super().__init__(**kw)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.stride = stride
        self.min_history = min_history
        self.seed = seed
        self._pretrained: dict[int, dict] = {}

    def __getstate__(self):
        state = super().__getstate__()
        state["_pretrained"] = {}
        return state

    def _fresh_model(self, horizon: int):
        """The cached model, reset to the pretrained weights."""
        import copy

        model = self._model(horizon)
        if horizon not in self._pretrained:
            self._pretrained[horizon] = copy.deepcopy(model.state_dict())
        model.load_state_dict(self._pretrained[horizon])
        return model

    def _samples(self, y: np.ndarray, horizon: int) -> tuple[list[np.ndarray], np.ndarray]:
        ends = range(self.min_history, len(y) - horizon + 1, self.stride)
        return [y[:t] for t in ends], np.array([y[t:t + horizon] for t in ends])

    def forecast(self, train, horizon):
        import torch

        y = np.asarray(train, dtype=float)
        model = self._fresh_model(horizon)
        histories, targets = self._samples(y, horizon)
        if len(histories) < 8:  # too little data to fine-tune: fall back to zero-shot
            return self._predict(model, y, horizon)

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        for p in model.backbone.parameters():
            p.requires_grad = False
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr)
        inputs = self._inputs(histories)
        target = torch.tensor(targets, dtype=torch.float32)
        model.train()
        for _ in range(self.epochs):
            for idx in np.array_split(rng.permutation(len(histories)),
                                      max(1, len(histories) // self.batch_size)):
                batch = {k: v[idx] if torch.is_tensor(v) else v for k, v in inputs.items()}
                out = self._call(model, batch)
                pred = out.prediction_outputs[:, :horizon, 0]
                scale = out.scale[:, 0, :1]  # per-window std used by TTM's own scaler
                loss = ((pred - target[idx]).abs() / scale).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self._predict(model, y, horizon)
