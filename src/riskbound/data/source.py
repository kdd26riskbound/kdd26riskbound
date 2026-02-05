from __future__ import annotations

from typing import Literal, Optional, Callable, Tuple, Dict, Any

import numpy as np
import torch

from riskbound.settings import MIN_TRAIN_STEPS, LOOKBACK_WINDOW, HOLDING_PERIOD, TRAIN_END_DATE, VAL_END_DATE, \
    TEST_END_DATE

class DataSource:
    def __init__(
        self,
        asset_features: np.ndarray,
        market_features: np.ndarray,
        rate_of_returns: np.ndarray,
        dates: np.ndarray,
        *,
        seed: Optional[int] = None,
    ):

        assert MIN_TRAIN_STEPS >= 1, "min_train_steps must be at least 1"

        self.asset_features = np.asarray(asset_features, dtype=np.float32)
        self.rate_of_returns = np.asarray(rate_of_returns, dtype=np.float32)
        self.market_features = np.asarray(market_features, dtype=np.float32)

        self.dates = np.asarray(dates)
        self.dates = self.dates.astype("datetime64[D]")

        N, T, F_a = self.asset_features.shape
        assert self.rate_of_returns.shape == (N, T), (
            f"rate_of_returns must be (N,T), got {self.rate_of_returns.shape}"
        )
        assert self.market_features.shape[0] == T, (
            f"market_features T mismatch: {self.market_features.shape[0]} vs {T}"
        )

        self.N, self.T, self.F_a = N, T, F_a
        self.F_m = self.market_features.shape[1]
        self.L, self.H = LOOKBACK_WINDOW, HOLDING_PERIOD

        self.global_t_min = self.L - 1
        self.global_t_max = self.T - self.H
        assert self.global_t_max >= self.global_t_min, "Not enough history for given L/H"

        self.split = self._build_splits()

        self.mode: Literal["train", "val", "test"] = "train"
        self.t_start, self.t_end = self.split[self.mode]

        self.time_index = self.t_start
        self.reallocation_step = 0

        self.rng = np.random.default_rng(seed)

        self.train_order = None
        self.train_ptr = 0
        self.min_train_steps = MIN_TRAIN_STEPS

    def _build_splits(self) -> Dict[str, Tuple[int, int]]:
        dates = self.dates

        def idx_leq(date_str: str) -> int:
            d = np.datetime64(date_str, "D")
            return int(np.searchsorted(dates, d, side="right") - 1)

        raw_train_end = idx_leq(TRAIN_END_DATE)
        raw_val_end = idx_leq(VAL_END_DATE)
        raw_test_end = idx_leq(TEST_END_DATE)

        t_train_end = min(raw_train_end - self.H, self.global_t_max)
        t_val_end = min(raw_val_end - self.H, self.global_t_max)
        t_test_end = min(raw_test_end - self.H, self.global_t_max)

        t_min = self.global_t_min

        t_train_end = max(t_train_end, t_min)
        t_val_end = max(t_val_end, t_train_end)
        t_test_end = max(t_test_end, t_val_end)

        if t_train_end < t_min:
            raise RuntimeError("Training split is empty. Adjust train_end_date or check data.")
        if t_val_end < t_train_end + 1:
            print("[Warn] Validation split is empty or too short.")
        if t_test_end < t_val_end + 1:
            print("[Warn] Test split is empty or too short.")

        return {
            "train": (t_min, t_train_end),
            "val": (t_train_end + 1, t_val_end),
            "test": (t_val_end + 1, t_test_end),
        }

    def set_mode(self, mode: Literal["train", "val", "test"]):
        self.mode = mode
        self.t_start, self.t_end = self.split[mode]
        self.train_order = None
        self.train_ptr = 0
        return self

    def _get_asset_window(self, t: int) -> np.ndarray:
        return self.asset_features[:, t - self.L + 1 : t + 1, :]

    def _get_market_window(self, t: int) -> np.ndarray:
        return self.market_features[t - self.L + 1 : t + 1, :]

    def _get_label(self, t: int) -> np.ndarray:
        return self.rate_of_returns[:, t : t + self.H]

    def _init_random_initialization(self):
        max_start = self.t_end - (self.min_train_steps - 1) * self.H
        max_start = max(self.t_start, max_start)
        self.train_order = np.arange(self.t_start, max_start + 1, dtype=np.int64)
        if len(self.train_order) == 0:
            raise RuntimeError(
                f"Not enough training steps for random initialization: "
                f"t_start={self.t_start}, t_end={self.t_end}, H={self.H}, "
                f"min_train_steps={self.min_train_steps}, max_start={max_start}"
            )
        self.rng.shuffle(self.train_order)
        self.train_ptr = 0

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.train_order = None
            self.train_ptr = 0

        self.reallocation_step = 0

        if self.mode == "train":
            if self.train_order is None or self.train_ptr >= len(self.train_order):
                self._init_random_initialization()
            self.time_index = int(self.train_order[self.train_ptr])
            self.train_ptr += 1
        else:
            self.time_index = int(self.t_start)

        t = self.time_index
        return {
            "asset_window": self._get_asset_window(t),
            "market_window": self._get_market_window(t),
            "t": t,
            "date": self.dates[t],
        }

    def step(self) -> Tuple[Optional[Dict[str, Any]], bool, Dict[str, Any]]:
        t_prev = int(self.time_index)
        label = self._get_label(t_prev)

        self.reallocation_step += 1
        self.time_index = int(self.time_index + self.H)

        terminated = self.time_index > self.t_end
        if terminated:
            obs = None
        else:
            t = int(self.time_index)
            obs = {
                "asset_window": self._get_asset_window(t),
                "market_window": self._get_market_window(t),
                "t": t,
                "date": self.dates[t],
            }

        info = {"label": label, "t_prev": t_prev}
        return obs, terminated, info

    def dataset(
        self,
        asset_feature_transform_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        market_feature_transform_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        label_transform_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
        *,
        stride: int = 1,
        device: Optional[torch.device | str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        ts = list(range(self.t_start, self.t_end + 1, stride))

        Xs_np, Xm_np, Ys_np = [], [], []
        for t in ts:
            Xs_np.append(self._get_asset_window(t))
            Xm_np.append(self._get_market_window(t))
            Ys_np.append(self._get_label(t))

        Xs = torch.as_tensor(np.stack(Xs_np, axis=0), dtype=torch.float32)
        Xm = torch.as_tensor(np.stack(Xm_np, axis=0), dtype=torch.float32)
        Ys = torch.as_tensor(np.stack(Ys_np, axis=0), dtype=torch.float32)

        if device is not None:
            Xs = Xs.to(device)
            Xm = Xm.to(device)
            Ys = Ys.to(device)

        if asset_feature_transform_fn is not None:
            Xs = asset_feature_transform_fn(Xs)
        if market_feature_transform_fn is not None:
            Xm = market_feature_transform_fn(Xm)
        if label_transform_fn is not None:
            Ys = torch.stack([label_transform_fn(Ys[i], t) for i, t in enumerate(ts)], dim=0)

        return Xs, Xm, Ys
