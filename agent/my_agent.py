"""Clean StochasticGoose-style ARC-AGI-3 baseline.

This agent follows the public StochasticGoose idea: learn online which
actions tend to change the current frame, then sample from those predictions.

References:
  - https://www.kaggle.com/code/inversion/arc3-sample-submission-stochastic-goose
  - https://github.com/DriesSmit/ARC3-solution
  - https://medium.com/@dries.epos/1st-place-in-the-arc-agi-3-agent-preview-competition-49263f6287db

This is a clean-room, single-file implementation for this repository. It keeps
the core algorithm but omits visualization, TensorBoard, file logging, and
other experiment artifacts so the Kaggle submission remains easy to inspect.
"""
from __future__ import annotations

import hashlib
import random
import time
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent


GRID_SIZE = 64
NUM_COLOURS = 16
NUM_COORDINATES = GRID_SIZE * GRID_SIZE
SIMPLE_ACTIONS = (
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
    GameAction.ACTION4,
    GameAction.ACTION5,
)
UNDO_ACTION = GameAction.ACTION7


class ActionEffectModel(nn.Module):
    """CNN that predicts frame-changing simple actions and click coordinates."""

    def __init__(self, input_channels: int = NUM_COLOURS) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.action_pool = nn.MaxPool2d(4, 4)
        self.action_fc = nn.Linear(256 * 16 * 16, 512)
        self.action_head = nn.Linear(512, len(SIMPLE_ACTIONS))

        self.coord_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.coord_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.coord_conv3 = nn.Conv2d(64, 32, kernel_size=1)
        self.coord_conv4 = nn.Conv2d(32, 1, kernel_size=1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        features = F.relu(self.conv4(x))

        action_features = self.action_pool(features).flatten(start_dim=1)
        action_features = F.relu(self.action_fc(action_features))
        action_features = self.dropout(action_features)
        action_logits = self.action_head(action_features)

        coord_features = F.relu(self.coord_conv1(features))
        coord_features = F.relu(self.coord_conv2(coord_features))
        coord_features = F.relu(self.coord_conv3(coord_features))
        coord_logits = self.coord_conv4(coord_features).flatten(start_dim=1)

        return torch.cat([action_logits, coord_logits], dim=1)


class MyAgent(Agent):
    """Online action-effect learner based on the StochasticGoose baseline."""

    MAX_ACTIONS = 1_000_000
    _MAX_FRAMES = 10
    _REPLAY_CAPACITY = 20_000
    _BATCH_SIZE = 64
    _TRAIN_FREQUENCY = 5
    _LEARNING_RATE = 1e-4
    _ACTION_ENTROPY_WEIGHT = 1e-4
    _COORD_ENTROPY_WEIGHT = 1e-5
    _RUNTIME_LIMIT_SECONDS = 8 * 3600 - 5 * 60

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed % (2**32 - 1))

        self.start_time = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_level = -1
        self.model: ActionEffectModel | None = None
        self.optimizer: optim.Optimizer | None = None
        self.experience_buffer: deque[tuple[np.ndarray, int, float]] = deque(
            maxlen=self._REPLAY_CAPACITY
        )
        self.experience_hashes: set[str] = set()
        self.prev_grid: np.ndarray | None = None
        self.prev_action_idx: int | None = None

    @property
    def name(self) -> str:
        return f"{super().name}.stochastic_goose"

    def append_frame(self, frame: FrameData) -> None:
        self.frames.append(frame)
        if len(self.frames) > self._MAX_FRAMES:
            self.frames = self.frames[-self._MAX_FRAMES :]
        if frame.guid:
            self.guid = frame.guid
        if hasattr(self, "recorder") and not self.is_playback:
            import json

            self.recorder.record(json.loads(frame.model_dump_json()))

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        try:
            return (
                latest_frame.state is GameState.WIN
                or time.time() - self.start_time >= self._RUNTIME_LIMIT_SECONDS
            )
        except Exception:
            return True

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        try:
            level = int(getattr(latest_frame, "levels_completed", 0) or 0)
            if level != self.current_level:
                self._reset_level_learner(level)

            if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                self.prev_grid = None
                self.prev_action_idx = None
                action = GameAction.RESET
                action.reasoning = {"policy": "reset", "state": latest_frame.state.name}
                return action

            current_grid = self._frame_to_grid(latest_frame)
            self._record_previous_experience(current_grid)

            if self.model is None:
                self._reset_level_learner(level)
            action, action_idx = self._sample_action(current_grid, latest_frame)

            self.prev_grid = current_grid.copy()
            self.prev_action_idx = action_idx

            if self.action_counter % self._TRAIN_FREQUENCY == 0:
                self._train_action_model()

            return action
        except Exception as exc:
            action = self._fallback_action(getattr(latest_frame, "available_actions", None))
            action.reasoning = {"policy": "fallback", "error": type(exc).__name__}
            return action

    def _reset_level_learner(self, level: int) -> None:
        self.current_level = level
        self.model = ActionEffectModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self._LEARNING_RATE)
        self.experience_buffer.clear()
        self.experience_hashes.clear()
        self.prev_grid = None
        self.prev_action_idx = None

    def _frame_to_grid(self, frame_data: FrameData) -> np.ndarray:
        frame = np.asarray(frame_data.frame, dtype=np.int64)
        if frame.ndim == 3:
            frame = frame[-1]
        if frame.shape != (GRID_SIZE, GRID_SIZE):
            raise ValueError(f"expected {GRID_SIZE}x{GRID_SIZE} frame, got {frame.shape}")
        if frame.min() < 0 or frame.max() >= NUM_COLOURS:
            raise ValueError("frame contains colour ids outside the 0-15 range")
        return frame.astype(np.uint8, copy=False)

    def _grid_batch_to_tensor(self, grids: np.ndarray) -> torch.Tensor:
        grid_tensor = torch.as_tensor(grids, dtype=torch.long, device=self.device)
        return F.one_hot(grid_tensor, num_classes=NUM_COLOURS).permute(0, 3, 1, 2).float()

    def _record_previous_experience(self, current_grid: np.ndarray) -> None:
        if self.prev_grid is None or self.prev_action_idx is None:
            return

        experience_hash = self._experience_hash(self.prev_grid, self.prev_action_idx)
        if experience_hash in self.experience_hashes:
            return

        frame_changed = not np.array_equal(self.prev_grid, current_grid)
        reward = 1.0 if frame_changed else 0.0
        self.experience_buffer.append((self.prev_grid.copy(), self.prev_action_idx, reward))
        self.experience_hashes.add(experience_hash)

    def _experience_hash(self, grid: np.ndarray, action_idx: int) -> str:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(grid.tobytes())
        digest.update(str(action_idx).encode("utf-8"))
        return digest.hexdigest()

    def _sample_action(
        self, current_grid: np.ndarray, latest_frame: FrameData
    ) -> tuple[GameAction, int | None]:
        if self.model is None:
            return self._fallback_action_with_index(latest_frame.available_actions)

        state = self._grid_batch_to_tensor(current_grid[None, :, :])
        with torch.no_grad():
            logits = self.model(state).squeeze(0)
        logits = self._mask_unavailable_logits(logits, latest_frame.available_actions)
        probs = self._logits_to_sampling_probs(logits)
        if probs is None:
            return self._fallback_action_with_index(latest_frame.available_actions)

        selected_idx = int(np.random.choice(len(probs), p=probs))
        if selected_idx < len(SIMPLE_ACTIONS):
            action = SIMPLE_ACTIONS[selected_idx]
            action.reasoning = {
                "policy": "stochastic_goose",
                "probability": round(float(probs[selected_idx]), 6),
            }
            return action, selected_idx

        coord_idx = selected_idx - len(SIMPLE_ACTIONS)
        y = coord_idx // GRID_SIZE
        x = coord_idx % GRID_SIZE
        action = GameAction.ACTION6
        action.set_data({"x": int(x), "y": int(y)})
        action.reasoning = {
            "policy": "stochastic_goose",
            "x": int(x),
            "y": int(y),
            "probability": round(float(probs[selected_idx]), 6),
        }
        return action, selected_idx

    def _mask_unavailable_logits(
        self, logits: torch.Tensor, available_actions: Any
    ) -> torch.Tensor:
        action_ids = self._available_action_ids(available_actions)
        if not action_ids:
            return logits

        masked = logits.clone()
        for idx, action in enumerate(SIMPLE_ACTIONS):
            if action.value not in action_ids:
                masked[idx] = -torch.inf
        if GameAction.ACTION6.value not in action_ids:
            masked[len(SIMPLE_ACTIONS) :] = -torch.inf
        return masked

    def _logits_to_sampling_probs(self, logits: torch.Tensor) -> np.ndarray | None:
        action_probs = torch.sigmoid(logits[: len(SIMPLE_ACTIONS)])
        coord_probs = torch.sigmoid(logits[len(SIMPLE_ACTIONS) :]) / NUM_COORDINATES
        combined = torch.cat([action_probs, coord_probs])

        total = combined.sum()
        if not torch.isfinite(total) or total <= 0:
            return None

        probs = (combined / total).detach().float().cpu().numpy()
        if not np.isfinite(probs).all() or probs.sum() <= 0:
            return None
        return probs / probs.sum()

    def _train_action_model(self) -> None:
        if (
            self.model is None
            or self.optimizer is None
            or len(self.experience_buffer) < self._BATCH_SIZE
        ):
            return

        batch = random.sample(list(self.experience_buffer), self._BATCH_SIZE)
        grids = np.stack([item[0] for item in batch], axis=0)
        action_indices = torch.tensor(
            [item[1] for item in batch], dtype=torch.long, device=self.device
        )
        rewards = torch.tensor(
            [item[2] for item in batch], dtype=torch.float32, device=self.device
        )

        states = self._grid_batch_to_tensor(grids)
        logits = self.model(states)
        selected_logits = logits.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        main_loss = F.binary_cross_entropy_with_logits(selected_logits, rewards)

        probabilities = torch.sigmoid(logits)
        action_entropy = probabilities[:, : len(SIMPLE_ACTIONS)].mean()
        coord_entropy = probabilities[:, len(SIMPLE_ACTIONS) :].mean()
        loss = (
            main_loss
            - self._ACTION_ENTROPY_WEIGHT * action_entropy
            - self._COORD_ENTROPY_WEIGHT * coord_entropy
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _available_action_ids(self, available_actions: Any) -> set[int]:
        if not available_actions:
            return set()

        ids: set[int] = set()
        for action in available_actions:
            value = getattr(action, "value", action)
            try:
                ids.add(int(value))
            except (TypeError, ValueError):
                continue
        return ids

    def _fallback_action(self, available_actions: Any) -> GameAction:
        return self._fallback_action_with_index(available_actions)[0]

    def _fallback_action_with_index(
        self, available_actions: Any
    ) -> tuple[GameAction, int | None]:
        action_ids = self._available_action_ids(available_actions)
        if action_ids:
            choices = [action for action in SIMPLE_ACTIONS if action.value in action_ids]
            if choices:
                action = random.choice(choices)
                return action, SIMPLE_ACTIONS.index(action)
            if GameAction.ACTION6.value in action_ids:
                action = GameAction.ACTION6
                x = random.randrange(GRID_SIZE)
                y = random.randrange(GRID_SIZE)
                action.set_data(
                    {"x": x, "y": y}
                )
                return action, len(SIMPLE_ACTIONS) + y * GRID_SIZE + x
            if UNDO_ACTION.value in action_ids:
                return UNDO_ACTION, None
            if GameAction.RESET.value in action_ids:
                return GameAction.RESET, None

        action = random.choice(SIMPLE_ACTIONS)
        return action, SIMPLE_ACTIONS.index(action)
