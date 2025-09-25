#!/usr/bin/env python3
"""Serve the point-tracking policy over a lightweight TCP JSON protocol."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation

try:  # tyro is optional; fall back to argparse if unavailable
    import tyro
except ImportError:  # pragma: no cover - optional dependency
    tyro = None

from point_policy import utils
from point_policy.eval_common import ArrayCodec, PointGrounder
from point_policy.eval_point_track import Workspace
from point_policy.robot_utils.franka.utils import matrix_to_rotation_6d


@dataclass
class Args:
    """CLI arguments for the evaluation server."""

    host: str = "0.0.0.0"
    port: int = 8181
    config: str = "config_eval"
    config_dir: Path = Path(__file__).resolve().parent / "cfgs"
    overrides: List[str] = field(default_factory=list)
    bc_weight: Optional[str] = None
    log_level: str = "INFO"


@dataclass
class SessionState:
    step: int = 0
    rotation_6d: Optional[np.ndarray] = None


class EvaluationService:
    """Implements the reset/act handlers used by the TCP server."""

    def __init__(self, workspace: Workspace, point_grounder: PointGrounder) -> None:
        self._workspace = workspace
        self._point_grounder = point_grounder
        self._sessions: Dict[str, SessionState] = {}
        self._primary_key = workspace.cfg.suite.pixel_keys[0]
        self._num_queries = int(workspace.cfg.num_queries)

    def handle_reset(self, message: Mapping[str, Any]) -> Dict[str, Any]:
        session_id = message.get("session_id") or str(uuid.uuid4())
        observation = message.get("observation", {})

        frames = self._decode_arrays(observation.get("frames"))
        depths = self._decode_arrays(observation.get("depths"))
        features = self._decode_array(observation.get("features"))

        self._point_grounder.reset()
        self._workspace.agent.buffer_reset()

        grounding = self._point_grounder.infer(frames, depths or None)
        object_points = grounding.points_3d[self._primary_key]

        rotation = self._features_to_rot6d(features)
        self._sessions[session_id] = SessionState(step=0, rotation_6d=rotation)

        return {
            "type": "reset_ok",
            "session_id": session_id,
            "object_points": ArrayCodec.encode(object_points),
            "num_queries": self._num_queries,
        }

    def handle_act(self, message: Mapping[str, Any]) -> Dict[str, Any]:
        session_id = message.get("session_id")
        if not session_id or session_id not in self._sessions:
            raise KeyError("Unknown or missing session_id. Call reset first.")

        state = self._sessions[session_id]
        observation = message.get("observation", {})

        frames = self._decode_arrays(observation.get("frames"))
        depths = self._decode_arrays(observation.get("depths"))
        features = self._decode_array(observation.get("features"))

        if features is not None:
            rot = self._features_to_rot6d(features)
            if rot is not None:
                state.rotation_6d = rot

        grounding = self._point_grounder.infer(frames, depths or None)
        object_points = grounding.points_3d[self._primary_key]

        obs = {f"point_tracks_{self._primary_key}": object_points.reshape(1, -1)}

        with torch.no_grad(), utils.eval_mode(self._workspace.agent):
            raw_action = self._workspace.agent.act(
                obs,
                None,
                state.step,
                self._workspace.global_step,
                eval_mode=True,
            )

        actions = raw_action.reshape(self._num_queries, self._workspace.agent._act_dim)
        state.step += 1

        actions_payload = []
        for query_idx, action in enumerate(actions):
            action = np.asarray(action)
            pos_ego = action[:3]
            pos_robot = self._workspace.ego_to_robot(pos_ego.copy())
            gripper = float((action[-1] > -0.2) * 2 - 1)
            payload = {
                "query_index": int(query_idx),
                "position_ego": pos_ego.tolist(),
                "position_robot": pos_robot.tolist(),
                "gripper": gripper,
                "raw_policy": action.tolist(),
            }
            if state.rotation_6d is not None:
                payload["rotation_6d"] = state.rotation_6d.tolist()
            actions_payload.append(payload)

        return {
            "type": "actions",
            "session_id": session_id,
            "step": state.step,
            "actions": actions_payload,
            "object_points": ArrayCodec.encode(object_points),
        }

    @staticmethod
    def _decode_arrays(block: Optional[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
        if not block:
            return {}
        return {str(k): ArrayCodec.decode(v) for k, v in block.items()}

    @staticmethod
    def _decode_array(payload: Optional[Mapping[str, Any]]) -> Optional[np.ndarray]:
        if payload is None:
            return None
        return ArrayCodec.decode(payload)

    @staticmethod
    def _features_to_rot6d(features: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if features is None or features.size < 7:
            return None
        quat = np.asarray(features[3:7], dtype=float)
        norm = np.linalg.norm(quat)
        if norm < 1e-6:
            return None
        quat = quat / norm
        return matrix_to_rotation_6d(Rotation.from_quat(quat).as_matrix())


class EvaluationServer:
    """Thin TCP server that exchanges newline-delimited JSON messages."""

    def __init__(self, host: str, port: int, service: EvaluationService) -> None:
        self._host = host
        self._port = port
        self._service = service

    async def run(self) -> None:
        server = await asyncio.start_server(self._handle_client, self._host, self._port)
        addresses = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
        logging.info("Serving evaluation policy on %s", addresses)
        async with server:
            await server.serve_forever()

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        logging.info("Client connected: %s", peer)
        try:
            while True:
                raw = await reader.readline()
                if not raw:
                    break
                try:
                    message = json.loads(raw.decode("utf-8"))
                    response = self._dispatch(message)
                except Exception as exc:  # pragma: no cover - best effort logging
                    logging.exception("Error while handling client request")
                    response = {"type": "error", "message": str(exc)}
                writer.write(json.dumps(response).encode("utf-8") + b"\n")
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            logging.info("Client disconnected: %s", peer)

    def _dispatch(self, message: Mapping[str, Any]) -> Dict[str, Any]:
        msg_type = message.get("type")
        if msg_type == "reset":
            return self._service.handle_reset(message)
        if msg_type == "act":
            return self._service.handle_act(message)
        if msg_type == "ping":
            return {"type": "pong"}
        raise ValueError(f"Unsupported message type: {msg_type}")


def load_cfg(args: Args) -> DictConfig:
    config_dir = str(Path(args.config_dir).resolve())
    with initialize_config_dir(config_dir=config_dir, job_name="eval_server"):
        cfg = compose(config_name=args.config, overrides=list(args.overrides))
    if args.bc_weight is not None:
        cfg.bc_weight = args.bc_weight
    return cfg


def resolve_bc_snapshot(cfg: DictConfig) -> Path:
    snapshot = Path(cfg.bc_weight).expanduser().resolve()
    if not snapshot.exists():
        raise FileNotFoundError(f"bc weight not found: {snapshot}")
    return snapshot


def setup_workspace(cfg: DictConfig) -> Workspace:
    workspace = Workspace(cfg)
    snapshots: Dict[str, Path] = {"bc": resolve_bc_snapshot(cfg)}
    workspace.load_snapshot(snapshots)
    return workspace


def main(args: Args) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logging.info("Loading configuration from %s", args.config_dir)

    cfg = load_cfg(args)
    workspace = setup_workspace(cfg)
    point_grounder = PointGrounder(workspace.cfg)

    service = EvaluationService(workspace, point_grounder)
    server = EvaluationServer(args.host, args.port, service)

    hostname = socket.gethostname()
    logging.info("Hostname: %s", hostname)

    torch.backends.cudnn.benchmark = True

    asyncio.run(server.run())


if __name__ == "__main__":
    if tyro is not None:
        main(tyro.cli(Args))
    else:  # pragma: no cover - fallback when tyro is missing
        import argparse

        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", type=int, default=8181)
        parser.add_argument("--config", default="config_eval")
        parser.add_argument(
            "--config-dir",
            default=str(Path(__file__).resolve().parent / "cfgs"),
        )
        parser.add_argument("--override", action="append", dest="overrides", default=[])
        parser.add_argument("--bc-weight", dest="bc_weight")
        parser.add_argument("--log-level", default="INFO")
        parsed = parser.parse_args()
        main(
            Args(
                host=parsed.host,
                port=parsed.port,
                config=parsed.config,
                config_dir=Path(parsed.config_dir),
                overrides=parsed.overrides,
                bc_weight=parsed.bc_weight,
                log_level=parsed.log_level,
            )
        )
