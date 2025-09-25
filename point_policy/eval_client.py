#!/usr/bin/env python3
"""Client helper to query the point policy evaluation server."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

try:  # optional dependency for nicer CLI
    import tyro
except ImportError:  # pragma: no cover - tyro is optional
    tyro = None

from point_policy.eval_common import ArrayCodec


@dataclass
class ObservationPayload:
    """Holds raw numpy arrays for a single observation."""

    frames: Mapping[str, np.ndarray]
    depths: Optional[Mapping[str, np.ndarray]] = None
    features: Optional[np.ndarray] = None

    def to_message(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.frames:
            payload["frames"] = {
                key: ArrayCodec.encode(np.asarray(value)) for key, value in self.frames.items()
            }
        if self.depths:
            payload["depths"] = {
                key: ArrayCodec.encode(np.asarray(value)) for key, value in self.depths.items()
            }
        if self.features is not None:
            payload["features"] = ArrayCodec.encode(np.asarray(self.features))
        return payload

    @staticmethod
    def from_npz(path: Path) -> "ObservationPayload":
        data = np.load(path, allow_pickle=True)
        frames: Dict[str, np.ndarray] = {}
        depths: Dict[str, np.ndarray] = {}
        features: Optional[np.ndarray] = None

        for key in data.files:
            value = data[key]
            if key.startswith("pixels"):
                frames[key] = value
            elif key.startswith("depth"):
                depths[key] = value
            elif key == "features":
                features = value

        return ObservationPayload(
            frames=frames,
            depths=depths or None,
            features=features,
        )


class EvaluationClient:
    """Simple asyncio-based TCP client to talk to the evaluation server."""

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

    async def __aenter__(self) -> "EvaluationClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def connect(self) -> None:
        if self._reader is None or self._writer is None:
            self._reader, self._writer = await asyncio.open_connection(self._host, self._port)

    async def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            await self._writer.wait_closed()
        self._reader = None
        self._writer = None

    async def reset(
        self, observation: ObservationPayload, session_id: Optional[str] = None
    ) -> Mapping[str, Any]:
        message: Dict[str, Any] = {"type": "reset", "observation": observation.to_message()}
        if session_id is not None:
            message["session_id"] = session_id
        return await self._send(message)

    async def act(
        self, observation: ObservationPayload, session_id: str
    ) -> Mapping[str, Any]:
        message: Dict[str, Any] = {
            "type": "act",
            "session_id": session_id,
            "observation": observation.to_message(),
        }
        return await self._send(message)

    async def ping(self) -> Mapping[str, Any]:
        return await self._send({"type": "ping"})

    async def _send(self, message: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._reader is None or self._writer is None:
            await self.connect()
        assert self._reader is not None and self._writer is not None

        payload = json.dumps(message).encode("utf-8") + b"\n"
        self._writer.write(payload)
        await self._writer.drain()

        response = await self._reader.readline()
        if not response:
            raise ConnectionError("Server closed the connection")
        return json.loads(response.decode("utf-8"))


@dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8181
    mode: str = "reset"
    session_id: Optional[str] = None
    observation_npz: Optional[Path] = None
    output_path: Optional[Path] = None


async def run_client(args: Args) -> None:
    async with EvaluationClient(args.host, args.port) as client:
        if args.mode == "ping":
            response = await client.ping()
        else:
            if args.observation_npz is None:
                raise ValueError("observation_npz is required for reset/act modes")
            observation = ObservationPayload.from_npz(args.observation_npz)
            if args.mode == "reset":
                response = await client.reset(observation, args.session_id)
            elif args.mode == "act":
                if args.session_id is None:
                    raise ValueError("session_id is required for act mode")
                response = await client.act(observation, args.session_id)
            else:
                raise ValueError(f"Unsupported mode: {args.mode}")

        pretty = json.dumps(response, indent=2)
        print(pretty)
        if args.output_path is not None:
            args.output_path.write_text(pretty + "\n", encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    if tyro is not None:
        args = tyro.cli(Args)
    else:  # pragma: no cover - argparse fallback when tyro missing
        import argparse

        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8181)
        parser.add_argument("--mode", choices=["reset", "act", "ping"], default="reset")
        parser.add_argument("--session-id")
        parser.add_argument("--observation-npz")
        parser.add_argument("--output-path")
        ns = parser.parse_args()
        args = Args(
            host=ns.host,
            port=ns.port,
            mode=ns.mode,
            session_id=ns.session_id,
            observation_npz=Path(ns.observation_npz) if ns.observation_npz else None,
            output_path=Path(ns.output_path) if ns.output_path else None,
        )

    asyncio.run(run_client(args))


if __name__ == "__main__":
    main()
