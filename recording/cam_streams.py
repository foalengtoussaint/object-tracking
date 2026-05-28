"""Discover ZMQ camera streams from the teleimager config or fall back to defaults.

Usage:
    from cam_streams import load_streams
    streams = load_streams()                        # reads teleimager/cam_config_server.yaml
    streams = load_streams(config="path/to.yaml")   # explicit config
    # streams == {"head_camera": 55555, "left_wrist_camera": 55556, ...}
"""
from __future__ import annotations

from pathlib import Path

_DEFAULT_CONFIG = Path(__file__).parent / "teleimager" / "cam_config_server.yaml"
_FALLBACK = {f"cam_{i}": 55554 + i for i in range(1, 6)}  # cam_1..cam_5 on 55555..55559


def load_streams(config: str | Path | None = None) -> dict[str, int]:
    """Return {stream_name: port} for every ZMQ-enabled camera.

    Reads teleimager/cam_config_server.yaml by default. Falls back to
    five hardcoded defaults if the config is missing or has no ZMQ cameras.
    """
    cfg_path = Path(config) if config else _DEFAULT_CONFIG
    if cfg_path.exists():
        try:
            import yaml
            with open(cfg_path) as fh:
                data = yaml.safe_load(fh)
            streams = {
                name: int(cfg["zmq_port"])
                for name, cfg in data.items()
                if isinstance(cfg, dict) and cfg.get("enable_zmq") and cfg.get("zmq_port")
            }
            if streams:
                print(f"[cam_streams] loaded {len(streams)} stream(s) from {cfg_path}")
                return streams
        except Exception as e:
            print(f"[cam_streams] could not read {cfg_path}: {e} — using defaults")
    else:
        print(f"[cam_streams] {cfg_path} not found — using defaults")
    return dict(_FALLBACK)
