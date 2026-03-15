from __future__ import annotations

import argparse
import socket
from pathlib import Path
from wsgiref.simple_server import make_server

from pdkzero.web.app import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PDKZero LAN web PvE server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind, default 0.0.0.0")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind, default 8000")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/model.pt",
        help="Path to the trained checkpoint used by the three AI seats",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    app = create_app(Path(args.checkpoint))
    local_ip = _get_local_ip()
    print(f"Serving PDKZero Web PvE on http://{args.host}:{args.port}")
    print(f"LAN access: http://{local_ip}:{args.port}")
    with make_server(args.host, args.port, app) as server:
        server.serve_forever()


def _get_local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


if __name__ == "__main__":
    main()
