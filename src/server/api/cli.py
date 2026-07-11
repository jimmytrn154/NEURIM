"""CLI for the local frontend API bridge."""

import argparse

import uvicorn

from .settings import ApiSettings


def main(argv: list[str] | None = None) -> None:
    settings = ApiSettings.from_env()
    parser = argparse.ArgumentParser(description="Host the local NEURIM frontend API bridge.")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    args = parser.parse_args(argv)
    uvicorn.run("src.server.api.app:app", host=args.host, port=args.port)
