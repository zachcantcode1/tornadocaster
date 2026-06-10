"""Post forecast products only when a new run/outlook is available."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from src.sources.nadocast import NadocastRequest, NadocastSource
from src.sources.spc_outlook import SpcOutlookSource


_DEFAULT_PRODUCTS = ("nadocast", "spc-tornado")
_WEBHOOK_ENV = {
    "nadocast": "IFTTT_NADOCAST_WEBHOOK_URL",
    "spc-tornado": "IFTTT_SPC_TORNADO_WEBHOOK_URL",
}


@dataclass(frozen=True)
class ProductStatus:
    product: str
    product_id: str
    label: str


async def latest_product_status(product: str) -> ProductStatus:
    if product == "nadocast":
        source = NadocastSource()
        request = await source.find_latest(NadocastRequest(hazard="tornado"))
        request = await source.resolve_request(request)
        product_id = request.filename or f"{request.run_date}_{request.cycle}_{request.window}"
        return ProductStatus(product=product, product_id=product_id, label=f"NADOCast {request.run_date} {request.cycle}Z")

    if product == "spc-tornado":
        outlook = await SpcOutlookSource().fetch_day1("tornado")
        product_id = outlook.issue_iso or f"{outlook.valid_iso}_{outlook.expire_iso}_{len(outlook.polygons)}"
        return ProductStatus(product=product, product_id=product_id, label=outlook.valid_period_label)

    raise ValueError(f"Unsupported product: {product}")


def load_state(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def webhook_url_for(product: str, args: argparse.Namespace) -> str:
    explicit = {
        "nadocast": args.nadocast_webhook_url,
        "spc-tornado": args.spc_tornado_webhook_url,
    }.get(product)
    if explicit:
        return explicit
    env_name = _WEBHOOK_ENV[product]
    value = os.getenv(env_name)
    if not value:
        raise ValueError(f"Missing webhook URL for {product}. Set {env_name}.")
    return value


def post_product(status: ProductStatus, args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{status.product}_{_safe_filename(status.product_id)}.png"
    command = [
        sys.executable,
        "post_ifttt.py",
        "--product",
        status.product,
        "--output",
        str(output),
        "--upload",
        "cloudinary",
        "--webhook-url",
        webhook_url_for(status.product, args),
    ]
    if args.cloudinary_cloud_name:
        command.extend(["--cloudinary-cloud-name", args.cloudinary_cloud_name])
    if args.cloudinary_upload_preset:
        command.extend(["--cloudinary-upload-preset", args.cloudinary_upload_preset])
    if args.cloudinary_folder:
        command.extend(["--cloudinary-folder", args.cloudinary_folder])
    subprocess.run(command, check=True)


async def run(args: argparse.Namespace) -> None:
    state_path = Path(args.state_file)
    state = load_state(state_path)
    changed = False

    for product in args.products:
        status = await latest_product_status(product)
        previous_id = state.get(product)
        if previous_id == status.product_id and not args.force:
            print(f"{product}: already posted {status.product_id}")
            continue

        if args.dry_run:
            reason = "forced" if args.force else "new"
            print(f"{product}: would post {status.product_id} ({reason})")
            continue

        print(f"{product}: posting {status.product_id}")
        post_product(status, args)
        state[product] = status.product_id
        changed = True

    if changed:
        save_state(state_path, state)
        print(f"Updated state: {state_path.resolve()}")


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:120]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post new forecast products when available.")
    parser.add_argument(
        "--products",
        nargs="+",
        default=list(_DEFAULT_PRODUCTS),
        choices=_DEFAULT_PRODUCTS,
        help="Products to check.",
    )
    parser.add_argument("--state-file", default=".tornado_caster_post_state.json")
    parser.add_argument("--output-dir", default="autopost_outputs")
    parser.add_argument("--dry-run", action="store_true", help="Check only; do not post or update state.")
    parser.add_argument("--force", action="store_true", help="Post even if the latest product was already posted.")
    parser.add_argument("--nadocast-webhook-url", help="NADOCast IFTTT webhook URL.")
    parser.add_argument("--spc-tornado-webhook-url", help="SPC tornado IFTTT webhook URL.")
    parser.add_argument("--cloudinary-cloud-name", help="Cloudinary cloud name.")
    parser.add_argument("--cloudinary-upload-preset", help="Cloudinary unsigned upload preset.")
    parser.add_argument("--cloudinary-folder", help="Cloudinary folder.")
    return parser


def main() -> None:
    asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
