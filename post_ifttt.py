"""Generate a forecast map and trigger an IFTTT webhook."""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from src.publishing.cloudinary_upload import upload_unsigned
from src.sources.nadocast import NadocastRequest, NadocastSource
from src.sources.spc_outlook import SpcOutlookSource


_SPC_PRODUCTS = {
    "spc-cat": "cat",
    "spc-tornado": "tornado",
    "spc-hail": "hail",
    "spc-wind": "wind",
}


def build_webhook_url(webhook_url: str | None, event: str | None, key: str | None) -> str:
    if webhook_url:
        return webhook_url
    if not event or not key:
        raise ValueError(
            "Set IFTTT_WEBHOOK_URL, or provide both IFTTT_EVENT and IFTTT_WEBHOOK_KEY."
        )
    return f"https://maker.ifttt.com/trigger/{event}/with/key/{key}"


def build_forecast_command(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, "forecast_now.py", "--output", args.output]
    if args.map_style:
        command.extend(["--map-style", args.map_style])
    if args.product in _SPC_PRODUCTS:
        command.extend(["--spc-day1", _SPC_PRODUCTS[args.product]])
    return command


def default_tweet_text(product: str) -> str:
    return asyncio.run(default_post_text(product))


async def default_post_text(product: str) -> str:
    generated = _format_chicago_datetime(datetime.now(timezone.utc))
    if product == "nadocast":
        source = NadocastSource()
        request = await source.find_latest(NadocastRequest(hazard="tornado"))
        request = await source.resolve_request(request)
        run_time = _format_run_time(request.run_date, request.cycle)
        valid_period = _nadocast_valid_period(request.run_date, request.cycle, request.window)
        return (
            "NADOCast Tornado Guidance\n"
            f"Generated: {generated}\n"
            f"Run: {run_time}\n"
            f"Valid: {valid_period}"
        )

    spc_product = _SPC_PRODUCTS[product]
    outlook = await SpcOutlookSource().fetch_day1(spc_product)
    return (
        f"SPC Day 1 {outlook.product_label} Outlook\n"
        f"Generated: {generated}\n"
        f"Valid: {_spc_valid_period(outlook.valid_iso, outlook.expire_iso)}"
    )


def trigger_ifttt(url: str, text: str, image_url: str, local_image_path: str) -> httpx.Response:
    payload = {
        "value1": text,
        "value2": image_url,
        "value3": local_image_path,
    }
    response = httpx.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    return response


def _format_chicago_datetime(value: datetime) -> str:
    local = value.astimezone(ZoneInfo("America/Chicago"))
    return f"{local.strftime('%I').lstrip('0')}:{local:%M %p %Z %b} {local.day}"


def _format_run_time(run_date, cycle: int | None) -> str:
    if run_date is None or cycle is None:
        return "unknown"
    run_time = datetime.combine(run_date, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=cycle)
    return _format_chicago_datetime(run_time)


def _nadocast_valid_period(run_date, cycle: int | None, window: str | None) -> str:
    if run_date is None or cycle is None or not window:
        return "unknown"
    match = re.fullmatch(r"f(\d{2})-(\d{2})", window)
    if not match:
        return window
    start_hour, end_hour = (int(match.group(1)), int(match.group(2)))
    run_time = datetime.combine(run_date, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=cycle)
    valid_start = run_time + timedelta(hours=start_hour)
    valid_end = run_time + timedelta(hours=end_hour)
    return f"{_format_chicago_datetime(valid_start)} - {_format_chicago_datetime(valid_end)}"


def _spc_valid_period(valid_iso: str | None, expire_iso: str | None) -> str:
    if not valid_iso:
        return "unknown"
    valid = _format_chicago_datetime(datetime.fromisoformat(valid_iso))
    if not expire_iso:
        return valid
    expire = _format_chicago_datetime(datetime.fromisoformat(expire_iso))
    return f"{valid} - {expire}"


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    subprocess.run(build_forecast_command(args), check=True)

    text = args.text or default_tweet_text(args.product)
    image_url = args.image_url or os.getenv("IFTTT_IMAGE_URL", "")
    local_path = str(output.resolve())
    if args.upload == "cloudinary":
        image_url = upload_unsigned(
            output,
            cloud_name=args.cloudinary_cloud_name,
            upload_preset=args.cloudinary_upload_preset,
            folder=args.cloudinary_folder,
        )
        print(f"Uploaded image: {image_url}")

    if args.dry_run:
        print("Dry run: forecast generated, webhook not triggered.")
        print(f"Text: {text}")
        print(f"Image URL: {image_url or '(not set)'}")
        print(f"Local image: {local_path}")
        return

    url = build_webhook_url(
        args.webhook_url or os.getenv("IFTTT_WEBHOOK_URL"),
        args.event or os.getenv("IFTTT_EVENT"),
        args.key or os.getenv("IFTTT_WEBHOOK_KEY"),
    )
    response = trigger_ifttt(url, text=text, image_url=image_url, local_image_path=local_path)
    print(f"IFTTT webhook triggered: HTTP {response.status_code}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a Tornado Caster map and trigger an IFTTT webhook."
    )
    parser.add_argument(
        "--product",
        default="nadocast",
        choices=("nadocast", "spc-cat", "spc-tornado", "spc-hail", "spc-wind"),
        help="Forecast product to render before posting.",
    )
    parser.add_argument("--output", default="forecast_post.png", help="Generated map path.")
    parser.add_argument("--map-style", default="dark", choices=("light", "dark"))
    parser.add_argument("--text", help="Text to send as IFTTT value1.")
    parser.add_argument(
        "--image-url",
        help="Public image URL to send as IFTTT value2. Required if your applet posts images.",
    )
    parser.add_argument(
        "--upload",
        default="none",
        choices=("none", "cloudinary"),
        help="Upload generated image before triggering IFTTT.",
    )
    parser.add_argument("--cloudinary-cloud-name", help="Cloudinary cloud name.")
    parser.add_argument("--cloudinary-upload-preset", help="Cloudinary unsigned upload preset.")
    parser.add_argument(
        "--cloudinary-folder",
        help="Cloudinary folder. Defaults to CLOUDINARY_FOLDER or tornado-caster.",
    )
    parser.add_argument("--webhook-url", help="Full IFTTT webhook URL.")
    parser.add_argument("--event", help="IFTTT Webhooks event name.")
    parser.add_argument("--key", help="IFTTT Webhooks key.")
    parser.add_argument("--dry-run", action="store_true", help="Generate only; do not call IFTTT.")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
