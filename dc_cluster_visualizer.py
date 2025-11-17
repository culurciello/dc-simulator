#!/usr/bin/env python3
"""
Simple rack diagram renderer used by dc_cluster_optimizer.py.

Given a layout payload (number of racks, servers per rack, GPUs per server,
active GPUs per rack, and summary text), draw a row of racks with server slots
colored to show utilisation, then save a PNG.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont

RACK_WIDTH = 200
TOP_PADDING = 15
BOTTOM_PADDING = 5
HEADER_SPACE = 80
FOOTER_HEIGHT = 320
MARGIN = 40
GPU_SLOT_SPACING = 0
CONTROL_SLOT_SPACING = 2
STORAGE_SLOT_SPACING = 2

COLORS = {
    "rack_border": (40, 40, 40),
    "active_gpu": (34, 139, 34),
    "inactive_gpu": (180, 180, 180),
    "storage": (244, 162, 97),
    "control": (74, 144, 226),
    "text": (30, 30, 30),
}

TEMPLATE_DIR = Path(__file__).parent / "images" / "dc-templates"


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _fit_icon(icon: Image.Image, max_width: float, max_height: float) -> Image.Image:
    width, height = icon.size
    if width <= 0 or height <= 0:
        return icon
    scale = max_width / width
    if height * scale > max_height:
        scale = max_height / height
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return icon.resize(new_size, Image.Resampling.LANCZOS)


def render_cluster_diagram(
    layout: Dict[str, any],
    output_path: Path,
    racks_per_row: Optional[int] = None,
) -> None:
    racks = layout["racks_needed"]
    servers_per_rack = layout["servers_per_rack"]
    gpus_per_server = layout["gpus_per_server"]
    active_gpus_per_rack: List[int] = layout["active_gpus_per_rack"]
    summary_lines = layout.get("summary_lines", [])
    kv_lines = layout.get("kv_lines", [])

    control_servers = max(0, layout.get("control_servers", 0))
    storage_servers = max(0, layout.get("storage_servers", 0))
    gpu_servers = max(1, layout.get("gpu_servers", servers_per_rack))
    per_rack_gpu_servers: Optional[List[int]] = layout.get("gpu_servers_per_rack")
    if per_rack_gpu_servers:
        # Clamp provided counts to sane bounds so rendering cannot explode.
        per_rack_gpu_servers = [
            max(0, min(gpu_servers, value)) for value in per_rack_gpu_servers
        ]
    per_rack_storage_servers: Optional[List[int]] = layout.get(
        "storage_servers_per_rack"
    )
    if per_rack_storage_servers:
        per_rack_storage_servers = [
            max(0, min(storage_servers, value)) for value in per_rack_storage_servers
        ]

    if racks_per_row is None or racks_per_row <= 0:
        racks_per_row = max(1, min(8, racks))

    slot_inner_width = RACK_WIDTH - 10
    gpu_icon = Image.open(TEMPLATE_DIR / "server-gpu.png").convert("RGBA")
    gpu_scale = slot_inner_width / gpu_icon.width if gpu_icon.width else 1.0
    gpu_icon_height = max(1, int(gpu_icon.height * gpu_scale))
    gpu_icon = gpu_icon.resize(
        (int(slot_inner_width), gpu_icon_height), Image.Resampling.LANCZOS
    )
    slot_height = gpu_icon_height + GPU_SLOT_SPACING

    control_icon = Image.open(TEMPLATE_DIR / "server-control.png").convert("RGBA")
    control_scale = slot_inner_width / control_icon.width if control_icon.width else 1.0
    control_icon_height = max(1, int(control_icon.height * control_scale))
    control_icon = control_icon.resize(
        (int(slot_inner_width), control_icon_height), Image.Resampling.LANCZOS
    )
    control_slot_height = control_icon_height + CONTROL_SLOT_SPACING

    storage_icon = Image.open(TEMPLATE_DIR / "server-storage.png").convert("RGBA")
    storage_scale = slot_inner_width / storage_icon.width if storage_icon.width else 1.0
    storage_icon_height = max(1, int(storage_icon.height * storage_scale))
    storage_icon = storage_icon.resize(
        (int(slot_inner_width), storage_icon_height), Image.Resampling.LANCZOS
    )
    storage_slot_height = storage_icon_height + STORAGE_SLOT_SPACING

    # Use the largest rack so that all racks share the same body height.
    servers_to_draw = servers_per_rack
    if per_rack_gpu_servers:
        servers_to_draw = max(servers_to_draw, max(per_rack_gpu_servers, default=0))
    storage_to_draw = storage_servers
    if per_rack_storage_servers:
        storage_to_draw = max(
            storage_to_draw, max(per_rack_storage_servers, default=0)
        )

    rows = math.ceil(racks / max(1, racks_per_row))
    rack_body_height = (
        TOP_PADDING
        + control_servers * control_slot_height
        + servers_to_draw * slot_height
        + storage_to_draw * storage_slot_height
        + BOTTOM_PADDING
    )
    rack_height = HEADER_SPACE + rack_body_height

    usable_row_width = min(racks_per_row, racks) * RACK_WIDTH
    canvas_width = usable_row_width + 2 * MARGIN
    canvas_height = rows * rack_height + FOOTER_HEIGHT

    img = Image.new("RGB", (canvas_width, canvas_height), color="white")
    draw = ImageDraw.Draw(img)
    font_label = _font(18)
    font_server = _font(14)

    for idx in range(racks):
        row = idx // racks_per_row
        col = idx % racks_per_row
        racks_in_this_row = min(racks_per_row, racks - row * racks_per_row)
        row_width = racks_in_this_row * RACK_WIDTH
        x_start = MARGIN + (canvas_width - 2 * MARGIN - row_width) / 2
        x = x_start + col * RACK_WIDTH
        y = row * rack_height + MARGIN
        body_top = y + HEADER_SPACE

        draw.rectangle(
            [x, body_top, x + RACK_WIDTH, body_top + rack_body_height],
            outline=COLORS["rack_border"],
            width=5,
        )
        active_servers = min(
            gpu_servers,
            math.ceil(active_gpus_per_rack[idx] / max(1, gpus_per_server)),
        )
        rack_capacity = (
            per_rack_gpu_servers[idx]
            if per_rack_gpu_servers and idx < len(per_rack_gpu_servers)
            else gpu_servers
        )
        total_slots = max(rack_capacity, active_servers)
        header_text = (
            f"Rack {idx + 1}\nGPU servers {active_servers}/{rack_capacity}\n"
            f"({active_gpus_per_rack[idx]} GPUs)"
        )
        draw.multiline_text(
            (x + RACK_WIDTH / 2, y + HEADER_SPACE / 2),
            header_text,
            fill=COLORS["text"],
            anchor="ms",
            align="center",
            font=font_label,
            spacing=2,
        )
        cursor_y = body_top + TOP_PADDING
        for _ in range(control_servers):
            icon_left = x + (RACK_WIDTH - slot_inner_width) / 2
            icon_top = cursor_y
            rect = [
                icon_left,
                icon_top,
                icon_left + slot_inner_width,
                icon_top + control_icon_height,
            ]
            img.paste(control_icon, (int(icon_left), int(icon_top)), control_icon)
            draw.rectangle(rect, outline=COLORS["control"], width=2)
            draw.text(
                (icon_left + slot_inner_width / 2, icon_top + control_icon_height / 2),
                "Control",
                fill=COLORS["control"],
                anchor="mm",
                font=font_server,
            )
            cursor_y += control_slot_height

        gpu_server_idx = 0
        gpu_start_y = cursor_y
        for slot in range(total_slots):
            slot_y = gpu_start_y + slot * slot_height
            active = gpu_server_idx < active_servers
            color = COLORS["active_gpu"] if active else COLORS["inactive_gpu"]
            gpu_server_idx += 1
            icon_left = x + (RACK_WIDTH - slot_inner_width) / 2
            icon_top = slot_y
            rect = [
                icon_left,
                icon_top,
                icon_left + slot_inner_width,
                icon_top + gpu_icon_height,
            ]
            img.paste(gpu_icon, (int(icon_left), int(icon_top)), gpu_icon)
            draw.rectangle(rect, outline=color, width=2)
            draw.text(
                (icon_left + slot_inner_width / 2, icon_top + gpu_icon_height / 2),
                f"{gpus_per_server}x GPU",
                fill=color,
                anchor="mm",
                font=font_server,
            )

        cursor_y = gpu_start_y + total_slots * slot_height
        storage_slots = (
            per_rack_storage_servers[idx]
            if per_rack_storage_servers and idx < len(per_rack_storage_servers)
            else storage_servers
        )
        for _ in range(storage_slots):
            icon_left = x + (RACK_WIDTH - slot_inner_width) / 2
            icon_top = cursor_y
            rect = [
                icon_left,
                icon_top,
                icon_left + slot_inner_width,
                icon_top + storage_icon_height,
            ]
            img.paste(storage_icon, (int(icon_left), int(icon_top)), storage_icon)
            draw.rectangle(rect, outline=COLORS["storage"], width=2)
            draw.text(
                (icon_left + slot_inner_width / 2, icon_top + storage_icon_height / 2),
                "Storage",
                fill=COLORS["storage"],
                anchor="mm",
                font=font_server,
            )
            cursor_y += storage_slot_height
        cursor_y += BOTTOM_PADDING

    footer_y = rows * rack_height + MARGIN + 10
    for line in summary_lines + kv_lines:
        draw.text(
            (MARGIN, footer_y),
            line,
            fill=COLORS["text"],
            font=font_label,
        )
        footer_y += 24

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a cluster diagram from JSON.")
    parser.add_argument("--layout-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--racks-per-row", type=int, default=8)
    args = parser.parse_args()

    payload = json.loads(args.layout_json.read_text())
    render_cluster_diagram(payload, args.output, racks_per_row=args.racks_per_row)
    print(f"Diagram saved to {args.output}")


if __name__ == "__main__":
    main()
