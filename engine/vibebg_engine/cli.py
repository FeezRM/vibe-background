import argparse
import sys
from pathlib import Path

from vibebg_engine.pipeline import (
    LIGHTING_MODES,
    TIERS,
    BackgroundSettings,
    LightingSettings,
    Progress,
    RenderSettings,
    render,
)
from vibebg_engine.placement import Placement


def _print_progress(p: Progress):
    eta = f"{p.eta:5.0f}s" if p.eta is not None else "    ?"
    fps = p.frame / p.elapsed if p.elapsed else 0.0
    sys.stderr.write(f"\r{p.frame}/{p.total} frames  {fps:5.1f} fps  eta {eta}")
    sys.stderr.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vibebg")
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("render", help="replace the background of a video")
    r.add_argument("input", type=Path)
    r.add_argument("-o", "--output", type=Path, required=True)
    bg = r.add_mutually_exclusive_group(required=True)
    bg.add_argument("--bg", type=Path, help="background image or looping video")
    bg.add_argument("--blur-original", action="store_true",
                    help="keep your own room, blurred, instead of replacing it")
    r.add_argument("--dof", type=float, default=0.0, help="depth-of-field blur for images, 0..1")
    r.add_argument("--blur", type=float, default=0.0,
                   help="uniform blur for video backgrounds or --blur-original, 0..1")
    r.add_argument("--depth", type=Path, help="precomputed depth map for --dof")
    r.add_argument("--tier", choices=TIERS, default="fast")
    r.add_argument("--codec", choices=["h264", "hevc"], default="h264")
    r.add_argument("--scale", type=float, default=1.0)
    r.add_argument("--x", type=float, default=0.0, help="shift right, fraction of width")
    r.add_argument("--y", type=float, default=0.0, help="shift down, fraction of height")
    r.add_argument("--light", choices=LIGHTING_MODES, default="off")
    r.add_argument("--strength", type=float, default=0.6)
    r.add_argument("--wrap", type=float, default=0.5)
    r.add_argument("--shadow", type=float, default=0.0)
    r.add_argument("--warmth", type=float, default=0.0)
    r.add_argument("--denoise", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "render":
        settings = RenderSettings(
            background=BackgroundSettings(path=args.bg, dof=args.dof, blur=args.blur,
                                          depth_path=args.depth),
            tier=args.tier,
            codec=args.codec,
            placement=Placement(scale=args.scale, x=args.x, y=args.y),
            lighting=LightingSettings(mode=args.light, strength=args.strength, wrap=args.wrap,
                                      shadow=args.shadow, warmth=args.warmth,
                                      denoise=args.denoise),
        )
        render(args.input, args.output, settings, on_progress=_print_progress)
        sys.stderr.write(f"\nwrote {args.output}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
