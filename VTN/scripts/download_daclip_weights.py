import argparse
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = ROOT_DIR.parent
DEFAULT_OUT = WORKSPACE_DIR / "baselines" / "DA-CLIP" / "pretrained" / "daclip_ViT-B-32.pt"
DEFAULT_URL = "https://huggingface.co/weblzw/daclip-uir-ViT-B-32-irsde/resolve/main/daclip_ViT-B-32.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Download DA-CLIP pretrained controller weights.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and out.stat().st_size > 0:
        print(f"[Exists] {out}")
        return

    print(f"[Download] {args.url}")
    print(f"[To      ] {out}")
    urllib.request.urlretrieve(args.url, out)
    print(f"[Saved   ] {out} ({out.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
