import argparse
import csv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT_DIR / "experiments"


VTN_ROW = {
    "Method": "VTN",
    "Status": "Ours",
    "Direct Comparison?": "Yes",
    "Training Data": "Same all-in-one mixture",
    "Rain(avg)": "31.91 / 0.9103",
    "GoPro": "29.80 / 0.9058",
    "RESIDE-6K": "22.28 / 0.8608",
    "CSD": "28.64 / 0.9141",
    "Avg": "28.16 / 0.8978",
}

METHODS = [
    ("Restormer", "restormer", "Retrained", "Yes", "Same as VTN"),
    ("PromptIR", "promptir", "Retrained", "Yes", "Same as VTN"),
    "DA-CLIP",
    "DiffUIR",
    "AdaIR",
    "MoCE-IR",
    "FoundIR",
    "MambaIRv2",
    "HINT",
]

COLUMNS = [
    "Method",
    "Status",
    "Direct Comparison?",
    "Training Data",
    "Rain(avg)",
    "GoPro",
    "RESIDE-6K",
    "CSD",
    "Avg",
]


METHOD_SLUGS = {
    "DA-CLIP": "daclip",
    "DiffU  ```IR": "diffuir",
    "AdaIR": "adair",
    "MoCE-IR": "moceir",
    "FoundIR": "foundir",
    "MambaIRv2": "mambairv2",
    "HINT": "hint",
}


def read_method_row(path: Path, method: str, default_status: str,
                    default_direct: str, default_training_data: str):
    if not path.exists():
        return {
            "Method": method,
            "Status": default_status,
            "Direct Comparison?": default_direct,
            "Training Data": default_training_data,
            "Rain(avg)": "running" if default_direct == "Yes" else "-",
            "GoPro": "running" if default_direct == "Yes" else "-",
            "RESIDE-6K": "running" if default_direct == "Yes" else "-",
            "CSD": "running" if default_direct == "Yes" else "-",
            "Avg": "running" if default_direct == "Yes" else "-",
        }
    with open(path, newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    return {col: row.get(col, "") for col in COLUMNS}


def parse_args():
    parser = argparse.ArgumentParser(description="Build Table 1 fair all-in-one comparison.")
    parser.add_argument(
        "--out",
        default=str(EXPERIMENTS_DIR / "table1_fair_all_in_one.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for item in METHODS:
        if isinstance(item, tuple):
            method, slug, status, direct, training_data = item
        else:
            method = item
            slug = METHOD_SLUGS[method]
            status = "Retrained"
            direct = "Yes"
            training_data = "Same as VTN"
        row_path = EXPERIMENTS_DIR / f"table1_{slug}_row.csv"
        rows.append(read_method_row(row_path, method, status, direct, training_data))
    rows.append(VTN_ROW)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    md_out = out.with_suffix(".md")
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(COLUMNS) + " |\n")
        f.write("|---|---|---|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write("| " + " | ".join(row[col] for col in COLUMNS) + " |\n")

    print(f"[Table 1 CSV] {out}")
    print(f"[Table 1 MD ] {md_out}")


if __name__ == "__main__":
    main()
