from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT_DIR / "experiments"

METHODS = [
    ("Restormer", "restormer"),
    ("PromptIR", "promptir"),
    ("DA-CLIP", "daclip"),
    ("DiffUIR", "diffuir"),
    ("AdaIR", "adair"),
    ("MoCE-IR", "moceir"),
    ("FoundIR", "foundir"),
    ("MambaIRv2", "mambairv2"),
    ("HINT", "hint"),
]


def main():
    print("| Method | Row file | Status |")
    print("|---|---|---|")
    for method, slug in METHODS:
        path = EXPERIMENTS_DIR / f"table1_{slug}_row.csv"
        status = "done" if path.exists() else "missing"
        print(f"| {method} | {path} | {status} |")

    print("\nRefresh final table:")
    print(r'python "E:\restormer+volterra\VTN\scripts\build_table1_fair_all_in_one.py"')


if __name__ == "__main__":
    main()
