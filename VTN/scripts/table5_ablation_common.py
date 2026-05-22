from pathlib import Path

from models.restormer_volterra import RestormerVolterra


VARIANTS = {
    "backbone": {
        "label": "Backbone",
        "save_name": "table5_backbone",
        "model_kwargs": dict(use_volterra_mdta=False, use_volterra_gdfn=False, volterra_rank=4, volterra_order=2),
    },
    "mdta_only": {
        "label": "+Volterra in MDTA only",
        "save_name": "table5_mdta_only",
        "model_kwargs": dict(use_volterra_mdta=True, use_volterra_gdfn=False, volterra_rank=4, volterra_order=2),
    },
    "gdfn_only": {
        "label": "+Volterra in GDFN only",
        "save_name": "table5_gdfn_only",
        "model_kwargs": dict(use_volterra_mdta=False, use_volterra_gdfn=True, volterra_rank=4, volterra_order=2),
    },
    "full_vtn": {
        "label": "Full VTN",
        "save_name": "table5_full_vtn",
        "model_kwargs": dict(use_volterra_mdta=True, use_volterra_gdfn=True, volterra_rank=4, volterra_order=2),
    },
    "rank1": {
        "label": "Rank R=1",
        "save_name": "table5_rank1",
        "model_kwargs": dict(use_volterra_mdta=True, use_volterra_gdfn=True, volterra_rank=1, volterra_order=2),
    },
    "rank2": {
        "label": "Rank R=2",
        "save_name": "table5_rank2",
        "model_kwargs": dict(use_volterra_mdta=True, use_volterra_gdfn=True, volterra_rank=2, volterra_order=2),
    },
    "rank4": {
        "label": "Rank R=4",
        "save_name": "table5_rank4",
        "model_kwargs": dict(use_volterra_mdta=True, use_volterra_gdfn=True, volterra_rank=4, volterra_order=2),
    },
    "order1": {
        "label": "Order 1 only",
        "save_name": "table5_order1",
        "model_kwargs": dict(use_volterra_mdta=True, use_volterra_gdfn=True, volterra_rank=4, volterra_order=1),
    },
    "order2": {
        "label": "Order 2",
        "save_name": "table5_order2",
        "model_kwargs": dict(use_volterra_mdta=True, use_volterra_gdfn=True, volterra_rank=4, volterra_order=2),
    },
}


TABLE_ORDER = [
    "backbone",
    "mdta_only",
    "gdfn_only",
    "full_vtn",
    "rank1",
    "rank2",
    "rank4",
    "order1",
    "order2",
]


def build_variant_model(variant: str):
    if variant not in VARIANTS:
        raise KeyError(f"Unknown variant: {variant}")
    return RestormerVolterra(**VARIANTS[variant]["model_kwargs"])


def latest_checkpoint(checkpoint_root: Path, variant: str):
    save_name = VARIANTS[variant]["save_name"]
    run_dir = checkpoint_root / save_name
    if not run_dir.exists():
        return None
    files = sorted(run_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None
