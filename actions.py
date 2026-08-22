"""One action list for the detector, the job watch, and the printer pause gate."""

ACTION_MAP = {
    "spaghetti_failure": {
        "label": "Spaghetti",
        "what_to_do": "Pause the print. Filament has come loose from the part.",
        "severity": "critical",
    },
    "warping": {
        "label": "Warping",
        "what_to_do": "Slow to 70% and raise the bed by 5 °C.",
        "severity": "warning",
    },
    "first_layer_failure": {
        "label": "First layer failed",
        "what_to_do": "Pause the print. The first layer did not stick.",
        "severity": "critical",
    },
    "under_extrusion": {
        "label": "Under-extrusion",
        "what_to_do": "Slow to 80% and raise the nozzle by 5 °C.",
        "severity": "warning",
    },
    "stringing": {
        "label": "Stringing",
        "what_to_do": "Drop the nozzle by 3 °C. Tune retraction later.",
        "severity": "warning",
    },
    "blob_or_zit": {
        "label": "Blobs",
        "what_to_do": "Slow to 85% and keep an eye on the nozzle.",
        "severity": "warning",
    },
    "layer_shift": {
        "label": "Layer shift",
        "what_to_do": "Pause the print. This is a mechanical slip.",
        "severity": "critical",
    },
    "nozzle_clog": {
        "label": "Nozzle clog",
        "what_to_do": "Pause the print and clear the nozzle.",
        "severity": "critical",
    },
    "good_print": {
        "label": "Looks fine",
        "what_to_do": "Let it run.",
        "severity": "ok",
    },
}
ACTION_MAP["spaghetti"] = ACTION_MAP["spaghetti_failure"]
ACTION_MAP["zits_blobs"] = ACTION_MAP["blob_or_zit"]
ACTION_MAP["blob"] = ACTION_MAP["blob_or_zit"]

RANK = {"ok": 0, "warning": 1, "critical": 2}


def pretty(name: str) -> dict:
    return ACTION_MAP.get(name, ACTION_MAP["good_print"])


def severity_rank(name: str) -> int:
    return RANK[pretty(name)["severity"]]


def is_critical(name: str) -> bool:
    return pretty(name)["severity"] == "critical"
