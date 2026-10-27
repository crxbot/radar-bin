#!/usr/bin/env python3
"""RV-Composite (DWD, ODIM-HDF5) -> NATIVES DWD-Raster (1100 x 1200, stereographisch).

Kein Warp nach Web Mercator, kein WebP. Ausgabe: rohe uint8-Bytes, keine Dateiendung.
Layout: ysize Zeilen x xsize Spalten, row-major, Zeile 0 = Norden (wie im HDF5).
Dateigroesse immer exakt xsize * ysize = 1.320.000 Bytes.

Jeder Pixel ist ein CODE (0-255), der Client faerbt ihn per Tabelle ein:
    0        kein Regen (unter MIN_VISIBLE_MMH)  -> transparent
    1 .. 9   schwacher Regen 0.01 .. 0.09 mm/h (Code = floor(mm/h * 100)), weiss mit Alpha-Rampe
    10 ..    Stufen der COLOR_TABLE: Code = 10 + Nummer der Stufe (0 = 0.1 mm/h, 1 = 0.12 mm/h, ...)
    255      kein Daten / ausserhalb  -> transparent

Farbtabelle fuer den Client (radar-colors.js) ausgeben:   python3 rv_native.py --colors
"""
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np
from pyproj import Transformer

# --------------------------------------------------------------------------- #
# Konfiguration
# --------------------------------------------------------------------------- #
SRC_DIR = Path("data/rv")
OUT_DIR = Path("output/rv")

FILENAME_RE = re.compile(r"composite_rv_(\d{8})_(\d{4})_(\d{3})-hd5")

# Falls das Dataset als Akkumulation vorliegt: Laenge des Intervalls in Minuten (RV = 5 min)
RV_ACCUM_MINUTES = 5
# Fallbacks, falls die Attribute (gain/nodata) in der Datei fehlen
RV_DEFAULT_GAIN = 0.01
RV_DEFAULT_NODATA = 65535

MIN_VISIBLE_MMH = 0.01      # darunter: Code 0 (transparent)
WHITE_MAX_MMH = 0.09        # ab hier volle Deckkraft der Rampe
WHITE_ALPHA_MIN = 20        # Alpha bei 0.01 mm/h (0-255)
WHITE_ALPHA_MAX = 160       # Alpha bei 0.09 mm/h (0-255)
FAINT_RGB = (255, 255, 255) # Farbe der schwachen Stufe (Codes 1-9)

NODATA_CODE = 255
FAINT_CODE_MAX = 9          # Codes 1..9 = 0.01 .. 0.09 mm/h
STEP_CODE_OFFSET = 10       # Code der ersten COLOR_TABLE-Stufe

# Farbtabelle: (mm/h, (R, G, B))
# Diskrete Stufen: jeder Wert bekommt die Farbe der letzten Schwelle <= Wert.
COLOR_TABLE: list[tuple[float, tuple[int, int, int]]] = [
    (0.1,   (0, 221, 238)),
    (0.12,  (0, 206, 240)),
    (0.14,  (1, 190, 242)),
    (0.16,  (1, 175, 244)),
    (0.18,  (1, 160, 246)),
    (0.2,   (1, 160, 246)),
    (0.24,  (1, 120, 246)),
    (0.28,  (1, 80, 246)),
    (0.32,  (0, 40, 246)),
    (0.36,  (0, 0, 246)),

    # Grün-Phase: leichter Regen 0.4-4.5 mm/h
    (0.4,   (0, 255, 0)),
    (0.52,  (0, 246, 0)),
    (0.64,  (0, 237, 0)),
    (0.76,  (0, 228, 0)),
    (0.88,  (0, 218, 0)),
    (1.0,   (0, 209, 0)),
    (1.26,  (0, 200, 0)),
    (1.52,  (0, 200, 0)),
    (1.78,  (0, 192, 0)),
    (2.04,  (0, 184, 0)),
    (2.3,   (0, 176, 0)),
    (2.86,  (0, 168, 0)),
    (3.42,  (0, 160, 0)),
    (3.98,  (0, 152, 0)),
    (4.56,  (0, 144, 0)),

    # Gelb-Phase:
    (5.1,   (255, 255, 0)),
    (6.48,  (249, 239, 0)),
    (7.86,  (243, 224, 0)),
    (9.24,  (237, 208, 0)),
    (10.62, (231, 192, 0)),
    (12.0,  (231, 192, 0)),
    (14.4,  (237, 180, 0)),
    (16.8,  (243, 168, 0)),
    (19.2,  (249, 156, 0)),
    (21.6,  (255, 144, 0)),

    # Orange-Phase:
    (24.0,  (255, 144, 0)),
    (28.0,  (255, 108, 0)),
    (32.0,  (255, 72, 0)),

    # Rot-Phase:
    (36.0,  (255, 36, 0)),
    (40.0,  (255, 0, 0)),
    (44.0,  (255, 0, 0)),
    (51.2,  (245, 0, 0)),
    (58.4,  (235, 0, 0)),
    (65.6,  (224, 0, 0)),
    (72.8,  (214, 0, 0)),
    (80.0,  (214, 0, 0)),
    (93.2,  (201, 0, 0)),
    (106.4, (187, 0, 0)),
    (119.6, (174, 0, 0)),
    (132.8, (160, 0, 0)),

    # Violett-Phase:
    (146.0, (255, 200, 255)),
    (170.4, (244, 180, 244)),
    (194.8, (232, 160, 232)),
    (219.2, (221, 140, 221)),
    (243.6, (209, 120, 209)),
    (268.0, (198, 100, 198)),
    (312.6, (186, 80, 186)),
    (357.2, (175, 60, 175)),
    (401.8, (163, 40, 163)),
    (446.4, (152, 20, 152)),
    (491.0, (140, 0, 140)),
]
assert STEP_CODE_OFFSET + len(COLOR_TABLE) <= NODATA_CODE, "Zu viele Stufen fuer uint8"

BERLIN = ZoneInfo("Europe/Berlin")


# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #
def parse_filename(filename: str) -> tuple[datetime, int]:
    """Liefert (Basiszeit UTC, Vorhersageschritt in Minuten) aus dem RV-Dateinamen."""
    m = FILENAME_RE.match(filename)
    if not m:
        raise ValueError(
            "Dateiname passt nicht zum Schema "
            f"'composite_rv_yyyymmdd_HHMM_LLL-hd5': {filename}"
        )
    date_str, time_str, lead_str = m.groups()
    naive = datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
    return naive.replace(tzinfo=timezone.utc), int(lead_str)


def _attr_str(v) -> str:
    return v.decode(errors="ignore") if isinstance(v, bytes) else str(v)


def _scalar(v):
    return v.item() if isinstance(v, np.ndarray) and v.size == 1 else v


# --------------------------------------------------------------------------- #
# HDF5 lesen
# --------------------------------------------------------------------------- #
def find_rate_dataset(h5file: h5py.File) -> h5py.Dataset:
    """Sucht das 2D-Niederschlags-Dataset; Fallback: erstes 2D-Dataset."""
    candidates: list[h5py.Dataset] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and name.endswith("/data") and obj.ndim == 2:
            candidates.append(obj)

    h5file.visititems(visitor)
    if not candidates:
        raise RuntimeError("Kein 2D-Datensatz in der RV-Datei gefunden.")

    for ds in candidates:
        what = ds.parent.get("what")
        if what is not None and "quantity" in what.attrs:
            q = _attr_str(what.attrs["quantity"]).upper()
            if any(k in q for k in ("ACRR", "RATE", "PRECIP", "RR")):
                return ds
    return candidates[0]


def read_rate_mmh(ds: h5py.Dataset) -> np.ndarray:
    """Liest das Dataset und gibt mm/h als float64 zurueck (NaN = kein Datum)."""
    what = None
    for grp in (ds.parent, ds.parent.parent, ds.file):
        w = grp.get("what")
        if w is not None and "gain" in w.attrs:
            what = w
            break
    attrs = what.attrs if what is not None else {}

    gain = float(_scalar(attrs.get("gain", RV_DEFAULT_GAIN)))
    offset = float(_scalar(attrs.get("offset", 0.0)))
    nodata = _scalar(attrs.get("nodata", RV_DEFAULT_NODATA))
    undetect = _scalar(attrs["undetect"]) if "undetect" in attrs else None
    quantity = _attr_str(attrs.get("quantity", "")).upper()

    raw = ds[()]
    # Pixelwert * gain = 5-Minuten-Niederschlagsmenge M[mm] (siehe DWD-Formatbeschreibung)
    values = raw.astype(np.float64) * gain + offset

    # RV ist immer ein 5-min-Produkt -> immer auf mm/h hochrechnen,
    # unabhaengig davon, wie 'quantity' in der Datei benannt ist
    factor = 60.0 / RV_ACCUM_MINUTES
    values *= factor

    values[raw == nodata] = np.nan
    if undetect is not None and undetect != nodata:
        values[raw == undetect] = 0.0

    print(
        f"RV-Dataset: quantity='{quantity}', gain={gain}, offset={offset}, "
        f"nodata={nodata}, undetect={undetect}, Faktor->mm/h={factor:g}"
    )
    valid = values[np.isfinite(values)]
    if valid.size:
        print(f"Wertebereich: {valid.min():.2f} .. {valid.max():.2f} mm/h")
    return values


def find_where_group(h5file: h5py.File) -> h5py.Group | None:
    required = ("projdef", "xsize", "ysize", "xscale", "yscale", "LL_lon", "LL_lat")

    def complete(grp) -> bool:
        return all(k in grp.attrs for k in required)

    root_where = h5file.get("where")
    if root_where is not None and complete(root_where):
        return root_where

    found: list[h5py.Group] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Group) and name.split("/")[-1] == "where" and complete(obj):
            found.append(obj)

    h5file.visititems(visitor)
    return found[0] if found else None


def extract_grid_info(where: h5py.Group) -> dict:
    def as_str(key: str) -> str:
        return _attr_str(where.attrs[key])

    def as_float(key: str) -> float:
        return float(_scalar(where.attrs[key]))

    return {
        "projdef": as_str("projdef"),
        "xsize": int(as_float("xsize")),
        "ysize": int(as_float("ysize")),
        "xscale": as_float("xscale"),
        "yscale": as_float("yscale"),
        "ll_lon": as_float("LL_lon"),
        "ll_lat": as_float("LL_lat"),
    }


# --------------------------------------------------------------------------- #
# mm/h -> Code
# --------------------------------------------------------------------------- #
def encode_codes(rate: np.ndarray) -> np.ndarray:
    """mm/h (nativ) -> uint8-Codes. Gleiche Stufen-Logik wie das alte colorize()."""
    thresholds = np.array([t for t, _ in COLOR_TABLE], dtype=np.float64)

    codes = np.zeros(rate.shape, dtype=np.uint8)            # 0 = kein Regen
    finite = np.isfinite(rate)
    codes[~finite] = NODATA_CODE

    visible = finite & (rate >= MIN_VISIBLE_MMH)
    faint = visible & (rate < thresholds[0])                # 0.01 .. <0.1
    strong = visible & ~faint

    # schwach: 0.01 -> 1, 0.02 -> 2, ... 0.09 -> 9
    k = np.floor(rate[faint] * 100 + 1e-6)
    codes[faint] = np.clip(k, 1, FAINT_CODE_MAX).astype(np.uint8)

    # ab der ersten Schwelle: Nummer der Stufe + Offset
    idx = np.searchsorted(thresholds - 1e-6, rate[strong], side="right") - 1
    idx = np.clip(idx, 0, len(thresholds) - 1)
    codes[strong] = (STEP_CODE_OFFSET + idx).astype(np.uint8)
    return codes


# --------------------------------------------------------------------------- #
# Farbtabelle fuer den Client (radar-colors.js)
# --------------------------------------------------------------------------- #
def print_client_colors() -> None:
    fr, fg, fb = FAINT_RGB
    print("  regenradar: {")
    print("    exact: true,            // jeder Code gilt nur fuer sich")
    print(f"    nodata: [0, {NODATA_CODE}],")
    print("    steps: [")
    print("      // schwacher Regen 0.01 - 0.09 mm/h (Alpha-Rampe)")
    for code in range(1, FAINT_CODE_MAX + 1):
        t = (code * 0.01 - MIN_VISIBLE_MMH) / (WHITE_MAX_MMH - MIN_VISIBLE_MMH)
        t = min(max(t, 0.0), 1.0)
        alpha = int(WHITE_ALPHA_MIN + t * (WHITE_ALPHA_MAX - WHITE_ALPHA_MIN))
        print(f"      [{code}, '#{fr:02x}{fg:02x}{fb:02x}{alpha:02x}'],   // {code * 0.01:.2f} mm/h")
    print("      // ab 0.1 mm/h")
    for i, (mmh, (r, g, b)) in enumerate(COLOR_TABLE):
        print(f"      [{STEP_CODE_OFFSET + i}, '#{r:02x}{g:02x}{b:02x}'],   // ab {mmh} mm/h")
    print("    ],")
    print("  },")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    if "--colors" in sys.argv:
        print_client_colors()
        return

    # Nur Analysedateien (_000), neueste zuletzt
    candidates = sorted(
        p for p in SRC_DIR.glob("composite_rv_*-hd5")
        if FILENAME_RE.match(p.name) and parse_filename(p.name)[1] == 0
    )
    if not candidates:
        sys.exit(f"Keine RV-Datei (composite_rv_*_000-hd5) in {SRC_DIR} gefunden.")
    src_path = candidates[-1]
    base_ts, lead_min = parse_filename(src_path.name)
    ts = base_ts + timedelta(minutes=lead_min)
    print(f"Quelle: {src_path.name}  ({ts:%Y-%m-%d %H:%M} UTC)")

    with h5py.File(src_path, "r") as f:
        rate = read_rate_mmh(find_rate_dataset(f))
        where = find_where_group(f)
        if where is None:
            sys.exit("Keine 'where'-Projektionsinfo in der HD5-Datei gefunden.")
        grid = extract_grid_info(where)

    h, w = rate.shape
    if (w, h) != (grid["xsize"], grid["ysize"]):
        sys.exit(f"Rastergroesse passt nicht: Array {w}x{h}, where {grid['xsize']}x{grid['ysize']}")
    print(f"Natives Raster: {w} x {h} px, {grid['xscale']} m / px")

    codes = encode_codes(rate)

    # Speichern: rohe Bytes, keine Dateiendung, Zeile 0 = Norden
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"regenradar_{ts.astimezone(BERLIN):%Y%m%d_%H%M}"
    np.ascontiguousarray(codes).tofile(out_path)
    print(f"Gespeichert: {out_path} ({w}x{h}, {codes.nbytes} Bytes)")

    # Metadaten fuer den Client (einmalig kopieren, aendern sich nicht)
    to_proj = Transformer.from_crs("EPSG:4326", grid["projdef"], always_xy=True)
    ll_x, ll_y = to_proj.transform(grid["ll_lon"], grid["ll_lat"])
    meta = {
        "format": "raw-uint8",
        "width": w,
        "height": h,
        "row_order": "north-to-south",
        "projdef": grid["projdef"],
        "pixel_size_m": grid["xscale"],
        "ll_x": ll_x,
        "ll_y": ll_y,
        "extent_native": [ll_x, ll_y, ll_x + w * grid["xscale"], ll_y + h * grid["yscale"]],
    }
    print("Client-Metadaten:", json.dumps(meta))


if __name__ == "__main__":
    main()