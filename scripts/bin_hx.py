#!/usr/bin/env python3
"""HX-Composite (DWD, ODIM-HDF5, Reflektivitaet in dBZ) -> NATIVES DWD-Raster (stereographisch, 250 m).

Kein Warp, kein WebP. Ausgabe: rohe uint8-Bytes, keine Dateiendung.
Layout: ysize Zeilen x xsize Spalten, row-major, Zeile 0 = Norden (wie im HDF5).
Dateigroesse immer exakt xsize * ysize Bytes.

Jeder Pixel ist ein CODE (0-255):
    0 .. 254   Reflektivitaet: Code = floor((dBZ - DBZ_MIN) / DBZ_STEP)  -> dBZ = DBZ_MIN + Code * DBZ_STEP
    255        kein Daten / ausserhalb -> transparent

Der Client faerbt per Schwellen-Tabelle (diskrete Stufen, "ab Code X"). Codes unter der
ersten Stufe (< 5 dBZ) sind transparent.

Farbtabelle fuer den Client (radar-colors.js) ausgeben:   python3 hx_native.py --colors
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
SRC_DIR = Path("data/hx")
OUT_DIR = Path("output/hx")

FILENAME_RE = re.compile(r"composite_hx_(\d{8})_(\d{4})-hd5")

DBZ_MIN = 0.0
DBZ_STEP = 1.0
NODATA_CODE = 255
MAX_CODE = 254

# Fallbacks, falls Attribute in der Datei fehlen
DEFAULT_NODATA = 65535
DEFAULT_SCALE_M = 250.0     # HX = 250 m (RV waere 1000 m)
DEFAULT_GAIN = 0.5          # ACHTUNG: nur Fallback - bitte gegen echte Datei pruefen
DEFAULT_OFFSET = -32.5      # ACHTUNG: nur Fallback - bitte gegen echte Datei pruefen

# Farbtabelle fuer den Client: (ab dBZ, (R, G, B)) - diskrete Stufen
COLOR_TABLE: list[tuple[float, tuple[int, int, int]]] = [
    (5,  (150, 150, 150)),
    (10, (0, 200, 255)),
    (15, (0, 120, 255)),
    (20, (0, 40, 246)),
    (25, (0, 200, 0)),
    (30, (0, 150, 0)),
    (35, (255, 255, 0)),
    (40, (255, 190, 0)),
    (45, (255, 120, 0)),
    (50, (255, 0, 0)),
    (55, (190, 0, 0)),
    (60, (255, 120, 255)),
    (65, (170, 0, 170)),
    (70, (255, 255, 255)),
]

BERLIN = ZoneInfo("Europe/Berlin")


# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #
def parse_filename(filename: str) -> datetime:
    """Liefert die Zeit (UTC) aus dem HX-Dateinamen 'composite_hx_yyyymmdd_HHMM-hd5'."""
    m = FILENAME_RE.match(filename)
    if not m:
        raise ValueError(
            "Dateiname passt nicht zum Schema "
            f"'composite_hx_yyyymmdd_HHMM-hd5': {filename}"
        )
    date_str, time_str = m.groups()
    naive = datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
    return naive.replace(tzinfo=timezone.utc)


def _attr_str(v) -> str:
    return v.decode(errors="ignore") if isinstance(v, bytes) else str(v)


def _scalar(v):
    return v.item() if isinstance(v, np.ndarray) and v.size == 1 else v


def dbz_to_code(dbz: float) -> int:
    return int(round((dbz - DBZ_MIN) / DBZ_STEP))


# --------------------------------------------------------------------------- #
# HDF5 lesen
# --------------------------------------------------------------------------- #
def find_dbz_dataset(h5file: h5py.File) -> h5py.Dataset:
    """Sucht das 2D-Reflektivitaets-Dataset; Fallback: erstes 2D-Dataset."""
    candidates: list[h5py.Dataset] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and name.endswith("/data") and obj.ndim == 2:
            candidates.append(obj)

    h5file.visititems(visitor)
    if not candidates:
        raise RuntimeError("Kein 2D-Datensatz in der HX-Datei gefunden.")

    for ds in candidates:
        what = ds.parent.get("what")
        if what is not None and "quantity" in what.attrs:
            q = _attr_str(what.attrs["quantity"]).upper()
            if any(k in q for k in ("DBZ", "TH", "REFL")):
                return ds
    return candidates[0]


def read_dbz(ds: h5py.Dataset) -> np.ndarray:
    """Liest das Dataset und gibt dBZ als float32 zurueck (NaN = kein Datum, -inf-artig = undetect)."""
    what = None
    for grp in (ds.parent, ds.parent.parent, ds.file):
        w = grp.get("what")
        if w is not None and "gain" in w.attrs:
            what = w
            break
    attrs = what.attrs if what is not None else {}
    if what is None:
        print("WARNUNG: kein 'what'-Block mit gain gefunden, nutze Fallback-Werte!")

    gain = float(_scalar(attrs.get("gain", DEFAULT_GAIN)))
    offset = float(_scalar(attrs.get("offset", DEFAULT_OFFSET)))
    nodata = _scalar(attrs.get("nodata", DEFAULT_NODATA))
    undetect = _scalar(attrs["undetect"]) if "undetect" in attrs else None
    quantity = _attr_str(attrs.get("quantity", "")).upper()

    raw = ds[()]
    dbz = raw.astype(np.float32) * np.float32(gain) + np.float32(offset)

    if undetect is not None:
        dbz[raw == undetect] = np.nan
    dbz[raw == nodata] = np.nan

    print(
        f"HX-Dataset: quantity='{quantity}', dtype={raw.dtype}, gain={gain}, offset={offset}, "
        f"nodata={nodata}, undetect={undetect}"
    )
    valid = dbz[np.isfinite(dbz)]
    if valid.size:
        print(f"Wertebereich: {valid.min():.1f} .. {valid.max():.1f} dBZ")
    return dbz


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

    def as_float(key: str, default: float | None = None) -> float:
        if key not in where.attrs and default is not None:
            return default
        return float(_scalar(where.attrs[key]))

    return {
        "projdef": as_str("projdef"),
        "xsize": int(as_float("xsize")),
        "ysize": int(as_float("ysize")),
        "xscale": as_float("xscale", DEFAULT_SCALE_M),
        "yscale": as_float("yscale", DEFAULT_SCALE_M),
        "ll_lon": as_float("LL_lon"),
        "ll_lat": as_float("LL_lat"),
    }


# --------------------------------------------------------------------------- #
# dBZ -> Code
# --------------------------------------------------------------------------- #
def encode_codes(dbz: np.ndarray) -> np.ndarray:
    """dBZ -> uint8-Codes. Werte < DBZ_MIN und NaN -> NODATA_CODE (255)."""
    codes = np.full(dbz.shape, NODATA_CODE, dtype=np.uint8)

    valid = np.isfinite(dbz) & (dbz >= DBZ_MIN - 1e-6)
    k = np.floor((dbz[valid] - DBZ_MIN) / DBZ_STEP + 1e-6)
    codes[valid] = np.clip(k, 0, MAX_CODE).astype(np.uint8)
    return codes


# --------------------------------------------------------------------------- #
# Farbtabelle fuer den Client (radar-colors.js)
# --------------------------------------------------------------------------- #
def print_client_colors() -> None:
    print("  reflektivitaet: {")
    print("    exact: false,           // Stufe gilt ab ihrem Code bis zur naechsten Stufe")
    print(f"    nodata: [{NODATA_CODE}],")
    print(f"    transparentBelow: {dbz_to_code(COLOR_TABLE[0][0])},   // < {COLOR_TABLE[0][0]} dBZ")
    print("    steps: [")
    for dbz, (r, g, b) in COLOR_TABLE:
        print(f"      [{dbz_to_code(dbz)}, '#{r:02x}{g:02x}{b:02x}'],   // ab {dbz} dBZ")
    print("    ],")
    print("  },")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    if "--colors" in sys.argv:
        print_client_colors()
        return

    candidates = sorted(
        p for p in SRC_DIR.glob("composite_hx_*-hd5")
        if FILENAME_RE.match(p.name)
    )
    if not candidates:
        sys.exit(f"Keine HX-Datei (composite_hx_*-hd5) in {SRC_DIR} gefunden.")
    src_path = candidates[-1]
    ts = parse_filename(src_path.name)
    print(f"Quelle: {src_path.name}  ({ts:%Y-%m-%d %H:%M} UTC)")

    with h5py.File(src_path, "r") as f:
        dbz = read_dbz(find_dbz_dataset(f))
        where = find_where_group(f)
        if where is None:
            sys.exit("Keine 'where'-Projektionsinfo in der HD5-Datei gefunden.")
        grid = extract_grid_info(where)

    h, w = dbz.shape
    if (w, h) != (grid["xsize"], grid["ysize"]):
        sys.exit(f"Rastergroesse passt nicht: Array {w}x{h}, where {grid['xsize']}x{grid['ysize']}")
    print(f"Natives Raster: {w} x {h} px, {grid['xscale']} m / px")

    codes = encode_codes(dbz)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"hx_{ts.astimezone(BERLIN):%Y%m%d_%H%M}"
    np.ascontiguousarray(codes).tofile(out_path)
    print(f"Gespeichert: {out_path} ({w}x{h}, {codes.nbytes} Bytes)")

    to_proj = Transformer.from_crs("EPSG:4326", grid["projdef"], always_xy=True)
    ll_x, ll_y = to_proj.transform(grid["ll_lon"], grid["ll_lat"])
    meta = {
        "format": "raw-uint8",
        "product": "hx-reflectivity",
        "unit": "dBZ",
        "dbz_min": DBZ_MIN,
        "dbz_step": DBZ_STEP,
        "nodata_code": NODATA_CODE,
        "max_code": MAX_CODE,
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