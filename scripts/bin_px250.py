#!/usr/bin/env python3
"""PX250-Standortdaten (DWD, 17 Radare, ODIM-HDF5) -> ein Komposit im NATIVEN HX-Raster (250 m).

Ueberlappung (Mischlogik):
  - nur ein Radar meldet Echo (>= 0 dBZ)  -> dieser Wert gilt (keine Zellen verlieren)
  - beide melden Echo                      -> der Radar mit der kleineren Entfernung zum Pixel gewinnt
  - "kein Echo" eines Radars ueberschreibt nie das Echo eines anderen

Pixel-Codes: 0..253 = floor(dBZ), 254 reserviert, 255 = keine Radardaten.
Layout: HEIGHT x WIDTH, row-major, Zeile 0 = Norden, keine Dateiendung.
"""
import argparse
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np
import requests
from pyproj import Transformer

BASE_URL = "https://opendata.dwd.de/weather/radar/sites/px250"
SITES = ["asb", "boo", "drs", "eis", "ess", "fbg", "fld", "hnr", "isn",
         "mem", "neu", "nhb", "oft", "pro", "ros", "tur", "umd"]

SRC_DIR = Path("data/px250")
OUT_DIR = Path("output/px250")

WAIT_SECONDS = 240
RETRY_SLEEP = 10
MIN_SITES = 10

COMP_PROJDEF = (
    "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=10 "
    "+x_0=543571.83521776402 +y_0=3622213.8619310022 +units=m "
    "+a=6378137 +b=6356752.3142451802 +no_defs"
)
WIDTH, HEIGHT = 4400, 4800
PIXEL_M = 250.0
LL_X, LL_Y = -125.0, -1199875.0
TOP_Y = LL_Y + HEIGHT * PIXEL_M

DBZ_MIN = 0.0
DBZ_STEP = 1.0
MAX_CODE = 254
NODATA_CODE = 255

BERLIN = ZoneInfo("Europe/Berlin")


# --------------------------------------------------------------------------- #
# Zeitslot
# --------------------------------------------------------------------------- #
def get_slot(cli_time):
    if cli_time:
        dt = datetime.strptime(cli_time, "%Y%m%d%H%M")
    elif os.environ.get("DATE") and os.environ.get("TIME"):
        dt = datetime.strptime(os.environ["DATE"] + os.environ["TIME"], "%Y%m%d%H%M")
    else:
        dt = datetime.now(timezone.utc).replace(tzinfo=None)
    dt = dt.replace(minute=dt.minute - dt.minute % 5, second=0, microsecond=0)
    return dt.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def find_file(site, stamp):
    r = requests.get(f"{BASE_URL}/{site}/", timeout=30)
    r.raise_for_status()
    m = re.search(r'href="([^"]*-' + stamp + r'-de[^"]*)"', r.text)
    return m.group(1) if m else None


def download_all(slot):
    stamp = slot.strftime("%Y%m%d%H%M") + "00"
    files = {}
    deadline = time.time() + WAIT_SECONDS
    attempt = 0

    while True:
        attempt += 1
        for site in SITES:
            if site in files:
                continue
            dest_dir = SRC_DIR / site
            try:
                name = find_file(site, stamp)
                if not name:
                    continue
                dest = dest_dir / name
                if not dest.exists():
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    r = requests.get(f"{BASE_URL}/{site}/{name}", timeout=60)
                    r.raise_for_status()
                    dest.write_bytes(r.content)
                files[site] = dest
                print(f"[{site}] {name}")
            except Exception as e:
                print(f"[{site}] Versuch {attempt}: {e}")

        missing = [s for s in SITES if s not in files]
        if not missing:
            break
        if time.time() >= deadline:
            print(f"Nicht verfuegbar nach {WAIT_SECONDS} s: {', '.join(missing)}")
            break
        print(f"Versuch {attempt}: noch {len(missing)} fehlen ({', '.join(missing)}) - warte {RETRY_SLEEP} s")
        time.sleep(RETRY_SLEEP)

    if len(files) < MIN_SITES:
        raise RuntimeError(f"Nur {len(files)} von {len(SITES)} Standorten (min. {MIN_SITES}) - Abbruch.")
    print(f"{len(files)}/{len(SITES)} Standorte vorhanden")
    return files


# --------------------------------------------------------------------------- #
# HDF5 lesen
# --------------------------------------------------------------------------- #
def _scalar(v):
    if isinstance(v, bytes):
        return v.decode(errors="ignore")
    return v.item() if isinstance(v, np.ndarray) and v.size == 1 else v


def _find_radar_lonlat(f):
    """Radarposition aus where/how suchen (verschiedene ODIM-Schreibweisen). None, wenn nicht da."""
    for grp in ("where", "how"):
        if grp not in f:
            continue
        a = f[grp].attrs
        for lo, la in (("lon", "lat"), ("site_lon", "site_lat"), ("radar_lon", "radar_lat")):
            if lo in a and la in a:
                return float(_scalar(a[lo])), float(_scalar(a[la]))
    return None


def read_site(path):
    with h5py.File(path, "r") as f:
        w = f["where"].attrs
        ll_lon, ll_lat = float(_scalar(w["LL_lon"])), float(_scalar(w["LL_lat"]))
        xscale, yscale = float(_scalar(w["xscale"])), float(_scalar(w["yscale"]))

        what = f["dataset1/data1/what"].attrs
        gain, offset = float(_scalar(what["gain"])), float(_scalar(what["offset"]))
        nodata = _scalar(what["nodata"])
        undetect = _scalar(what["undetect"]) if "undetect" in what else None
        raw = f["dataset1/data1/data"][()]
        radar_lonlat = _find_radar_lonlat(f)

    dbz = raw.astype(np.float32) * np.float32(gain) + np.float32(offset)
    dbz[raw == nodata] = np.nan
    if undetect is not None:
        dbz[raw == undetect] = np.nan

    return {
        "dbz": dbz,
        "covered": raw != nodata,
        "ll_lon": ll_lon, "ll_lat": ll_lat,
        "xscale": xscale, "yscale": yscale,
        "radar_lonlat": radar_lonlat,
    }


# --------------------------------------------------------------------------- #
# Komposit
# --------------------------------------------------------------------------- #
def place_site(site, to_comp, comp, cover, echo_dist, name):
    """Legt das Standortbild ins HX-Raster. echo_dist = Entfernung (Pixel) des Radars,
    dessen Echo-Wert aktuell im Pixel steht (inf = noch kein Echo)."""
    if abs(site["xscale"] - PIXEL_M) > 1e-6 or abs(site["yscale"] - PIXEL_M) > 1e-6:
        print(f"[{name}] uebersprungen: Pixelgroesse {site['xscale']} m statt {PIXEL_M} m")
        return

    ny, nx = site["dbz"].shape
    x_ll, y_ll = to_comp.transform(site["ll_lon"], site["ll_lat"])
    col_f = (x_ll - LL_X) / PIXEL_M
    row_f = (TOP_Y - (y_ll + ny * PIXEL_M)) / PIXEL_M
    col0, row0 = int(round(col_f)), int(round(row_f))
    resid = max(abs(col_f - col0), abs(row_f - row0))
    if resid > 0.1:
        print(f"[{name}] WARNUNG: Raster nicht pixelgenau (Rest {resid:.2f} px)")

    r0, r1 = max(row0, 0), min(row0 + ny, HEIGHT)
    c0, c1 = max(col0, 0), min(col0 + nx, WIDTH)
    if r0 >= r1 or c0 >= c1:
        print(f"[{name}] liegt ausserhalb des Rasters")
        return
    sr0, sc0 = r0 - row0, c0 - col0
    sl = (slice(sr0, sr0 + (r1 - r0)), slice(sc0, sc0 + (c1 - c0)))
    sub_d = site["dbz"][sl]
    sub_c = site["covered"][sl]

    # Radarposition im HX-Raster (Pixel, Zeile/Spalte)
    if site["radar_lonlat"] is not None:
        xr, yr = to_comp.transform(*site["radar_lonlat"])
        rc, rr = (xr - LL_X) / PIXEL_M, (TOP_Y - yr) / PIXEL_M
        src = "aus Datei"
    else:
        rc, rr = col0 + nx / 2.0, row0 + ny / 2.0
        src = "Bildmitte (Annahme)"

    rows = np.arange(r0, r1, dtype=np.float32)[:, None]
    cols = np.arange(c0, c1, dtype=np.float32)[None, :]
    dist = np.hypot(rows - rr, cols - rc).astype(np.float32)

    cur = comp[r0:r1, c0:c1]
    cur_dist = echo_dist[r0:r1, c0:c1]

    # Echo dieses Radars: gueltig und >= DBZ_MIN
    has_echo = np.isfinite(sub_d) & (sub_d >= DBZ_MIN - 1e-6)
    # gewinnt, wo noch kein Echo steht oder dieser Radar naeher dran ist
    wins = has_echo & (dist < cur_dist)
    cur[wins] = sub_d[wins]
    cur_dist[wins] = dist[wins]

    cover[r0:r1, c0:c1] |= sub_c
    print(f"[{name}] bei Spalte {col0}, Zeile {row0}; Radarposition {src}: Spalte {rc:.0f}, Zeile {rr:.0f}")


def encode_codes(comp, cover):
    codes = np.full(comp.shape, NODATA_CODE, dtype=np.uint8)
    codes[cover] = 0
    valid = np.isfinite(comp) & (comp >= DBZ_MIN - 1e-6)
    k = np.floor((comp[valid] - DBZ_MIN) / DBZ_STEP + 1e-6)
    codes[valid] = np.clip(k, 0, MAX_CODE).astype(np.uint8)
    return codes


def build_composite(slot):
    files = download_all(slot)

    to_comp = Transformer.from_crs("EPSG:4326", COMP_PROJDEF, always_xy=True)
    comp = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
    cover = np.zeros((HEIGHT, WIDTH), dtype=bool)
    echo_dist = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float32)

    for name, path in files.items():
        try:
            place_site(read_site(path), to_comp, comp, cover, echo_dist, name)
        except Exception as e:
            print(f"[{name}] Fehler: {e}")

    return comp, cover


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", help="UTC, JJJJMMTTHHMM")
    args = parser.parse_args()

    slot = get_slot(args.time)
    print(f"Slot: {slot:%Y-%m-%d %H:%M} UTC")

    comp, cover = build_composite(slot)
    codes = encode_codes(comp, cover)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"px250_{slot.astimezone(BERLIN):%Y%m%d_%H%M}"
    np.ascontiguousarray(codes).tofile(out_path)
    print(f"Gespeichert: {out_path} ({WIDTH}x{HEIGHT}, {codes.nbytes} Bytes)")


if __name__ == "__main__":
    main()