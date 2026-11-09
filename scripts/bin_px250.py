#!/usr/bin/env python3
"""PX250-Standortdaten (DWD, 17 Radare, ODIM-HDF5) -> ein Komposit im NATIVEN HX-Raster (250 m).

Ueberlappung (wradlib.comp):
  Gewicht = exp(-r/D) (steil, Zhang et al. 2005), nur die MAX_RADARS besten Radare je Pixel.
  weighted: gewichtetes Mittel in linearem Z; ko: naechster Radar gewinnt.
  NOECHO_AS_ZERO: misst ein naeherer Radar "kein Echo", verduennt das ein Fernbereichs-Echo
  des anderen Radars (statt es wie beim Maximum ungeprueft zu uebernehmen).

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
import wradlib as wrl
from scipy import ndimage
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

# Komposition mit wradlib
WEIGHT_D_KM = 40.0      # Gewicht = exp(-r / D); kleiner = naeherer Radar dominiert staerker (Zhang: 25-50 km)
MAX_RADARS = 2          # pro Pixel nur die N Radare mit dem hoechsten Gewicht beruecksichtigen (Jurczyk: 2)
NOECHO_AS_ZERO = True   # "Radar misst, aber kein Echo" zaehlt als Z=0 mit Gewicht (verduennt Fernbereichs-Fehlechos)
BAND_ROWS = 400         # Komposit zeilenweise in Baendern -> wenig Speicher

# Filter gegen Fehlechos am Rand / in der Fernzone (pro Standort, VOR dem Komposit)
MAX_RANGE_KM = None     # None = AUS (keine harte Reichweitengrenze); z.B. 150.0 = Standort endet dort
EDGE_ERODE_PX = 0       # 0 = AUS; Abdeckungsrand um so viele Pixel (1 px = 250 m) nach innen schieben
MIN_NEIGHBORS = 0       # 0 = AUS; Echo-Pixel braucht >= so viele Echo-Nachbarn im 3x3-Fenster (0 = aus)


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
        projdef = _scalar(w["projdef"]) if "projdef" in w else None

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
        "projdef": projdef,
    }


# --------------------------------------------------------------------------- #
# Komposit
# --------------------------------------------------------------------------- #
def clean_site(site):
    """Entfernt Randartefakte eines Standortbilds:
    1. Abdeckung am aeusseren Rand (letzte Entfernungstore, Sektorraender) nach innen schieben
    2. isolierte Einzelpixel (Rauschen/Speckle) entfernen, nur Echo mit genug Echo-Nachbarn bleibt"""
    dbz, cov = site["dbz"], site["covered"]
    if EDGE_ERODE_PX > 0:
        cov = ndimage.binary_erosion(cov, structure=np.ones((3, 3), bool),
                                     iterations=EDGE_ERODE_PX, border_value=1)
    echo = np.isfinite(dbz) & (dbz >= DBZ_MIN - 1e-6) & cov
    if MIN_NEIGHBORS > 0:
        kernel = np.ones((3, 3), np.uint8)
        kernel[1, 1] = 0
        nb = ndimage.convolve(echo.astype(np.uint8), kernel, mode="constant", cval=0)
        echo &= nb >= MIN_NEIGHBORS
    return np.where(echo, dbz, np.nan).astype(np.float32), cov


def prepare_site(site, to_comp, name):  # to_comp bleibt fuer Kompatibilitaet
    """Sortiert ein Standortbild ins HX-Raster ein (reine Pixelverschiebung) und liefert
    ausgeschnittene Teilfelder: lineares Z (NaN = kein Echo), Qualitaet, Abdeckung."""
    if abs(site["xscale"] - PIXEL_M) > 1e-6 or abs(site["yscale"] - PIXEL_M) > 1e-6:
        print(f"[{name}] uebersprungen: Pixelgroesse {site['xscale']} m statt {PIXEL_M} m")
        return None

    site = dict(site)
    site["dbz"], site["covered"] = clean_site(site)

    ny, nx = site["dbz"].shape
    site_proj = site.get("projdef") or COMP_PROJDEF
    # Die Standortbilder liegen NICHT auf dem HX-Gitter (Rest 0.1-0.5 px je Standort) -> echt umprojizieren:
    # jeden HX-Pixelmittelpunkt in die Standortprojektion bringen und dort den naechsten Pixel nehmen.
    t_ll = Transformer.from_crs("EPSG:4326", site_proj, always_xy=True)
    x_ll, y_ll = t_ll.transform(site["ll_lon"], site["ll_lat"])      # untere linke ECKE des Bildes
    t_s2h = Transformer.from_crs(site_proj, COMP_PROJDEF, always_xy=True)
    t_h2s = Transformer.from_crs(COMP_PROJDEF, site_proj, always_xy=True)

    # Footprint des Standortbildes im HX-Raster (HX-Pixelmitte: X = Spalte*250, Y = -Zeile*250)
    cx = np.array([x_ll, x_ll + nx * site["xscale"], x_ll, x_ll + nx * site["xscale"]])
    cy = np.array([y_ll, y_ll, y_ll + ny * site["yscale"], y_ll + ny * site["yscale"]])
    hx, hy = t_s2h.transform(cx, cy)
    c0 = max(int(np.floor(hx.min() / PIXEL_M)) - 2, 0)
    c1 = min(int(np.ceil(hx.max() / PIXEL_M)) + 3, WIDTH)
    r0 = max(int(np.floor(-hy.max() / PIXEL_M)) - 2, 0)
    r1 = min(int(np.ceil(-hy.min() / PIXEL_M)) + 3, HEIGHT)
    if r0 >= r1 or c0 >= c1:
        print(f"[{name}] liegt ausserhalb des Rasters")
        return None

    gx = (np.arange(c0, c1, dtype=np.float64) * PIXEL_M)[None, :] * np.ones((r1 - r0, 1))
    gy = (-np.arange(r0, r1, dtype=np.float64) * PIXEL_M)[:, None] * np.ones((1, c1 - c0))
    sx, sy = t_h2s.transform(gx, gy)
    jj = np.rint((sx - x_ll) / site["xscale"] - 0.5).astype(np.int64)       # Spalte im Standortbild
    ii = np.rint(ny - 0.5 - (sy - y_ll) / site["yscale"]).astype(np.int64)  # Zeile (0 = Norden)
    ok = (jj >= 0) & (jj < nx) & (ii >= 0) & (ii < ny)
    dbz = np.full(jj.shape, np.nan)
    covered = np.zeros(jj.shape, bool)
    dbz[ok] = site["dbz"][ii[ok], jj[ok]]
    covered[ok] = site["covered"][ii[ok], jj[ok]]

    # Radarposition im HX-Raster (Pixelindex-Einheiten)
    if site["radar_lonlat"] is not None:
        xr, yr = Transformer.from_crs("EPSG:4326", COMP_PROJDEF, always_xy=True).transform(*site["radar_lonlat"])
        src = "aus Datei"
    else:
        xr, yr = t_s2h.transform(x_ll + nx * site["xscale"] / 2.0, y_ll + ny * site["yscale"] / 2.0)
        src = "Bildmitte (Annahme)"
    rc, rr = xr / PIXEL_M, -yr / PIXEL_M
    col0, row0 = c0, r0

    rows = np.arange(r0, r1, dtype=np.float64)[:, None]
    cols = np.arange(c0, c1, dtype=np.float64)[None, :]
    r_km = np.hypot(rows - rr, cols - rc) * PIXEL_M / 1000.0
    quality = np.exp(-r_km / WEIGHT_D_KM)

    # Reichweitenbegrenzung: Echo UND Abdeckung nur innerhalb MAX_RANGE_KM
    in_range = (r_km <= MAX_RANGE_KM) if MAX_RANGE_KM else np.ones(r_km.shape, bool)
    covered = covered & in_range

    # Mittelung in linearem Z (nicht in dBZ!)
    has_echo = np.isfinite(dbz) & (dbz >= DBZ_MIN - 1e-6) & in_range
    z = np.full(dbz.shape, np.nan)
    if NOECHO_AS_ZERO:
        z[covered] = 0.0                      # Radar misst hier, aber kein Echo
    z[has_echo] = 10.0 ** (dbz[has_echo] / 10.0)
    quality = np.where(np.isfinite(z), quality, np.nan)

    print(f"[{name}] Projektion {'aus Datei' if site.get('projdef') else 'HX (Annahme)'}; Ausschnitt Spalte {c0}-{c1}, Zeile {r0}-{r1}; "
          f"Radarposition {src}: Spalte {rc:.1f}, Zeile {rr:.1f}")
    return {"name": name, "r0": r0, "r1": r1, "c0": c0, "c1": c1,
            "z": z.astype(np.float32), "q": quality.astype(np.float32), "cover": covered}


def compose_bands(placed, method="weighted"):
    """Komposit mit wradlib. Pro Zeilenband werden alle ueberlappenden Standorte auf volle
    Bandbreite gelegt und mit wrl.comp.compose_weighted / compose_ko zusammengefuehrt.
    Rueckgabe: (comp [dBZ float32, NaN = kein Echo], cover [bool])."""
    comp = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
    cover = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for p in placed:
        cover[p["r0"]:p["r1"], p["c0"]:p["c1"]] |= p["cover"]

    for b0 in range(0, HEIGHT, BAND_ROWS):
        b1 = min(b0 + BAND_ROWS, HEIGHT)
        grids, quals = [], []
        for p in placed:
            r0, r1 = max(p["r0"], b0), min(p["r1"], b1)
            if r0 >= r1:
                continue
            gz = np.full((b1 - b0, WIDTH), np.nan, dtype=np.float32)
            gq = np.full((b1 - b0, WIDTH), np.nan, dtype=np.float32)
            gz[r0 - b0:r1 - b0, p["c0"]:p["c1"]] = p["z"][r0 - p["r0"]:r1 - p["r0"]]
            gq[r0 - b0:r1 - b0, p["c0"]:p["c1"]] = p["q"][r0 - p["r0"]:r1 - p["r0"]]
            grids.append(gz)
            quals.append(gq)
        if not grids:
            continue

        if MAX_RADARS and len(grids) > MAX_RADARS:
            gz_all, gq_all = np.stack(grids), np.stack(quals)
            q_srt = np.where(np.isnan(gq_all), -np.inf, gq_all)
            thr = np.sort(q_srt, axis=0)[-MAX_RADARS]          # N-hoechstes Gewicht je Pixel
            drop = q_srt < thr
            gz_all[drop] = np.nan
            gq_all[drop] = np.nan
            grids, quals = list(gz_all), list(gq_all)

        with np.errstate(invalid="ignore", divide="ignore"):
            if len(grids) == 1:
                z = grids[0]
            elif method == "ko":
                z = wrl.comp.compose_ko(list(grids), list(quals))   # nimmt Listen mit append -> Kopien
            else:
                z = wrl.comp.compose_weighted(grids, quals)
            comp[b0:b1] = np.where(z > 0, 10.0 * np.log10(np.where(z > 0, z, 1.0)), np.nan)   # zurueck nach dBZ
    return comp, cover


def encode_codes(comp, cover):
    codes = np.full(comp.shape, NODATA_CODE, dtype=np.uint8)
    codes[cover] = 0
    valid = np.isfinite(comp) & (comp >= DBZ_MIN - 1e-3)
    k = np.floor((comp[valid] - DBZ_MIN) / DBZ_STEP + 1e-3)       # Toleranz gegen Rundung Z<->dBZ
    codes[valid] = np.clip(k, 0, MAX_CODE).astype(np.uint8)
    return codes


def build_composite(slot, method="weighted"):
    files = download_all(slot)
    to_comp = Transformer.from_crs("EPSG:4326", COMP_PROJDEF, always_xy=True)

    placed = []
    for name, path in files.items():
        try:
            p = prepare_site(read_site(path), to_comp, name)
            if p is not None:
                placed.append(p)
        except Exception as e:
            print(f"[{name}] Fehler: {e}")

    return compose_bands(placed, method)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", help="UTC, JJJJMMTTHHMM")
    parser.add_argument("--method", choices=["weighted", "ko"], default="weighted",
                        help="weighted = Mittel (1/r^2, linear Z); ko = bester (naechster) Radar")
    args = parser.parse_args()

    slot = get_slot(args.time)
    print(f"Slot: {slot:%Y-%m-%d %H:%M} UTC")

    comp, cover = build_composite(slot, args.method)
    codes = encode_codes(comp, cover)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"px250_{slot.astimezone(BERLIN):%Y%m%d_%H%M}"
    np.ascontiguousarray(codes).tofile(out_path)
    print(f"Gespeichert: {out_path} ({WIDTH}x{HEIGHT}, {codes.nbytes} Bytes)")


if __name__ == "__main__":
    main()