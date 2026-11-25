"""
Command line: compute biometeo.js indices on GRIB, NetCDF or GeoTIFF files.

Examples::

    python -m salutepy.raster run.grb2 -i utci,pet,humidex -o indices.nc -w 8
    python -m salutepy.raster era5.nc -i all --deaccumulate -o era5_indices.nc
    python -m salutepy.raster tair.tif --var rh=rh.tif --var tmrt=mrt.tif -i utci,pet \\
        --time 2026-07-15T12:00 -o out.tif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engine import compute, list_indices
from .io import open_raster, open_rasters, write_geotiff, write_netcdf
from .variables import standardize


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m salutepy.raster", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="GRIB, NetCDF or GeoTIFF file (for GeoTIFF: the air temperature, "
                                             "unless --var is used for all variables)")
    ap.add_argument("-i", "--indices", default="utci", help=f"comma separated list or 'all'. Available: {', '.join(list_indices())}")
    ap.add_argument("-o", "--output", required=True, help="output file: .nc (all variables) or .tif (one file per variable)")
    ap.add_argument("-w", "--workers", type=int, default=None, help="parallel processes (default: all CPUs)")
    ap.add_argument("--var", action="append", default=[], metavar="NAME=FILE",
                    help="extra single-variable file, e.g. tmrt=mrt.tif (GeoTIFF/NetCDF)")
    ap.add_argument("--map", action="append", default=[], metavar="STD=NAME",
                    help="force the variable used for a standard input, e.g. t=TMP_2m")
    ap.add_argument("--time", default=None, help="valid time for files without time (ISO, UTC)")
    ap.add_argument("--deaccumulate", action="store_true", help="de-accumulate radiation given in J/m2")
    ap.add_argument("--no-classes", action="store_true", help="do not write the index classes")
    ap.add_argument("--tiff-units", action="append", default=[], metavar="NAME=UNITS",
                    help="units of a GeoTIFF variable, e.g. t=K (default: degC)")
    ap.add_argument("--vpd-unit", default="hPa", choices=["hPa", "kPa"], help="unit of the vapour pressure deficit (vpd)")
    args = ap.parse_args(argv)

    files = dict(kv.split("=", 1) for kv in args.var)
    units = dict(kv.split("=", 1) for kv in args.tiff_units)
    if args.input:
        if Path(args.input).suffix.lower() in (".tif", ".tiff"):
            files = {"t": args.input, **files}
            ds = open_rasters(files, units=units) if len(files) > 1 else open_raster(args.input, names=["t"], units=units)
        else:
            ds = open_raster(args.input)
            if files:
                import xarray as xr

                extra = open_rasters(files)
                ds = xr.merge([ds, extra], compat="override", join="left", combine_attrs="override")
    elif files:
        ds = open_rasters(files, units=units)
    else:
        ap.error("give an input file or --var files")
    std = standardize(ds, mapping=dict(kv.split("=", 1) for kv in args.map), deaccumulate=args.deaccumulate)
    out = compute(std, args.indices, workers=args.workers, classes=not args.no_classes, time=args.time,
                  params={"vpd": {"unit": args.vpd_unit}})
    outp = Path(args.output)
    if outp.suffix.lower() in (".tif", ".tiff"):
        for v in out.data_vars:
            p = outp.with_name(f"{outp.stem}_{v}{outp.suffix}")
            write_geotiff(out[v].assign_attrs(**{k: out.attrs[k] for k in ("transform", "crs") if k in out.attrs}), p)
            print(p)
    else:
        write_netcdf(out, outp)
        print(outp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
