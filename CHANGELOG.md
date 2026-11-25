# Changelog

## 0.3.0 (2026-11-25)

- `raster.indici_tif`: UTCI, PET, WBGT sun/shade, mean radiant and globe temperature and work-time
  loss from GeoTIFF rasters without date and time (sun elevation as input).
- `pipeline.meteosalute_nc`: the 42-variable hourly NetCDF files of Meteosalute from a MOLOCH GRIB2
  run, validated against an operational file; `--all` for every hour of the run.
- `pipeline.wbgt_grib`, `pipeline.meteosalute_wbgt_nc`, `pipeline.verifica_wbgt`: WBGT in sun and
  shade from GRIB, and comparison with NetCDF files produced in R.
- Compiled (numba) WBGT of Liljegren, PET, PMV and clothing insulation.
- Ports of the HeatStress indices; corrected THI and frostbite time.
- Mean radiant temperature from globe temperature with a 0.15 m globe by default.
- Bilingual README, CITATION.cff and Zenodo metadata.

## 0.2.0

- Parallel computation of the biometeo.js indices on GRIB, NetCDF and GeoTIFF (`salutepy.raster`).

## 0.1.0

- First port of the Meteosalute R pipeline.
