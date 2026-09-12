# Data Sources

This document describes all data sources used in the microplastic transport prediction project.

## 1. Microplastic Observations

### 1.1 Delaware River Microplastics (2018)

- **Source:** U.S. Geological Survey (USGS)
- **URL:** https://doi.org/10.5066/P9QVIVX3
- **Provider:** USGS Idaho Water Science Center
- **License:** Public domain (U.S. Government)
- **Spatial Coverage:** Delaware River and tributaries, New York to New Jersey
- **Temporal Coverage:** July 2018 - March 2019
- **Number of Observations:** 9 water sampling locations
- **Resolution:** Point measurements at sampling locations
- **Units:** Particles per cubic meter (particles/m³)
- **Variables:** 
  - Concentration (particles/m³)
  - Particle type (fiber, fragment, film, foam, beads/pellets, tire particles)
  - Sampling method (grab sampling, net sampling)
  - Maximum concentration observed
- **Known Biases:** 
  - Baseflow conditions only (no storm events)
  - Single time-point sampling per location
  - Different sampling methods between locations
- **Access:** CSV download from USGS ScienceBase
- **Citation:** Baldwin, A.K., Spanjer, A.R., Hayhurst, B., and Hamilton, D., 2020, Microplastics in the Delaware River, 2018: U.S. Geological Survey data release, https://doi.org/10.5066/P9QVIVX3

### 1.2 Great Lakes Tributaries Microplastics (2014-2015)

- **Source:** USGS, data release for "Microplastics in 29 Great Lakes tributaries"
- **URL:** Available through USGS ScienceBase
- **Provider:** USGS
- **License:** Public domain
- **Spatial Coverage:** 29 tributaries across 6 Great Lakes states
- **Temporal Coverage:** Spring 2014 - Spring 2015
- **Number of Observations:** 29+ tributary sampling locations
- **Units:** Particles per km² (surface), estimated conversion to particles/m³
- **Variables:** Particle count, morphology, concentration
- **Known Biases:** Surface sampling only, seasonal bias (spring/summer)
- **Citation:** Baldwin, A. et al., 2016, Environmental Science & Technology

### 1.3 Northeastern U.S. Streams Microplastics (2017-2018)

- **Source:** USGS, data release for stream microplastic survey
- **Spatial Coverage:** 17 streams from New York to Virginia
- **Temporal Coverage:** 2017-2018
- **Number of Observations:** 17 locations
- **Units:** Varies (particles/L reported in some publications)
- **Known Biases:** Single sampling event per location

## 2. River Network Data

### 2.1 NHDPlus v2

- **Source:** EPA/USGS
- **URL:** https://www.epa.gov/waterdata/nhdplus-national-hydrography-dataset-plus
- **Provider:** U.S. Environmental Protection Agency, U.S. Geological Survey
- **License:** Public domain
- **Spatial Coverage:** Conterminous United States
- **Spatial Resolution:** 1:100,000 scale
- **Variables:** 
  - Stream segments (flowlines)
  - Catchments
  - Flow direction and connectivity
  - Stream order
  - Stream slope
  - Flow velocity (modeled estimates)
  - Drainage area
- **Format:** Shapefile, File Geodatabase
- **Known Biases:** Medium resolution may miss small tributaries
- **Citation:** McKay, L., Bondelid, T., Dewald, T., Johnston, J., Moore, R., and Rea, A., 2012, NHDPlus Version 2: User Guide

### 2.2 NHDPlus High Resolution (NHDPlus HR)

- **Source:** USGS
- **URL:** https://www.usgs.gov/national-hydrography/access-national-hydrography-products
- **Provider:** U.S. Geological Survey
- **License:** Public domain
- **Spatial Coverage:** CONUS, Hawaii, Puerto Rico, Guam
- **Spatial Resolution:** 1:24,000 scale
- **Variables:** 
  - High-resolution stream network
  - Elevation-derived catchments
  - Hydrologic sequencing
  - Mean annual streamflow (EROM)
  - Mean annual velocity
  - Flow withdrawals and transfers
- **Format:** File Geodatabase
- **Known Biases:** Irregular update schedule
- **Citation:** Moore, R.B., McKay, L.D., Rea, A.H., Bondelid, T.R., Price, C.V., Dewald, T.G., and Johnston, C.M., 2019

## 3. Hydrology Data

### 3.1 USGS Water Services API (NWIS)

- **Source:** USGS Water Data
- **URL:** https://api.waterdata.usgs.gov/
- **Provider:** U.S. Geological Survey
- **License:** Public domain
- **Spatial Coverage:** 13,500+ monitoring stations nationwide
- **Temporal Coverage:** Continuous real-time data with historical records
- **Temporal Resolution:** 15-minute to daily
- **Variables:**
  - Streamflow/discharge (cfs)
  - Gage height (ft)
  - Water temperature (°C)
  - Additional water quality parameters where available
- **Known Biases:** 
  - Streamgages not co-located with microplastic sampling sites
  - Potential gaps in historical records
- **Access:** REST API (JSON format)

### 3.2 NHDPlus Streamflow Estimates

- **Source:** NHDPlus v2 and HR datasets
- **Variables:** 
  - Mean annual and monthly streamflow (modeled)
  - Flow velocity estimates
- **Known Biases:** Modeled estimates, not direct observations

## 4. Meteorological Data

### 4.1 Daymet v4

- **Source:** Oak Ridge National Laboratory
- **URL:** https://daymet.ornl.gov/
- **Provider:** ORNL Distributed Active Archive Center (DAAC)
- **License:** Public domain
- **Spatial Coverage:** North America (limited by station density)
- **Spatial Resolution:** 1 km × 1 km grid
- **Temporal Coverage:** 1980-present
- **Temporal Resolution:** Daily
- **Variables:**
  - Precipitation (mm/day)
  - Maximum/minimum temperature (°C)
  - Shortwave radiation (W/m²)
  - Wind speed (m/s)
  - Snow water equivalent (mm)
  - Vapor pressure (kPa)
- **Known Biases:** 
  - Gridded interpolation may smooth local extremes
  - Coastal and mountainous areas have reduced accuracy
- **Access:** HTTP API and bulk download
- **Citation:** Thornton, M.M., Shrestha, R.M., Wei, Y., et al. 2024. Daymet: Daily surface weather data for North America, Version 4. ORNL DAAC, Oak Ridge, TN, USA.

## 5. Geographic Data

### 5.1 National Land Cover Database (NLCD) 2019

- **Source:** USGS
- **URL:** https://www.usgs.gov/landsat-missions/land-cover
- **Provider:** U.S. Geological Survey
- **License:** Public domain
- **Spatial Coverage:** Conterminous United States
- **Spatial Resolution:** 30m
- **Temporal Coverage:** 2019 (most recent available)
- **Variables:**
  - 22 land cover classes (categorical)
  - Impervious surface percentage
  - Tree canopy cover
- **Format:** GeoTIFF
- **Known Biases:** 
  - Static over short time periods
  - Mixed pixels in heterogeneous landscapes
  - Classification accuracy varies by class (80-95%)

### 5.2 National Elevation Dataset (NED)

- **Source:** USGS
- **URL:** https://www.usgs.gov/core-science-support/ngtoc/national-elevation-dataset
- **Provider:** U.S. Geological Survey
- **License:** Public domain
- **Spatial Coverage:** Conterminous United States
- **Spatial Resolution:** 10m (CONUS)
- **Variables:** 
  - Elevation (meters)
  - Slope (derived)
  - Aspect (derived)
- **Format:** GeoTIFF
- **Known Biases:** 
  - Void areas filled with interpolation
  - Vertical accuracy varies by source

### 5.3 Gridded Population Data

- **Source:** WorldPop or GPW
- **URL:** https://www.worldpop.org/
- **Provider:** WorldPop, NASA SEDAC
- **License:** CC-BY-4.0
- **Spatial Resolution:** ~100m-1km
- **Variables:** Population count, population density
- **Known Biases:** 
  - Nighttime estimation may underestimate daytime populations
  - Rural areas have higher uncertainty

## 6. Data Compatibility Notes

### Unit Harmonization

All microplastic concentrations are standardized to **particles per cubic meter** (particles/m³):
- particles/L × 1000 = particles/m³
- particles/km² converted using sampling depth (assumed 0.1-0.5m)

### Spatial Alignment

All datasets are reprojected to **EPSG:4326** (WGS84) for consistency:
- NHDPlus data: Native projection → WGS84
- Daymet data: Native grid → WGS84 extraction
- NLCD/NED: Native projection → WGS84 extraction

### Temporal Alignment

Features are aligned to microplastic sampling dates:
- Environmental data extracted for the sampling date and preceding days
- Lag features computed for 1-day, 3-day, and 7-day lags
- No future information used in features (verified by temporal leakage audit)

### Missing Data Handling

1. **Streamgage data:** If a streamgage lacks measurements on a sampling date, interpolate from adjacent dates (max 7-day gap)
2. **Daymet data:** Grid cells missing data filled using inverse-distance weighting
3. **Microplastic samples:** Samples without concentration measurements excluded
4. **Geospatial data:** NLCD/NED coverage is complete for CONUS - no missing data expected
