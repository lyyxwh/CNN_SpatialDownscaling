# configs/local.py

local_config = {
    "data_paths": {
        "dem": "I:\\SRTM_DEM\\2018_1000\\SRTM_DEM_1000_clip.img",
        "slope": "I:\\SRTM_DEM\\2018_1000\\SRTM_DEM_1000_slope_clip.img",
        "aspect": "I:\\SRTM_DEM\\2018_1000\\SRTM_DEM_1000_aspect_clip.img",
        "clcd": "I:\\CLCD\\2018_1000m\\CLCD_v01_2018_albert_pr_re_clip.tif",
        "era5": "I:\\CNN_SpatialDownscaling\\Processed_Data\\processed_era5",
        "smap": "I:\\CNN_SpatialDownscaling\\Processed_Data\\processed_smap",
        "modis": "G:\\CNN_SpatialDownscaling\\MODIS11\\2018_clip",
        "ndvi": "I:\\CNN_SpatialDownscaling\\Processed_Data\\processed_ndvi"
    },
    
    "var_names": {
        "era5": ["t2m", "ssrd", "strd", "vpd", "d2m", "rh"],
        "smap": "sm_surface_wetness",
        "modis": "LST_1km",
        "ndvi": "_1_km_16_days_NDVI"
    },
    
    "resolution": 0.008333333
}
