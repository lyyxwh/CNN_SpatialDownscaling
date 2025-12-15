# configs/global.py

global_config = {
    "data_paths": {
        "dem": "I:\\SRTM_DEM\\2018_0p0625\\SRTM_DEM_resample_clip.img",
        "slope": "I:\\SRTM_DEM\\2018_0p0625\\SRTM_DEM_resample_clip_slope.img",
        "aspect": "I:\\SRTM_DEM\\2018_0p0625\\SRTM_DEM_resample_clip_aspect.img",
        "clcd": "I:\\CLCD\\2018_0p0625\\CLCD_v01_2018__ProjectRaster_clip.tif",
        "era5": "I:\\CNN_SpatialDownscaling\\Processed_Data_0p0625\\processed_era5",
        "smap": "I:\\CNN_SpatialDownscaling\\Processed_Data_0p0625\\processed_smap",
        "cldas": "G:\\cldas\\data\\2018_clip_filtered",
        "ndvi": "I:\\CNN_SpatialDownscaling\\Processed_Data_0p0625\\processed_ndvi"
    },
    
    "var_names": {
        "era5": ["t2m", "ssrd", "strd", "vpd", "d2m", "rh"],
        "smap": "sm_surface_wetness",
        "cldas": "TG",
        "ndvi": "_1_km_16_days_NDVI"
    },
    
    "resolution": 0.0625
}
