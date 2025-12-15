import geopandas as gpd
from shapely.geometry import Polygon
import os

shp_path = r'G:\Agr-qu\clip.shp'

# 定义面范围（左下、左上、右上、右下、左下）
polygon = Polygon([
    (110.0, 27.0),  # 西南
    (110.0, 43.0),  # 西北
    (125.0, 43.0),  # 东北
    (125.0, 27.0),  # 东南
    (110.0, 27.0)   # 回到西南
])

gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs='EPSG:4326')

# 如果文件存在则删除
if os.path.exists(shp_path):
    os.remove(shp_path)

gdf.to_file(shp_path)
