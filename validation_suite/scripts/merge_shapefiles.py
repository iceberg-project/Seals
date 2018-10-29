import geopandas as gpd
from fiona.crs import from_epsg
import os

merged = gpd.GeoDataFrame()
merged.crs = from_epsg(3031)


for path, subfolder, file in os.walk('./sealnet_predictions_test'):
    for filename in file:
        if filename.split('_')[-1] == 'locations.dbf':
            if len(gpd.read_file('{}/{}'.format(path, filename))) > 0:
                merged = merged.append(gpd.read_file('{}/{}'.format(path, filename)), ignore_index=True)


merged.to_file('merged_locations.shp')