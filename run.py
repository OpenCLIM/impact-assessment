import os
import glob
from glob import glob
import pandas as pd
import rasterio as rio
from rasterio import features
from rasterstats import zonal_stats as zs
import geopandas as gpd
import numpy as np
from shapely.geometry import shape

# Set basic data paths
data_path = os.getenv('DATA','/data')
inputs_path = os.path.join(data_path, 'inputs')
outputs_path = os.path.join(data_path, 'outputs')
if not os.path.exists(outputs_path):
    os.mkdir(outputs_path)

#Set model specific data paths
flood_impact_path = os.path.join(inputs_path, 'flood_impact')
dd_curves_path = os.path.join(inputs_path, 'dd-curves')
grid_path = os.path.join(inputs_path,'grid')
run_path = os.path.join(inputs_path, 'run')
uprn_lookup = glob(os.path.join(inputs_path, 'uprn', '*.csv'))
parameters_path=os.path.join(inputs_path,'parameters')

# Identify the CityCat output raster
archive = glob(run_path + "/max_depth.tif", recursive = True)

# Set buffer and threshold for the buildings
threshold = float(os.getenv('THRESHOLD'))
print('threshold:',threshold)
buffer = 5

# Identify the building files for the baseline buildings and new buildings allocated by the udm model (if available)
buildings = glob(flood_impact_path + "/*.gpkg", recursive = True)

# Search for a parameter file which outline the input parameters defined by the user
parameter_file = glob(parameters_path + "/*.csv", recursive = True)
print('parameter_file:', parameter_file)
 
# If the parameter file exists, read in the location, chosen SSP, year and storm event
if len(parameter_file) == 1 :
    file_path = os.path.splitext(parameter_file[0])
    print('Filepath:',file_path)
    filename=file_path[0].split("/")
    print('Filename:',filename[-1])

    parameters = pd.read_csv(os.path.join(parameters_path + '/' + filename[-1] + '.csv'))
    location = parameters.loc[0][1]
    ssp = parameters.loc[1][1]
    year = parameters.loc[2][1]
    depth1 = parameters.loc[5][1]

# Read in the baseline builings
with rio.open(archive[0]) as max_depth :
    e_builds = os.path.join(flood_impact_path, 'buildings_exist.gpkg')
    e_builds = gpd.read_file(e_builds, bbox=max_depth.bounds)
    # Redefine the toid number to include osgb
    columns=list(e_builds.columns)
    if 'toid_number' in columns:
        e_builds['toid'] = 'osgb' + e_builds['toid_number'].astype(str)
        e_builds.pop('toid_number')
    if 'toid_numbe' in columns:
        e_builds['toid'] = 'osgb' + e_builds['toid_numbe'].astype(str)
        e_builds.pop('toid_numbe')
    if 'building_use' in columns:
        e_builds['building_u'] = e_builds['building_use']
        e_builds.pop('building_use')

    # If there are udm buildings within the flood impact folder, read them in
    if len(buildings) == 2 :
        u_builds = os.path.join(flood_impact_path, 'buildings_udm.gpkg')
        u_builds = gpd.read_file(u_builds, bbox=max_depth.bounds)
        # Redefine the index
        u_builds['index'] = u_builds.index
        # Assign a toid
        u_builds['toid'] = 'osgb' + u_builds['index'].astype(str)
        # Note all buildings as type 'residential'
        u_builds['building_u'] = 'residential'
        # Merge the existing and building datasets
        all_buildings = u_builds.append(e_builds)
    else :
        # If there are no udm buildings, all the buildings in the simulation are defined by the baseline buildings
        all_buildings = e_builds

    # Create a copy of the original geometry
    all_buildings['geometry_copy'] = all_buildings['geometry']
    
    # Read flood depths and vd_product
    depth = max_depth.read(1)

    # Find flooded areas
    flooded_areas = gpd.GeoDataFrame(
        geometry=[shape(s[0]) for s in features.shapes(
            np.ones(depth.shape, dtype=rio.uint8), mask=np.logical_and(depth >= threshold, max_depth.read_masks(1)),
            transform=max_depth.transform)], crs=max_depth.crs)

    # Store original areas for damage calculation
    all_buildings['original_area'] = all_buildings.area

    # Buffer buildings
    all_buildings['geometry'] = all_buildings.buffer(buffer)

    # Extract maximum depth and vd_product for each building
    all_buildings['depth'] = [row['max'] for row in
                        zs(all_buildings, depth, affine=max_depth.transform, stats=['max'],
                            all_touched=True, nodata=max_depth.nodata)]

    # Filter buildings
    all_buildings = all_buildings[all_buildings['depth'] > threshold]

    # Calculate depth above floor level
    all_buildings['depth'] = all_buildings.depth - threshold
    
    # If no buildings are flooded, write an empty excel sheet and exit the code
    if len(all_buildings) == 0:
        with open(os.path.join(outputs_path, 'buildings.csv'), 'w') as f:
            f.write('')
        exit(0)                   
                                
    # Read in the preassigned damage curves
    residential = pd.read_csv(os.path.join(dd_curves_path, 'residential.csv'))
    nonresidential = pd.read_csv(os.path.join(dd_curves_path, 'nonresidential.csv'))

    # Calculate damage based on property types
    all_buildings['damage'] = (np.interp(
        all_buildings.depth, residential.depth, residential.damage) * all_buildings.original_area).round(0)
    all_buildings['damage'] = all_buildings['damage'].where(
        all_buildings.building_u != 'residential', (np.interp(
            all_buildings.depth, nonresidential.depth, nonresidential.damage
        ) * all_buildings.original_area).round(0)).astype(int)

    # Create a new data frame called centres which is a copy of buildings
    building_centroid=all_buildings.filter(['building_u','geometry_copy','damage','depth'])
    building_centroid['geometry'] = building_centroid['geometry_copy']
    building_centroid.pop('geometry_copy')
    building_centroid.crs=all_buildings.crs
    all_buildings.pop('geometry_copy')

    # Get the flooded perimeter length for each building
    flooded_perimeter = gpd.overlay(gpd.GeoDataFrame({'toid': all_buildings.toid}, geometry=all_buildings.geometry.boundary,
                                                    crs=all_buildings.crs), flooded_areas)
    flooded_perimeter['flooded_perimeter'] = flooded_perimeter.geometry.length.round(2)

    all_buildings['perimeter'] = all_buildings.geometry.length

    all_buildings = all_buildings.merge(flooded_perimeter, on='toid', how='left')
    all_buildings['flooded_perimeter'] = all_buildings.flooded_perimeter.divide(
        all_buildings.perimeter).fillna(0).multiply(100).round(0).astype(int)

    # Lookup UPRN if available
    if len(uprn_lookup) > 0:
        uprn = pd.read_csv(uprn_lookup[0], usecols=['IDENTIFIER_1', 'IDENTIFIER_2'],
                        dtype={'IDENTIFIER_1': str}).rename(columns={'IDENTIFIER_1': 'uprn',
                                                                        'IDENTIFIER_2': 'toid'})
        all_buildings = all_buildings.merge(uprn, how='left')
    
    # Save building centroids and their geometrys to CSV
    building_centroid.to_csv(
        os.path.join(outputs_path, 'affected_buildings_' + location + '_' + ssp + '_'  + year + '_' + depth1 +'mm.csv'), index=False,  float_format='%g') 
    
    # Read in the 1km OS grid cells
    km_grid = glob(grid_path + "/*.gpkg", recursive = True)
    grid = gpd.read_file(km_grid[0],bbox=max_depth.bounds)


# Apply the centroid function to the geometry column to determin the centre of each polygon
building_centroid.geometry=building_centroid['geometry'].centroid
# Ensure buildings and the new data frame have the same coordinate system
#building_centroid.crs=buildings.crs
# Redefine the index layer with sequential numbers
building_centroid=building_centroid.assign(Index=range(len(building_centroid))).set_index('Index')

grid.set_crs(epsg=27700, inplace=True)
building_centroid.set_crs(epsg=27700, inplace=True)

pointsInPolygon = gpd.sjoin(grid,building_centroid, how="inner", op="intersects")

pointsInPolygon['building_u']=pointsInPolygon['building_u'].fillna('unknown')

print('pointsInPolygon:', pointsInPolygon.head())

dfpivot = pd.pivot_table(pointsInPolygon,index='tile_name',
                         columns='building_u',aggfunc={'building_u':len}, fill_value=0)

dfpivot2 = pd.pivot_table(pointsInPolygon,index='tile_name', aggfunc={'damage':np.sum,                                                                
                                                                    'depth':np.average,
                                                                    'index_right':len}, fill_value=0)

stacked = dfpivot.stack(level = [0])

half_data = pd.merge(stacked,grid, on='tile_name')
all_data = pd.merge(dfpivot2,half_data, on='tile_name')

list=list(all_data.columns.values)
print('List:',list)

all_data['Total_Building_Count'] = all_data['index_right']
all_data.pop('index_right')

if 'residential' in list:
    all_data['Residential_Count'] = all_data['residential']
    all_data.pop('residential')
else:
   all_data['Residential_Count']=[0 for n in range(len(all_data))]

if 'non-residential' in list:
    all_data['Non_Residential_Count'] = all_data['non-residential']
    all_data.pop('non-residential')
else:
   all_data['Non_Residential_Count']=[0 for n in range(len(all_data))]

if 'mixed' in list:
    all_data['Mixed_Count'] = all_data['mixed']
    all_data.pop('mixed')
else:
   all_data['Mixed_Count']=[0 for n in range(len(all_data))]

if 'unclassified' in list:   
    all_data['Unclassified_Count'] = all_data['unclassified']
    all_data.pop('unclassified')
else:
   all_data['Unclassified_Count']=[0 for n in range(len(all_data))]

if 'unknown' in list:   
    all_data['Unknown_Count'] = all_data['unknown']
    all_data.pop('unknown')
else:
   all_data['Unknown_Count']=[0 for n in range(len(all_data))]


all_data['Damage'] = all_data['damage']
all_data.pop('damage')
all_data['Depth'] = all_data['depth']
all_data.pop('depth')

all_data.to_csv(
    os.path.join(outputs_path, '1km_data_' + location + '_' + ssp + '_'  + year + '_' + depth1 +'mm.csv'), index=False,  float_format='%g') 
