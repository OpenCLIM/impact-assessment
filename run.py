import os
import glob
from glob import glob
import pandas as pd
import rasterio as rio
from rasterio import features
from rasterio.crs import CRS
from rasterstats import zonal_stats as zs
import geopandas as gpd
import shutil
import numpy as np
from shapely.geometry import shape
import rtree
import re
from shapely.geometry import shape, Point
import datetime
from operator import itemgetter

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
udm_para_in_path = os.path.join(inputs_path, 'udm_parameters')

categorys_path = os.path.join(inputs_path, 'categories')
if not os.path.exists(categorys_path):
    os.mkdir(categorys_path)

def process_data(ev):
    
    T_start = datetime.datetime.now()
    builds_file = all_buildings
    input_folder = run_path
    
    print(builds_file)
        
    #Creating the spatial index with the rtree module is only done for ine depth file using X,Y only
    print('..creating spatial index..')
    
    #Find the csv file:
    csv_lookup = glob(os.path.join(run_path, '*.csv'))
     
    #first get the resolution of the grid:
    df_res = pd.read_csv(csv_lookup[0], nrows = 3)
    xdiff = df_res.iloc[2, 0] - df_res.iloc[1, 0]
    ydiff = df_res.iloc[2, 1] - df_res.iloc[1, 1]
    if xdiff != 0:
        dx = xdiff
    elif xdiff == 0:
        dx = ydiff
    del(df_res)
    buffer_distance = ((buffer_value)/100) * dx # in % grid resolution
    
    x = []
    y = []
    with open(csv_lookup[0], 'r') as t:
        aline = t.readline().strip()
        aline = t.readline()
        while aline != '':
            column = re.split('\s|\s\s|\t|,',str(aline))
            x.append(float(column[0]))
            y.append(float(column[1]))
            aline = t.readline()
    t.close()
    
    cell_idx = []
    for idx, xi in enumerate(x):
        cell_idx.append(idx)
        
    index = rtree.index.Index() #create the spatial index
    for pt_idx, xi, yi in zip(cell_idx, x, y):
        index.insert(pt_idx, (xi, yi))
        
    del(cell_idx)
    
    cell_index = []
    buffer_list = []
    builds = builds_file
    builds_n = len(builds)
    builds_df = gpd.GeoDataFrame(builds[[str(builds_field1), 'geometry']])
    del(builds)
    
    for b_id, b_geom in zip(builds_df[str(builds_field1)], builds_df['geometry']):
        buffer = shape(b_geom.buffer(float(buffer_distance), resolution=10)) #create a buffer polygon for the building polygons from resolution 10 to 16
        for cell in list(index.intersection(buffer.bounds)): #first check if the point is within the bounding box of a building buffer
            cell_int = Point(x[cell], y[cell])  
            if cell_int.intersects(buffer): #then check if the point intersects with buffer polygon
                buffer_list.append(b_id) #store the building ID
                cell_index.append(cell) #store the line inedex of the intersecting points
                
    df_b = pd.DataFrame(list(zip(buffer_list, cell_index)), columns = [str(builds_field1), 'cell'])
    df_b = df_b.sort_values(by = ['cell'])
    print('spatial index created')
    
    #------------------------------------------------------------------------------reading depth files 
        
    files = csv_lookup
    print('files:',files)
    
    for i, filename in enumerate(files):
        f = open(filename)
        print('processing file: ' + str(filename))
        Z=[]
        aline = f.readline().strip()       
        aline = f.readline()
        while aline != '':
            column = re.split('\s|\s\s|\t|,',str(aline))
            Z.append(float(column[2]))
            aline = f.readline()
        f.close()
        
        #--------------------------------------------------------------------------spatial intersection and classification
        #the next line reads the depth values from the file according to cell index from above and stores the depth with the intersecting building ID
        df = pd.DataFrame(list(zip(itemgetter(*cell_index)(Z),buffer_list)), columns=['depth',str(builds_field1)])
        del(Z)
        
        #based on the building ID the mean and maximum depth are established and stored in a new data frame:
        mean_depth = pd.DataFrame(df.groupby([str(builds_field1)])['depth'].mean().astype(float)).round(3).reset_index(level=0).rename(columns={'depth':'mean_depth'}) 
        p90ile_depth = pd.DataFrame(df.groupby([str(builds_field1)])['depth'].quantile(0.90).astype(float)).round(3).reset_index(level=0).rename(columns={'depth':'p90ile_depth'})
        damages_df = pd.merge(mean_depth, p90ile_depth)
        del(mean_depth, p90ile_depth)
        
        #calculate the damages according to the water depth in buffer zone and the type of the building
        damages_df['Class'] = 'A) Low'
        damages_df['Class'][(damages_df['mean_depth'] >= 0) & (damages_df['mean_depth'] < 0.10) & (damages_df['p90ile_depth'] < 0.30)] = 'A) Low'
        damages_df['Class'][(damages_df['mean_depth'] >= 0) & (damages_df['mean_depth'] < 0.10) & (damages_df['p90ile_depth'] >= 0.30)] = 'B) Medium'
        damages_df['Class'][(damages_df['mean_depth'] >= 0.10) & (damages_df['mean_depth'] < 0.30) & (damages_df['p90ile_depth'] < 0.30)] = 'B) Medium' 
        damages_df['Class'][(damages_df['mean_depth'] >= 0.10) & (damages_df['p90ile_depth'] >= 0.30)] = 'C) High'  
        
        #------------------------------------------------------------------------------merge results with a copy of the building layer and create output files
        builds_data = builds_file
        builds_df = gpd.GeoDataFrame(builds_data[[str(builds_field1), 'geometry']])
        finalf = builds_df.merge(damages_df, on = str(builds_field1), how = 'left') #the merging of the building shapefile
        
        finalf['Area'] = (finalf.area).astype(int)#calculate the area for each building
        finalf.to_file(os.path.join(categorys_path, 
                        location + '-' + ssp + '-' + year + '-' + depth1 + '_exposure.shp'))
        class_low = (finalf['Class'] == 'A) Low').sum()
        class_medium = (finalf['Class'] == 'B) Medium').sum()        
        class_high = (finalf['Class'] == 'C) High').sum()
        del(damages_df)
        
        del(finalf['geometry'])
        finalf_csv = pd.DataFrame(finalf)
        finalf_csv.to_csv(os.path.join(categorys_path, 
                            location + '-' + ssp + '-' + year + '-' + depth1 + '_exposure.csv'))
        
        del(builds_data, builds_df, finalf, finalf_csv, df)
        
        with open(os.path.join(outputs_path,
                    location + '-' + ssp + '-' + year + '-' + depth1 + '_exposure_summary.txt'), 'w') as sum_results:
            sum_results.write('Summary of Exposure Analysis for: ' + str(filename) + '\n\n'
                        + 'Number of Buildings: ' + str(builds_n) + '\n'
                        + 'Grid Resolution: ' + str(dx) + 'm' + '\n'
                        + 'Buffer Distance: ' +str(buffer_value) + '% or' + str(buffer_distance) + 'm' + '\n\n'
                        + 'Low: ' + str(class_low) + '\n'
                        + 'Medium: ' +str(class_medium) + '\n'
                        + 'High: ' +str(class_high) + '\n\n')
            sum_results.close()
            
    del(x, y)
    del(buffer_list, cell_index, df_b)
    print('The Exposure Analysis is Finished. Time required: ' + str(datetime.datetime.now() - T_start)[:-4])


# Identify the CityCat output raster
archive = glob(run_path + "/*.tif", recursive = True)

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

    src = parameter_file[0]
    dst = os.path.join(parameters_out_path,filename[-1] + '.csv')
    shutil.copy(src,dst)

    parameters = pd.read_csv(os.path.join(parameters_path + '/' + filename[-1] + '.csv'))
    location = parameters.loc[0][1]
    ssp = parameters.loc[1][1]
    year = parameters.loc[2][1]
    depth1 = parameters.loc[4][1]
    
    # Define output path for parameters
    parameters_out_path=os.path.join(outputs_path,'parameters')
    if not os.path.exists(parameters_out_path):
        os.mkdir(parameters_out_path)
    
if len(parameter_file) == 0:
    location = os.getenv('LOCATION')
    ssp = os.getenv('SSP')
    year = os.getenv('YEAR')
    depth1 = os.getenv('TOTAL_DEPTH')
    
dtmres = int(os.getenv('DTM_RESOLUTION'))

# Read in the baseline builings
with rio.open(archive[0],'r+') as max_depth :
    # Set crs of max_depth raster
    max_depth.crs = CRS.from_epsg(27700)
    # Find existing buildings
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
        u_builds.crs = e_builds.crs
        # Merge the existing and building datasets
        all_buildings = u_builds.append(e_builds)
    else :
        # If there are no udm buildings, all the buildings in the simulation are defined by the baseline buildings
        all_buildings = e_builds
        
    # Create a list of all of the column headers in the buildings file:
    cols_list = []
    for n in all_buildings.columns:
        cols_list.append(n)
    print('cols_list:',cols_list)

    #import the field of buildings
    builds_field1 = "toid" #check the field from the shapefile
    builds_field2 = "building_use" #check the field from the shapefile
    
    if dtmres == 5:
        buffer_value = 100
    else:
        buffer_value = 150
    print('buffer_value:',buffer_value)
    
    process_data('Risk_Levels')
    
    #Create a copy of the original geometry
    #all_buildings['geometry_copy'] = all_buildings['geometry']
    
    class_data = glob(categorys_path + "/*.shp", recursive = True)
    class_data = gpd.read_file(class_data[0], bbox=max_depth.bounds)
    
    all_buildings.set_crs(epsg=27700, inplace=True)
    class_data.set_crs(epsg=27700, inplace=True)

    all_buildings1 = gpd.sjoin(all_buildings, class_data,how='left',op="contains")
    
    #all_buildings1.to_file(os.path.join(outputs_path, 'builds.gpkg'),driver='GPKG')
    
    #Create a copy of the original geometry
    all_buildings1['geometry_copy'] = all_buildings1['geometry']
    
    # Read flood depths and vd_product
    depth = max_depth.read(1)

    # Find flooded areas
    flooded_areas = gpd.GeoDataFrame(
        geometry=[shape(s[0]) for s in features.shapes(
            np.ones(depth.shape, dtype=rio.uint8), mask=np.logical_and(depth >= threshold, max_depth.read_masks(1)),
            transform=max_depth.transform)], crs=max_depth.crs)

    # Store original areas for damage calculation
    all_buildings1['original_area'] = all_buildings1.area

    # Buffer buildings
    all_buildings1['geometry'] = all_buildings1.buffer(buffer)

    # Extract maximum depth and vd_product for each building
    all_buildings1['depth'] = [row['max'] for row in
                        zs(all_buildings1, depth, affine=max_depth.transform, stats=['max'],
                            all_touched=True, nodata=max_depth.nodata)]

    # Filter buildings
    all_buildings1 = all_buildings1[all_buildings1['depth'] > threshold]

    # Filter buildings
    all_buildings1 = all_buildings1[all_buildings1['Class'] != "A) Low"]

    # Calculate depth above floor level
    all_buildings1['depth'] = all_buildings1.depth - threshold
    
    # If no buildings are flooded, write an empty excel sheet and exit the code
    if len(all_buildings1) == 0:
        with open(os.path.join(outputs_path, 'buildings.csv'), 'w') as f:
            f.write('')
        exit(0)                   
                
    # all_buildings1.to_csv(
    #       os.path.join(outputs_path, 'affected_buildings_' + location + '_' + ssp + '_'  + year + '_' + depth1 +'mm.csv'), index=False,  float_format='%g')                  
                                
    # Read in the preassigned damage curves
    residential = pd.read_csv(os.path.join(dd_curves_path, 'residential.csv'))
    nonresidential = pd.read_csv(os.path.join(dd_curves_path, 'nonresidential.csv'))

    # Calculate damage based on property types
    res_data = all_buildings1.loc[all_buildings1['building_u']=='residential']
    non_res_data = all_buildings1.loc[all_buildings1['building_u']!='residential']

    res_data['damage'] = (np.interp(
        res_data.depth, residential.depth, residential.damage) * res_data.original_area).round(0)
    non_res_data['damage'] = (np.interp(
        non_res_data.depth, nonresidential.depth, nonresidential.damage) * non_res_data.original_area).round(0).astype(int)

    all_buildings1=res_data.append(non_res_data)


    # # Lookup UPRN if available
    # if len(uprn_lookup) > 0:
    #     uprn = pd.read_csv(uprn_lookup[0], usecols=['IDENTIFIER_1', 'IDENTIFIER_2'],
    #                     dtype={'IDENTIFIER_1': str}).rename(columns={'IDENTIFIER_1': 'uprn',
    #                                                                     'IDENTIFIER_2': 'toid'})
    #     if len(uprn) != 0:
    #         all_buildings1 = all_buildings1.merge(uprn, how='left')

    # Create a new data frame called centres which is a copy of buildings
    building_centroid=all_buildings1.filter(['building_u','geometry_copy','damage','depth','Class'])
    building_centroid['geometry'] = building_centroid['geometry_copy']
    building_centroid.pop('geometry_copy')
    building_centroid.crs=e_builds.crs
    all_buildings1.pop('geometry_copy')

    building_centroid['building_u']=building_centroid['building_u'].fillna('unknown')
    
    #Save building centroids and their geometrys to CSV
    # building_centroid.to_csv(
    #     os.path.join(outputs_path, 'affected_buildings_' + location + '_' + ssp + '_'  + year + '_' + depth1 +'mm.csv'), index=False,  float_format='%g') 
    
    # Read in the 1km OS grid cells
    km_grid = glob(grid_path + "/*.gpkg", recursive = True)
    grid = gpd.read_file(km_grid[0],bbox=max_depth.bounds)
    grid.set_crs(epsg=27700, inplace=True)

# Create a geo dataframe for the centroids
centre = gpd.GeoDataFrame(building_centroid,geometry="geometry",crs="EPSG:27700")

# Apply the centroid function to the geometry column to determine the centre of each polygon
centre.geometry=centre['geometry'].centroid

grid.set_crs(epsg=27700, inplace=True)

pointsInPolygon = gpd.sjoin(grid,centre, how="left", op="intersects")

pointsInPolygon=pointsInPolygon.fillna(0)

# pointsInPolygon.to_csv(
#             os.path.join(outputs_path, 'pointsInPolygon_' + location + '_' + ssp + '_'  + year + '_' + depth1 +'.csv'), index=False,  float_format='%g') 

dfpivot = pd.pivot_table(pointsInPolygon,index='tile_name',
                        columns='building_u',aggfunc={'building_u':len}, fill_value=0)

dfpivot2 = pd.pivot_table(pointsInPolygon,index='tile_name', aggfunc={'damage':np.sum,                                                                
                                                                    'depth':np.average,
                                                                    'index_right':len}, fill_value=0)

stacked = dfpivot.stack(level = [0])

half_data=pd.DataFrame()
all_data=pd.DataFrame()

half_data = pd.merge(stacked,grid, on='tile_name')
all_data = pd.merge(dfpivot2,half_data, on='tile_name')

check = list(all_data.columns.values)

print('check:',check)

#all_data['Total_Building_Count'] = all_data['index_right']
all_data.pop('index_right')

if 'residential' in check:
    all_data['Residential_Count'] = all_data['residential']
    all_data.pop('residential')
else:
    all_data['Residential_Count']=[0 for n in range(len(all_data))]

if 'non-residential' in check:
    all_data['Non_Residential_Count'] = all_data['non-residential']
    all_data.pop('non-residential')
else:
    all_data['Non_Residential_Count']=[0 for n in range(len(all_data))]

if 'mixed' in check:
    all_data['Mixed_Count'] = all_data['mixed']
    all_data.pop('mixed')
else:
    all_data['Mixed_Count']=[0 for n in range(len(all_data))]

if 'unclassified' in check:   
    all_data['Unclassified_Count'] = all_data['unclassified']
    all_data.pop('unclassified')
else:
    all_data['Unclassified_Count']=[0 for n in range(len(all_data))]

if 'unknown' in check:   
    all_data['Unknown_Count'] = all_data['unknown']
    all_data.pop('unknown')
else:
    all_data['Unknown_Count']=[0 for n in range(len(all_data))]

all_data['Total_Building_Count'] = (all_data['Residential_Count'] + all_data['Non_Residential_Count'] +
        all_data['Mixed_Count'] + all_data['Unclassified_Count'] + all_data['Unknown_Count'])

all_data['Damage'] = all_data['damage']
all_data.pop('damage')
all_data['Depth'] = all_data['depth']
all_data.pop('depth')

# If linked to UDM results, pass the udm details through to the outputs
udm_para_out_path = os.path.join(outputs_path, 'udm_parameters')
if not os.path.exists(udm_para_out_path):
    os.mkdir(udm_para_out_path)

meta_data_txt = glob(udm_para_in_path + "/**/metadata.txt", recursive = True)
meta_data_csv = glob(udm_para_in_path + "/**/metadata.csv", recursive = True)
attractors = glob(udm_para_in_path + "/**/attractors.csv", recursive = True)
constraints = glob(udm_para_in_path + "/**/constraints.csv", recursive = True)

if len(meta_data_txt)==1:
    src = meta_data_txt[0]
    dst = os.path.join(udm_para_out_path,'metadata.txt')
    shutil.copy(src,dst)

if len(meta_data_csv)==1:
    src = meta_data_csv[0]
    dst = os.path.join(udm_para_out_path,'metadata.csv')
    shutil.copy(src,dst)

if len(attractors)==1:
    src = attractors[0]
    dst = os.path.join(udm_para_out_path,'attractors.csv')
    shutil.copy(src,dst)

if len(constraints)==1:
    src = constraints[0]
    dst = os.path.join(udm_para_out_path,'constraints.csv')
    shutil.copy(src,dst)
    

all_data.to_csv(
    os.path.join(outputs_path, '1km_data_' + location + '_' + ssp + '_'  + year + '_' + depth1 +'mm.csv'), index=False,  float_format='%g') 
