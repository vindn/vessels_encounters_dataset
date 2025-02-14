# %%
import pandas as pd
import numpy as np
import pickle
import geopandas as gpd
import movingpandas as mpd
import shapely as shp
import hvplot.pandas
import warnings
from geopandas import GeoDataFrame, read_file
from datetime import datetime, timedelta
from holoviews import opts
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
from shapely.geometry import Point
from shapely.geometry import LineString
import time
from distance_to_coast import CalcDistanceToCoast
from tqdm import tqdm
from geopy.distance import geodesic
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from pandas import Timestamp
from folium.plugins import MeasureControl
import socket
import rasterio
from rasterio.transform import from_origin
import warnings
import movingpandas as mpd
import shapely as shp
import hvplot.pandas
import time
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
from sklearn.cluster import HDBSCAN
import h3

# %%

# Rule of vessel type creation
# https://coast.noaa.gov/data/marinecadastre/ais/VesselTypeCodes2018.pdf
vessel_type_mapping = {
    0: "Not Available",
    1: "Other",
    2: "Other",
    3: "Other",
    4: "Other",
    5: "Other",
    6: "Other",
    7: "Other",
    8: "Other",
    9: "Other",
    10: "Other",
    20: "Other",
    21: "Tug Tow",
    22: "Tug Tow",
    23: "Other",
    24: "Other",
    25: "Other",
    26: "Other",
    27: "Other",
    28: "Other",
    29: "Other",
    30: "Fishing",
    31: "Tug Tow",
    32: "Tug Tow",
    33: "Other",
    34: "Other",
    35: "Military",
    36: "Pleasure Craft/Sailing",
    37: "Pleasure Craft/Sailing",
    38: "Other",
    39: "Other",
    40: "Other",
    41: "Other",
    42: "Other",
    43: "Other",
    44: "Other",
    45: "Other",
    46: "Other",
    47: "Other",
    48: "Other",
    49: "Other",
    50: "Other",
    51: "Other",
    52: "Tug Tow",
    53: "Other",
    54: "Other",
    55: "Other",
    56: "Other",
    57: "Other",
    58: "Other",
    59: "Other",
    60: "Passenger",
    61: "Passenger",
    62: "Passenger",
    63: "Passenger",
    64: "Passenger",
    65: "Passenger",
    66: "Passenger",
    67: "Passenger",
    68: "Passenger",
    69: "Passenger",
    70: "Cargo",
    71: "Cargo",
    72: "Cargo",
    73: "Cargo",
    74: "Cargo",
    75: "Cargo",
    76: "Cargo",
    77: "Cargo",
    78: "Cargo",
    79: "Cargo",
    80: "Tanker",
    81: "Tanker",
    82: "Tanker",
    83: "Tanker",
    84: "Tanker",
    85: "Tanker",
    86: "Tanker",
    87: "Tanker",
    88: "Tanker",
    89: "Tanker",
    90: "Other",
    91: "Other",
    92: "Other",
    93: "Other",
    94: "Other",
    95: "Other",
    96: "Other",
    97: "Other",
    98: "Other",
    99: "Other",
    1001: "Fishing",
    1002: "Fishing",
    1003: "Cargo",
    1004: "Cargo",
    1005: "Other",
    1006: "Other",
    1007: "Other",
    1008: "Other",
    1009: "Other",
    1010: "Other",
    1011: "Other",
    1012: "Passenger",
    1013: "Passenger",
    1014: "Passenger",
    1015: "Passenger",
    1016: "Cargo",
    1017: "Tanker",
    1018: "Other",
    1019: "Pleasure Craft/Sailing",
    1020: "Other",
    1021: "Military",
    1022: "Other",
    1023: "Tug Tow",
    1024: "Tanker",
    1025: "Tug Tow",
}

vessel_type_mapping_target = {
    75: 'Fishing boat',
    751: 'Trawler',
    752: 'Cutter',
    30: 'Fishing',
    1010: 'Offshore',
    52: 'Bulk carrier',
    82: 'Sailing boat with auxiliary motor'

}

# Dictionary mapping STATUS codes to descriptions
status_mapping = {
    0: "Underway using engine",
    1: "At anchor",
    2: "Not under command",
    3: "Restricted maneuverability",
    4: "Constrained by her draught",
    5: "Moored",
    6: "Aground",
    7: "Engaged in fishing",
    8: "Underway sailing",
    9: "Reserved for future use",
    10: "Reserved for future use",
    11: "Reserved for future use",
    12: "Reserved for future use",
    13: "Reserved for future use",
    14: "AIS-SART (Search and Rescue Transmitter)",
    15: "Undefined (default)"
}

def msg_telegram(message):
    ip_address = "127.0.0.1"    
    try:
        # Criar um socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Conectar ao servidor
        s.connect((ip_address, 12345))
        
        # Enviar a mensagem
        s.sendall(message.encode())
        
        # Fechar o socket
        s.close()
        print(f"Message sent to {ip_address}:12345")
    except Exception as e:
        print(f"An error occurred: {e}")

class Encounter:
    def __init__(self, encounter_count_id=0):
        self.data_set_folder = "/home/vindn/SynologyDrive/4_UFRJ/artigo_escrita/Rendezvous Project/2 - Datasets/2015/"

        # self.data_set_folder = self.data_set_folder
        # self.url = self.data_set_folder + "AIS_2022_01_01.csv"
        self.filename = "AIS_2015_01_01.csv"
        self.url = self.data_set_folder + self.filename
        self.chunksize = 10 ** 6
        self.chunks = []
        # large data
        self.gdf = None
        self.vessel_trajectories = None
        self.dc = CalcDistanceToCoast( )
        self.raster = None
        self.raster_data = None
        self.transform = None
        self.init_raster_distances( )
        self.encounters_count = encounter_count_id

    def setAISfile( self, filename ):
        self.filename = filename
        self.url = self.data_set_folder + self.filename
            
    def init_raster_distances( self ):
        # Distances in kilometers!!!
        raster_file = self.data_set_folder + 'distance-from-shore.tif'
        # 1. Carregar o raster na memória
        with rasterio.open(raster_file) as src:
            self.raster_data = src.read(1)  # Ler o primeiro band do raster
            self.transform = src.transform

    # 2. Função para converter coordenadas geográficas para coordenadas de linha e coluna
    def geo_to_rowcol(self, x, y ):
        col, row = ~self.transform * (x, y)
        return int(row), int(col)

    def get_distance_from_coast_km(self, p):
        # Distances in kilometers!!!
        row, col = self.geo_to_rowcol(p.x, p.y)

        value = self.raster_data[row, col]
        return value

    # def apply_distance_to_coast( self, gdf ):
    #     # gdf['distance_to_coast'] = gdf['geometry'].apply(self.get_distance_from_coast_km)
    #     # achar um raster com as distancias dos pixels para a costa
        
    #     dist_nm = []
    #     for i in tqdm( range(len( gdf )) ):
    #         dist_nm.append( self.dc.distance_to_brazil_coast( gdf.iloc[i].LON, gdf.iloc[i].LAT ) )
        
    #     gdf["distance_to_coast"] = dist_nm

    #     return gdf      


    # def apply_distance_to_coast_parallel(self, gdf):
    #     num_cores = cpu_count()
    #     print(f"Using {num_cores} cores for parallel processing.")

    #     coords = list(zip(gdf['LON'], gdf['LAT']))

    #     dist_nm = Parallel(n_jobs=num_cores)(
    #         delayed(self.dc.distance_to_brazil_coast)(lon, lat) 
    #         for lon, lat in tqdm(coords)
    #     )

    #     gdf["distance_to_coast"] = dist_nm
    #     return gdf

    
    def apply_distance_from_coast_km( self, gdf ):
        gdf['distance_to_coast'] = gdf['geometry'].apply(self.get_distance_from_coast_km)
        return gdf

    def get_gdf( self ):
        return self.gdf
    
    def get_coastline( self ):
        return self.coast
    

    def read_ais_csv(self):
        for chunk in pd.read_csv(self.url, chunksize=self.chunksize):
            self.chunks.append(chunk)

        data = pd.concat(self.chunks, axis=0)

        # print(data.head())
        data.dropna()

        return data

    def write_pickle_obj(self, data, file_name):
        with open(self.data_set_folder + file_name, 'wb') as data_file:
            pickle.dump(data, data_file)

    def read_pickle_obj(self, file_name):
        try:
            with open(self.data_set_folder + file_name, 'rb') as data_file:
                data = pickle.load(data_file)
                return data
        except Exception as e:
            print(e, "File not Found!")

    def load_gdf(self, df):        
        warnings.filterwarnings('ignore')

        opts.defaults(opts.Overlay(
            active_tools=['wheel_zoom'], frame_width=500, frame_height=400))

        gdf = gpd.GeoDataFrame(
            df.set_index('BaseDateTime'), geometry=gpd.points_from_xy(df.LON, df.LAT))

        gdf.set_crs('epsg:4326')
        return gdf
    
    # Trajectories
    def create_trajectory(self, gdf, verbose=True, gap_minutes=30):

        # reset index
        gdf = gdf.reset_index()
        gdf['BaseDateTime'] = pd.to_datetime(gdf['BaseDateTime'], utc=True)
        gdf['traj_id'] = gdf['MMSI']
        # limit to avoid slow
    #     gdf = gdf[:10000]

        # create trajectories

        start_time = time.time()

        # Specify minimum length for a trajectory (in meters)
        minimum_length = 0
        # collection = mpd.TrajectoryCollection(gdf, "imo",  t='timestamp', min_length=0.001)
        collection = mpd.TrajectoryCollection(
            gdf, "traj_id",  t='BaseDateTime', min_length=0.001, crs='epsg:4326')
        # collection.add_direction(gdf.COG)
        # collection.add_speed(gdf.SOG, units='nm')
        # collection.add_traj_id(overwrite=True)

        # set time gap between trajectories for split
        # collection = mpd.ObservationGapSplitter(
        #     collection).split(gap=timedelta(minutes=90))
        collection = mpd.ObservationGapSplitter(
            collection).split(gap=timedelta(minutes=gap_minutes))
        
        collection.add_speed(overwrite=True, name="speed_nm", units=('nm', 'h'))
        collection.add_direction(overwrite=True, name='direction')
        collection.add_timedelta(overwrite=True, name='time_delta')
        collection.add_angular_difference(overwrite=True, name='ang_diff')
        collection.add_distance(overwrite=True, name="dist_diff", units="km")     

        end_time = time.time()
        if verbose:
            print("Time creation trajectories: ", (end_time-start_time)/60,  " min")

        return collection
    
    def built_trajectories_from_scratch( self, aisfilename ):
        self.setAISfile( aisfilename )
        df_ais = self.read_ais_csv( )[:]
        self.gdf = self.load_gdf( df_ais )        
        # self.gdf = self.filter_gdf( self.gdf )
        # self.vessel_trajectories = self.create_trajectory( self.gdf )
        # self.write_pickle_obj( self.vessel_trajectories, "vessel_trajectories.obj" )
        print("Creating H3 cells...")
        self.gdf = self.apply_h3_cells_in_gdf( self.gdf, 11 )
        print("Creating distances points from coast...")
        # self.gdf = self.apply_distance_to_coast( self.gdf )
        # self.gdf = self.apply_distance_to_coast_parallel( self.gdf )
        self.gdf = self.apply_distance_from_coast_km( self.gdf )
        self.write_gdf_to_file( self.gdf )
        # msg_telegram("Finished built trajectories from scratch!")

        return self.gdf, self.vessel_trajectories
    
    def built_trajectories_from_files( self ):
        self.gdf = self.read_gdf_from_file()
        self.gdf['VesselType'] = self.gdf['VesselType'].fillna(0)
        self.vessel_trajectories = self.read_trajectories_from_file()
        self.insert_columns_trajectories( self.vessel_trajectories )
        # self.gdf = self.filter_gdf( self.gdf )

    # def build_gdf_from_batch_files_from_scratch( self ):
    #     import gc
    #     prefix_filename = "AIS_2015_01_"
    #     n_files = 2
    #     for i in range(1, n_files+1):
    #         file_path = prefix_filename + "{:02}".format(i) + ".csv"
    #         self.setAISfile( file_path )
    #         self.read_ais_csv( )

    #     df_ais = pd.concat(self.chunks, axis=0)
    #     self.chunks = None
    #     gc.collect()
    #     print("df lines: ", len(df_ais) )
    #     self.gdf = self.load_gdf( df_ais )
    #     df_ais = None
    #     gc.collect()
    #     self.gdf['VesselType'] = self.gdf['VesselType'].fillna(0)

    #     self.gdf = self.apply_h3_cells_in_gdf( self.gdf, 9 )
    #     print("Creating distances points from coast...")
    #     self.gdf = self.apply_distance_from_coast_nm( self.gdf )
    #     self.write_gdf_to_file( self.gdf )
    #     msg_telegram("Finished built trajectories from scratch!")

    #     return self.gdf
        

    def read_trajectories_from_file(  self ):
        data = self.read_pickle_obj( "vessel_trajectories.obj" )
        return data

    def write_trajectories_to_file( self, vessel_trajectories ):
        self.write_pickle_obj( vessel_trajectories, "vessel_trajectories.obj" )

    def write_gdf_to_file( self, gdf ):
        self.write_pickle_obj( gdf, "gdf.obj" )

    def read_gdf_from_file( self ):
        data = self.read_pickle_obj( "gdf.obj" )
        return data    
    
    # Insert data columns in trajectories
    def insert_columns_trajectories( self, collection ):
        
        for traj in collection.trajectories:
            traj.df['ais_interval'] = traj.df.index.to_series().diff()
            traj.df['ais_interval'] = traj.df['ais_interval'].dt.total_seconds() / 60
            traj.df['ais_interval'] = traj.df['ais_interval'].abs()

            traj.df['n_points'] = len( traj.df )

    def filter_gdf( self, gdf ):
        data = gdf[ gdf['SOG'] > 1 ]
        return data
    
    def clustering_trajectories( self, trajs ):

        kms_per_radian = 6371.0088
        epsilon = 0.1 / kms_per_radian

        n_trajs = len( trajs )
        distances = np.zeros(( n_trajs, n_trajs ))
        for i in range(n_trajs):
            for j in range(n_trajs):
                d = trajs[i].to_linestring().distance(
                    trajs[j].to_linestring() )
                distances[i, j] = d

        # print(distances)    

        db = DBSCAN(eps=epsilon, min_samples=1, metric='precomputed').fit(distances)
        cluster_labels = db.labels_
        num_clusters = len(set(cluster_labels))
        clusters = pd.Series([distances[cluster_labels == n] for n in range(num_clusters)])
        print(f'Number of clusters: {num_clusters}')

        for traj, i in zip( trajs, cluster_labels ):
            traj.df["cluster"] = i


    def clustering_points( self, gdf ):

        matrix = gdf[ ['LAT','LON'] ].values
        kms_per_radian = 6371.0088
        epsilon = 0.1 / kms_per_radian

        db = HDBSCAN(cluster_selection_epsilon=epsilon, min_samples=1, algorithm='balltree', metric='haversine', n_jobs=-1).fit(np.radians(matrix))
        cluster_labels = db.labels_
        num_clusters = len(set(cluster_labels))
        clusters = pd.Series([matrix[cluster_labels == n] for n in range(num_clusters)])
        print(f'Number of clusters: {num_clusters}')

        gdf["cluster"] = cluster_labels

        return gdf    
    
    def distance_points_meters( self, p1, p2 ):
        point1_series = gpd.GeoSeries([p1], crs="EPSG:4326")
        point1_meters = point1_series.to_crs(epsg=32619)

        point2_series = gpd.GeoSeries([p2], crs="EPSG:4326")
        point2_meters = point2_series.to_crs(epsg=32619)

        return point1_meters.distance(point2_meters).min()

    
    def distance_between_ships_inside_cluster_meters( self, cluster ):
        count = cluster.groupby(['MMSI']).size()
        count = count.sort_values(ascending=False)

        ship1 = cluster[ cluster["MMSI"] == count.index[0] ]
        ship2 = cluster[ cluster["MMSI"] == count.index[1] ]

        return self.distance_points_meters( ship1.iloc[0].geometry, ship2.iloc[0].geometry )

   
    # Resolução	    Raio (km)
    # 0	    1279.0
    # 1	    483.4
    # 2	    183.0
    # 3	    69.09
    # 4	    26.10
    # 5	    9.87
    # 6	    3.73
    # 7	    1.41
    # 8	    0.53
    # 9 	0.20
    # 10	0.076
    # 11	0.0287
    # 12	0.0109
    # 13	0.00411
    # 14	0.00155
    # 15	0.000587
    def apply_h3_cells_in_gdf( self, gdf, resolution=11 ):
        import h3

        # Converta os pontos do gdf para índices H3
        # gdf['h3_index'] = gdf.apply(lambda row: h3.geo_to_h3(row['geometry'].y, row['geometry'].x, resolution), axis=1)
        gdf['h3_index'] = gdf.apply(lambda row: h3.latlng_to_cell(row['geometry'].y, row['geometry'].x, resolution), axis=1)
        return gdf
    
    # def get_h3_cells_with_multiple_points( self, gdf, min_distance_from_coast=10 ):
    #     import h3

    #     # get clusters indexes with only two different MMSIs in cluster
    #     couting = gdf.groupby('h3_index')['MMSI'].nunique()
    #     indexes = couting.where(couting == 2).dropna().index

    #     encounters = []
    #     for i in indexes:
    #         # get cluster by index
    #         g_cluster = gdf[ gdf["h3_index"] == i ]

    #         count = g_cluster.groupby(['MMSI']).size()
    #         count = count.sort_values(ascending=False)

    #         # ship trajectories
    #         ship1 = self.gdf[ self.gdf["MMSI"] == count.index[0] ]
    #         ship2 = self.gdf[ self.gdf["MMSI"] == count.index[1] ]

    #         encounters.append( [g_cluster, ship1, ship2 ] )                

    #     return encounters

    # def get_trajectory_for_train(self, gdf, ts, mmsi):
    #     # Filter by MMSI
    #     # filtered_gdf = gdf[gdf['MMSI'] == mmsi].reset_index(drop=True)
    #     ts = pd.Timestamp(ts)
    #     filtered_gdf = gdf[gdf['MMSI'] == mmsi]
    #     # print("gdf_size = " + str(len(filtered_gdf)) + "ts = " + str(ts) )
        
    #     # # Find the index at the specific timestamp
    #     # timestamp_matches = filtered_gdf.index == ts
    #     # if not timestamp_matches.any():
    #     #     raise ValueError(f"Timestamp {ts} not found for MMSI {mmsi}.")
    #     index_at_timestamp = filtered_gdf.index.get_loc(ts)
    #     # index_at_timestamp = index_at_timestamp.nonzero()[0]

    #     number_positions = 10
    #     total_positions = len(filtered_gdf)

    #     # Calculate start and end indices, ensuring they are within bounds
    #     start_index = max(0, index_at_timestamp - number_positions)
    #     end_index = min(total_positions, index_at_timestamp + number_positions + 1)        

    #     # Get the subset of data
    #     result = filtered_gdf.iloc[start_index:end_index]

    #     return result


    def get_trajectory_for_train(self, gdf, ts, mmsi):
        import pandas as pd
        import numpy as np

        # Filter by MMSI
        filtered_gdf = gdf[gdf['MMSI'] == mmsi]

        # Check if 'BaseDateTime' is the index
        if filtered_gdf.index.name != 'BaseDateTime':
            # If not, set it as the index
            if 'BaseDateTime' in filtered_gdf.columns:
                filtered_gdf = filtered_gdf.set_index('BaseDateTime')
            else:
                raise KeyError("The 'BaseDateTime' column is missing in filtered_gdf.")

        # Ensure the index is sorted
        filtered_gdf = filtered_gdf.sort_index()

        # Convert ts to pd.Timestamp if necessary
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)

        # Proceed with finding the index position
        try:
            index_at_timestamp = filtered_gdf.index.get_loc(ts)
        except KeyError:
            print(f"Timestamp {ts} not found for MMSI {mmsi}. Skipping.")
            return None

        # Handle different types of index_at_timestamp
        if isinstance(index_at_timestamp, int):
            indices = [index_at_timestamp]
        elif isinstance(index_at_timestamp, slice):
            indices = list(range(index_at_timestamp.start, index_at_timestamp.stop))
        elif isinstance(index_at_timestamp, np.ndarray):
            indices = index_at_timestamp.tolist()
        else:
            raise ValueError(f"Unexpected type for index_at_timestamp: {type(index_at_timestamp)}")

        # Use the first occurrence
        index_at_timestamp = indices[0]

        number_positions = 20
        # number_positions = 10
        total_positions = len(filtered_gdf)

        # Calculate start and end indices
        start_index = max(0, index_at_timestamp - number_positions)
        end_index = min(total_positions, index_at_timestamp + number_positions + 1)

        # Get the subset of data
        result = filtered_gdf.iloc[start_index:end_index]

        return result


    def get_h3_cells_with_multiple_points( self, gdf ):
        import h3

        # get clusters indexes with only two different MMSIs in cluster
        couting = gdf.groupby('h3_index')['MMSI'].nunique()
        indexes = couting.where(couting == 2).dropna().index

        encounters = []
        for i in indexes:
            # get cluster by index
            g_cluster = gdf[ gdf["h3_index"] == i ]
            # group by mmsi
            count = g_cluster.groupby(['MMSI']).size()
            # sort by number of rows
            count = count.sort_values(ascending=False)
            # get all rows by both mmsis
            gdf_filtered = self.get_gdf()[ 
                (self.get_gdf()["MMSI"] == count.index[0]) |
                (self.get_gdf()["MMSI"] == count.index[1])
                ]
            # gdf_filtered = gdf_filtered.sort_index()
            gdf_filtered.index = pd.to_datetime( gdf_filtered.index )
            # get last time inside cluster
            ship1OldestTime = g_cluster[ g_cluster["MMSI"] == count.index[0] ].index.max()
            ship2OldestTime = g_cluster[ g_cluster["MMSI"] == count.index[1] ].index.max()
            # get 10 positions before and after select ts
            # print("DEBUG## MMSI1: "+ str(count.index[0]) + 
            #       "/ MMSI2: " + str(count.index[1]) + " / ship1OldestTime " + str(ship1OldestTime) )
            ship1 = self.get_trajectory_for_train( gdf_filtered, ship1OldestTime, count.index[0] )
            ship2 = self.get_trajectory_for_train( gdf_filtered, ship2OldestTime, count.index[1] )

            # ship1 = ship1OldestTime
            # ship2 = ship2OldestTime
            # if len(ship1) == 21 and len(ship2) == 21 :
            if len(ship1) == 41 and len(ship2) == 41 :
                ship1['encounter_id'] = self.encounters_count
                ship2['encounter_id'] = self.encounters_count
                encounters.append( [g_cluster, ship1, ship2, self.encounters_count ] )                
                self.encounters_count += 1

        return encounters


    # remove df that is not the target in clusters    
    def filter_targets( self, encounters ):
        enc_target = []
        for index, value in enumerate(encounters):
            filtered_gdf = value[0][  value[0]['VesselType'].isin(vessel_type_mapping_target.keys())  ]
            if not filtered_gdf.empty:
                enc_target.append( filtered_gdf )
        return enc_target

    def detect_encounters_h3( self, gdf, time_interval_m=240, min_distance_from_coast=10, min_distance_between_ships_km=10 ):
        import datetime
        import traceback

        gdf.index = pd.to_datetime(gdf.index)
        min_time = gdf.index.min()
        max_time = gdf.index.max()
        start_time = min_time
        end_time = min_time + datetime.timedelta(minutes=time_interval_m)
        if end_time > max_time:
            end_time = max_time
        encounters = []

        while start_time < max_time:       
            # gdf_filtered = gdf[(gdf.index >= start_time) & (gdf.index <= end_time)]
            gdf_filtered = gdf[(gdf.index >= start_time) & (gdf.index <= end_time) & (gdf.SOG <= 1.0 ) ]
            print('gdf size: ', str(len( gdf_filtered )) )
            try:
                print( "start_time=", start_time, " end_time=", end_time )
                # do filter points longer than 10km from coast
                gdf_filtered = gdf_filtered[ gdf_filtered[ "distance_to_coast" ] > min_distance_from_coast ]
                # cluster points
                print("Clustering points...")
                tmp_encounters = self.get_h3_cells_with_multiple_points( gdf_filtered )
                encounters += tmp_encounters
            except Exception as e: 
                print("Error! Size cluster: ", len( gdf_filtered ) )
                traceback.print_exc()

            start_time += datetime.timedelta(minutes=time_interval_m)
            end_time += datetime.timedelta(minutes=time_interval_m)
            if end_time > max_time:
                end_time = max_time

        return encounters

    def mmsi_to_color(self, mmsi):
        import folium
        import hashlib
        # Converta MMSI para string e obtenha seu hash
        mmsi_hash = hashlib.md5(str(mmsi).encode()).hexdigest()
        # Use os primeiros 6 caracteres do hash como código de cor hexadecimal
        color = '#' + mmsi_hash[:6]
        return color


    def calculate_initial_compass_bearing(self, coord1, coord2):
        import geopandas as gpd
        import math

        """
        Calcula o rumo entre dois pontos.
        A fórmula é baseada em: http://mathforum.org/library/drmath/view/55417.html
        Retorna o rumo como um valor entre 0 e 360
        """
        lat1 = math.radians(coord1[0])
        lat2 = math.radians(coord2[0])

        diffLong = math.radians(coord2[1] - coord1[1])

        x = math.sin(diffLong) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))

        initial_bearing = math.atan2(x, y)

        # Convertendo de radianos para graus
        initial_bearing = math.degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) % 360

        return compass_bearing

    def calc_angle_between_points( self, f_gdf ):

        angles = []
        for idx, (i, row) in enumerate(f_gdf.iterrows()):
            if idx < len(f_gdf) - 1:  # Se não for o último ponto
                current_point = row['geometry']
                next_point = f_gdf.iloc[idx+1]['geometry']
                angle = self.calculate_initial_compass_bearing((current_point.y, current_point.x), (next_point.y, next_point.x))
                angles.append(angle)
            else:
                angles.append( angles[-1] )  # para o último ponto, não há "próximo ponto"

        f_gdf['angle_to_next'] = angles
        return f_gdf

    # TODO improve triangle angle, its wrong
    def plot_arrow_gdf( self, gdf ):
        import folium

        latitude_initial = gdf.iloc[0]['LAT']
        longitude_initial = gdf.iloc[0]['LON']
        
        gdf = self.calc_angle_between_points( gdf )

        m = folium.Map(location=[latitude_initial, longitude_initial], zoom_start=10)

        for _, row in gdf.iterrows():

            vessel_description = vessel_type_mapping.get( int( row['VesselType'] ), "Unknown")

            # Concatenar colunas para o popup
            popup_content = f"<b>Timestamp:</b> {row.name}<br><b>VesselName:</b> {row['VesselName']}<br><b>MMSI</b>: {row['MMSI']}<br><b>LAT:</b> {row['LAT']}<br><b>LON:</b> {row['LON']}<br><b>SOG:</b> {row['SOG']}<br><b>Type:</b> {vessel_description}<br><b>COG:</b> {row['COG']}<br><b>Heading:</b> {row['Heading']}"
            color = self.mmsi_to_color( row['MMSI'] )
            
            # folium.Marker(
            #     location=[row['geometry'].y, row['geometry'].x],
            #     popup=popup_content
            # ).add_to(m)

            folium.RegularPolygonMarker(
                location=[row['geometry'].y, row['geometry'].x],
                popup=popup_content,
                radius=3,  # Define o tamanho do ponto
                color=color,  # Define a cor do ponto
                fill=False,
                fill_color=color,
                fill_opacity=1,
                number_of_sides=3, 
                rotation=row['angle_to_next']
            ).add_to(m)            

        return m    

    def create_linestring(self, group):        
        # Ordenar por timestamp
        group = group.sort_values(by='BaseDateTime')      
        # Se há mais de um ponto no grupo, crie uma LineString, caso contrário, retorne None
        return LineString(group.geometry.tolist()) if len(group) > 1 else None

   
    def report_data( self, combined_gdf ):
        combined_gdf['vessel_description'] = combined_gdf['VesselType'].apply( vessel_type_mapping.get )
        combined_gdf['vessel_description'] = combined_gdf['vessel_description'].fillna('Unknow')

        h3_groups_per_description = combined_gdf.groupby('vessel_description')['h3_index'].nunique().reset_index(name='number_of_h3_groups')

        # Sort the dataframe by the number of groups for better visualization
        h3_groups_per_description = h3_groups_per_description.sort_values(by='number_of_h3_groups', ascending=False)

        return h3_groups_per_description
    
    def heat_map_gdf( self, gdf ):
        import folium
        from folium.plugins import HeatMap

        # Converta as colunas de latitude e longitude em uma lista de coordenadas
        data = gdf[['LAT', 'LON']].values.tolist()

        # Crie uma instância básica do mapa usando a média das latitudes e longitudes para centralizar
        m = folium.Map(location=[gdf['LAT'].mean(), gdf['LON'].mean()], zoom_start=10)

        # Adicione os dados ao mapa usando o plugin HeatMap
        HeatMap(data).add_to(m)
        m.save("heatmap.html")
        return m
    
    def heat_map( self, list_encounters ):
        return self.heat_map_gdf( self.get_concat_gdf( list_encounters ) )

    def get_concat_gdf( self, list_encounters, idx=0 ):
        flat_encounter = []
        for i in range(len(list_encounters)):
            for j in range(len(list_encounters[i])):
                gdf_encounter = list_encounters[i][j][idx]
                # gdf_encounter['encounter_id'] = i*len(list_encounters[i]) + j
                flat_encounter.append( gdf_encounter )

        return pd.concat( flat_encounter, ignore_index=False)    
    
    
    def get_not_encounters( self, gdf, list_encounters, distance_from_coast ):

        concat_gdf = self.get_concat_gdf(list_encounters)
        gdf_not_encounter = gdf[ gdf[ "distance_to_coast" ] > distance_from_coast ]
        gdf_not_encounter = gdf_not_encounter[ ~gdf_not_encounter.index.isin( concat_gdf.index ) ]

        trajs_no_encounter = encounter.create_trajectory( gdf_not_encounter )
        # get points by trajectory
        size_trajectory = 21
        # size_trajectory = 41
        t = []
        for traj in trajs_no_encounter:
            if traj.df['MMSI'].count() >= size_trajectory: 
                t.append( traj.df[-size_trajectory:])

        # return trajectories that's haven't encounter in this gdf
        return t

    # plot gdf points
    def plot_gdf( self, gdf, vessel_description, m=None, color='blue' ):
        import folium

        latitude_initial = gdf.iloc[0]['LAT']
        longitude_initial = gdf.iloc[0]['LON']

        if not m:
            m = folium.Map(location=[latitude_initial, longitude_initial], zoom_start=10)

        for _, row in gdf.reset_index().iterrows():

            # vessel_description = vessel_type_mapping.get( int( row['VesselType'] ), "Unknown")
            # vessel_description = vessel_type_mapping.get( int( row['VesselType'] ), "Unknown")

            # Concatenar colunas para o popup
            popup_content = f"<b>Timestamp:</b> {row['BaseDateTime']}<br><b>VesselName:</b> {row['VesselName']}<br><b>MMSI</b>: {row['MMSI']}<br><b>LAT:</b> {row['LAT']}<br><b>LON:</b> {row['LON']}<br><b>SOG:</b> {row['SOG']}<br><b>Type:</b> {row['VesselType']}<br><b>COG:</b> {row['COG']}"
            # color = mmsi_to_color( row['MMSI'] )
            
            folium.CircleMarker(
                location=[row['geometry'].y, row['geometry'].x],
                popup=popup_content,
                radius=1,  # Define o tamanho do ponto
                color=color,  # Define a cor do ponto
                fill=True,
                fill_opacity=1,
            ).add_to(m)            

        return m
       
    # plot trajectories from points
    def plot_trajectory( self, gdf, vessel_description, m=None, color='blue' ):
        import folium

        lines = gdf.groupby('MMSI').apply(self.create_linestring)

        # Remove possíveis None (se algum grupo tiver apenas um ponto)
        lines = lines.dropna()

        # Crie um novo GeoDataFrame com as LineStrings
        lines_gdf = gpd.GeoDataFrame(lines, columns=['geometry'], geometry='geometry')

        lines_gdf.reset_index(inplace=True)

        # start_point = Point(lines_gdf.iloc[0].geometry.coords[0])
        # m = folium.Map(location=[start_point.y, start_point.x], zoom_start=10)

        if not m:
            m = self.plot_gdf( gdf, vessel_description, color=color )
        else:
            self.plot_gdf( gdf, vessel_description, m, color=color )

        for _, row in lines_gdf.iterrows():            
            if row['geometry'].geom_type == 'LineString':
                popup_content = f"{row['MMSI']}"
                coords = list(row['geometry'].coords)
                    
                folium.PolyLine(locations=[(lat, lon) for lon, lat in coords], 
                            popup=popup_content,
                            weight=0.5,
                            color=color
                ).add_to(m)

        return m


    def plot_encounter( self, gdf1, gdf2, m=None ):
        if m is None:
            id_encounter = gdf1['encounter_id'].unique()[0]
            m = self.plot_trajectory( gdf1, "vessel 1 enc: " + str(id_encounter), m=None, color='green' )
            m = self.plot_trajectory( gdf2, "vessel 2 enc: " + str(id_encounter), m, color='red' )
            m.add_child(MeasureControl())
        else:
            id_encounter = gdf1['encounter_id'].unique()[0]
            m = self.plot_trajectory( gdf1, "vessel 1 enc: " + str(id_encounter), m=m, color='green' )
            m = self.plot_trajectory( gdf2, "vessel 2 enc: " + str(id_encounter), m, color='red' )
          

        return m
   
    def get_encounter_by_mmsi(self, mmsi ):
        mmsi_encounters = []
        for i in range(len(self.df_clusters)):
            if mmsi in self.df_clusters[i]['MMSI']:
                mmsi_encounters.append(  self.df_clusters[i] )

        return mmsi_encounters

# %%
#######
## MAIN
########

# %%
## Execution for many AIS files
import gc
import traceback

encounter_count_id = 0
number_ais_messages_processed = 0
folder_path = "/home/vindn/SynologyDrive/4_UFRJ/artigo_escrita/Rendezvous Project/2 - Datasets/2015/"

prefix_filename = "dados_ais_janeiro/AIS_2015_01_"
n_files = 31
# n_files = 1
list_encounters = []
list_not_encouters = []
erro=False
for i in tqdm( range(1, n_files+1) ):
    filename = prefix_filename + "{:02}".format(i) + ".csv"
    encounter = Encounter(encounter_count_id=encounter_count_id)
    encounter.built_trajectories_from_scratch( filename )

    try: 
        gdf_filtered = encounter.get_gdf( )[:]
        number_ais_messages_processed += len(gdf_filtered)
        # 18km ~ 10 MN
        list_encounters.append( encounter.detect_encounters_h3(gdf_filtered, time_interval_m=120, min_distance_from_coast=18, min_distance_between_ships_km=1) )
        print("SUSPECTED CLUSTERS: ", len(list_encounters))
        list_not_encouters.append( encounter.get_not_encounters( gdf_filtered, list_encounters, 18) )
        encounter_count_id = encounter.encounters_count

        # msg_telegram( "Execução dos arquivos AIS finalizada com sucesso!" )
    except Exception as e   :
        print("Ocorreu um erro!")
        traceback.print_exc()
        msg_telegram( "Execução finalizada com erro!" )
        erro=True
        break  

if not erro:
    msg_telegram( "Execução dos arquivos AIS Janeiro finalizada com sucesso!" )

# %%

prefix_filename = "dados_ais_fevereiro/AIS_2015_02_"
n_files = 28

erro=False
for i in tqdm( range(1, n_files+1) ):
    filename = prefix_filename + "{:02}".format(i) + ".csv"
    encounter = Encounter(encounter_count_id=encounter_count_id)
    encounter.built_trajectories_from_scratch( filename )

    try: 
        gdf_filtered = encounter.get_gdf( )[:]
        number_ais_messages_processed += len(gdf_filtered)
        # 18km ~ 10 MN
        list_encounters.append( encounter.detect_encounters_h3(gdf_filtered, time_interval_m=120, min_distance_from_coast=18, min_distance_between_ships_km=1) )
        print("SUSPECTED CLUSTERS: ", len(list_encounters))
        list_not_encouters.append( encounter.get_not_encounters( gdf_filtered, list_encounters, 18) )
        encounter_count_id = encounter.encounters_count

        # msg_telegram( "Execução dos arquivos AIS finalizada com sucesso!" )
    except Exception as e   :
        print("Ocorreu um erro!")
        traceback.print_exc()
        msg_telegram( "Execução finalizada com erro!" )
        erro=True
        break

if not erro:
    msg_telegram( "Execução dos arquivos AIS Fevereiro finalizada com sucesso!" )

# %%

prefix_filename = "dados_ais_marco/AIS_2015_03_"
n_files = 31

erro=False
for i in tqdm( range(1, n_files+1) ):
    filename = prefix_filename + "{:02}".format(i) + ".csv"
    encounter = Encounter(encounter_count_id=encounter_count_id)
    encounter.built_trajectories_from_scratch( filename )

    try: 
        gdf_filtered = encounter.get_gdf( )[:]
        number_ais_messages_processed += len(gdf_filtered)
        # 18km ~ 10 MN
        list_encounters.append( encounter.detect_encounters_h3(gdf_filtered, time_interval_m=120, min_distance_from_coast=18, min_distance_between_ships_km=1) )
        print("SUSPECTED CLUSTERS: ", len(list_encounters))
        list_not_encouters.append( encounter.get_not_encounters( gdf_filtered, list_encounters, 18) )
        encounter_count_id = encounter.encounters_count

        # msg_telegram( "Execução dos arquivos AIS finalizada com sucesso!" )
    except Exception as e   :
        print("Ocorreu um erro!")
        traceback.print_exc()
        msg_telegram( "Execução finalizada com erro!" )
        erro=True
        break

if not erro:
    msg_telegram( "Execução dos arquivos AIS Março finalizada com sucesso!" )

# %%

prefix_filename = "dados_ais_abril/AIS_2015_04_"
n_files = 30

erro=False
for i in tqdm( range(1, n_files+1) ):
    filename = prefix_filename + "{:02}".format(i) + ".csv"
    encounter = Encounter(encounter_count_id=encounter_count_id)
    encounter.built_trajectories_from_scratch( filename )

    try: 
        gdf_filtered = encounter.get_gdf( )[:]
        number_ais_messages_processed += len(gdf_filtered)
        # 18km ~ 10 MN
        list_encounters.append( encounter.detect_encounters_h3(gdf_filtered, time_interval_m=120, min_distance_from_coast=18, min_distance_between_ships_km=1) )
        print("SUSPECTED CLUSTERS: ", len(list_encounters))
        list_not_encouters.append( encounter.get_not_encounters( gdf_filtered, list_encounters, 18) )
        encounter_count_id = encounter.encounters_count

        # msg_telegram( "Execução dos arquivos AIS finalizada com sucesso!" )
    except Exception as e   :
        print("Ocorreu um erro!")
        traceback.print_exc()
        msg_telegram( "Execução finalizada com erro!" )
        erro=True
        break

if not erro:
    msg_telegram( "Execução dos arquivos AIS Abril finalizada com sucesso!" )

# %%

prefix_filename = "dados_ais_maio/AIS_2015_05_"
n_files = 30

erro=False
for i in tqdm( range(1, n_files+1) ):
    filename = prefix_filename + "{:02}".format(i) + ".csv"
    encounter = Encounter(encounter_count_id=encounter_count_id)
    encounter.built_trajectories_from_scratch( filename )

    try: 
        gdf_filtered = encounter.get_gdf( )[:]
        number_ais_messages_processed += len(gdf_filtered)
        # 18km ~ 10 MN
        list_encounters.append( encounter.detect_encounters_h3(gdf_filtered, time_interval_m=120, min_distance_from_coast=18, min_distance_between_ships_km=1) )
        print("SUSPECTED CLUSTERS: ", len(list_encounters))
        list_not_encouters.append( encounter.get_not_encounters( gdf_filtered, list_encounters, 18) )
        encounter_count_id = encounter.encounters_count

        # msg_telegram( "Execução dos arquivos AIS finalizada com sucesso!" )
    except Exception as e   :
        print("Ocorreu um erro!")
        traceback.print_exc()
        msg_telegram( "Execução finalizada com erro!" )
        erro=True
        break

if not erro:
    msg_telegram( "Execução dos arquivos AIS Maio finalizada com sucesso!" )


# %%

prefix_filename = "dados_ais_junho/AIS_2015_06_"
n_files = 30

erro=False
for i in tqdm( range(1, n_files+1) ):
    filename = prefix_filename + "{:02}".format(i) + ".csv"
    encounter = Encounter(encounter_count_id=encounter_count_id)
    encounter.built_trajectories_from_scratch( filename )

    try: 
        gdf_filtered = encounter.get_gdf( )[:]
        number_ais_messages_processed += len(gdf_filtered)
        # 18km ~ 10 MN
        list_encounters.append( encounter.detect_encounters_h3(gdf_filtered, time_interval_m=120, min_distance_from_coast=18, min_distance_between_ships_km=1) )
        print("SUSPECTED CLUSTERS: ", len(list_encounters))
        list_not_encouters.append( encounter.get_not_encounters( gdf_filtered, list_encounters, 18) )
        encounter_count_id = encounter.encounters_count

        # msg_telegram( "Execução dos arquivos AIS finalizada com sucesso!" )
    except Exception as e   :
        print("Ocorreu um erro!")
        traceback.print_exc()
        msg_telegram( "Execução finalizada com erro!" )
        erro=True
        break

if not erro:
    msg_telegram( "Execução dos arquivos AIS Junho finalizada com sucesso!" )

# %%

prefix_filename = "dados_ais_julho/AIS_2015_07_"
n_files = 30
# AFAZER: ver pq o arquivo 31 de julho nao esta presente!
erro=False
for i in tqdm( range(1, n_files+1) ):
    filename = prefix_filename + "{:02}".format(i) + ".csv"
    encounter = Encounter(encounter_count_id=encounter_count_id)
    encounter.built_trajectories_from_scratch( filename )

    try: 
        gdf_filtered = encounter.get_gdf( )[:]
        number_ais_messages_processed += len(gdf_filtered)
        # 18km ~ 10 MN
        list_encounters.append( encounter.detect_encounters_h3(gdf_filtered, time_interval_m=120, min_distance_from_coast=18, min_distance_between_ships_km=1) )
        print("SUSPECTED CLUSTERS: ", len(list_encounters))
        list_not_encouters.append( encounter.get_not_encounters( gdf_filtered, list_encounters, 18) )
        encounter_count_id = encounter.encounters_count

        # msg_telegram( "Execução dos arquivos AIS finalizada com sucesso!" )
    except Exception as e   :
        print("Ocorreu um erro!")
        traceback.print_exc()
        msg_telegram( "Execução finalizada com erro!" )
        erro=True
        break

if not erro:
    msg_telegram( "Execução dos arquivos AIS Julho finalizada com sucesso!" )

# %%

prefix_filename = "dados_ais_agosto/AIS_2015_08_"
n_files = 30

erro=False
for i in tqdm( range(1, n_files+1) ):
    filename = prefix_filename + "{:02}".format(i) + ".csv"
    encounter = Encounter(encounter_count_id=encounter_count_id)
    encounter.built_trajectories_from_scratch( filename )

    try: 
        gdf_filtered = encounter.get_gdf( )[:]
        number_ais_messages_processed += len(gdf_filtered)
        # 18km ~ 10 MN
        list_encounters.append( encounter.detect_encounters_h3(gdf_filtered, time_interval_m=120, min_distance_from_coast=18, min_distance_between_ships_km=1) )
        print("SUSPECTED CLUSTERS: ", len(list_encounters))
        list_not_encouters.append( encounter.get_not_encounters( gdf_filtered, list_encounters, 18) )
        encounter_count_id = encounter.encounters_count

        # msg_telegram( "Execução dos arquivos AIS finalizada com sucesso!" )
    except Exception as e   :
        print("Ocorreu um erro!")
        traceback.print_exc()
        msg_telegram( "Execução finalizada com erro!" )
        erro=True
        break

if not erro:
    msg_telegram( "Execução dos arquivos AIS Agosto finalizada com sucesso!" )

# %%

prefix_filename = "dados_ais_setembro/AIS_2015_09_"
n_files = 30

erro=False
for i in tqdm( range(1, n_files+1) ):
    filename = prefix_filename + "{:02}".format(i) + ".csv"
    encounter = Encounter(encounter_count_id=encounter_count_id)
    encounter.built_trajectories_from_scratch( filename )

    try: 
        gdf_filtered = encounter.get_gdf( )[:]
        number_ais_messages_processed += len(gdf_filtered)
        # 18km ~ 10 MN
        list_encounters.append( encounter.detect_encounters_h3(gdf_filtered, time_interval_m=120, min_distance_from_coast=18, min_distance_between_ships_km=1) )
        print("SUSPECTED CLUSTERS: ", len(list_encounters))
        list_not_encouters.append( encounter.get_not_encounters( gdf_filtered, list_encounters, 18) )
        encounter_count_id = encounter.encounters_count

        # msg_telegram( "Execução dos arquivos AIS finalizada com sucesso!" )
    except Exception as e   :
        print("Ocorreu um erro!")
        traceback.print_exc()
        msg_telegram( "Execução finalizada com erro!" )
        erro=True
        break

if not erro:
    msg_telegram( "Execução dos arquivos AIS Setembro finalizada com sucesso!" )

# %%

# AFAZER: Nao achou o arquivo 31OUT!!
prefix_filename = "dados_ais_outubro/AIS_2015_10_"
n_files = 31

erro=False
for i in tqdm( range(1, n_files+1) ):
    filename = prefix_filename + "{:02}".format(i) + ".csv"   
    encounter = Encounter(encounter_count_id=encounter_count_id)
    encounter.built_trajectories_from_scratch( filename )
    try: 
        gdf_filtered = encounter.get_gdf( )[:]
        number_ais_messages_processed += len(gdf_filtered)
        # 18km ~ 10 MN
        list_encounters.append( encounter.detect_encounters_h3(gdf_filtered, time_interval_m=120, min_distance_from_coast=18, min_distance_between_ships_km=1) )
        print("SUSPECTED CLUSTERS: ", len(list_encounters))
        list_not_encouters.append( encounter.get_not_encounters( gdf_filtered, list_encounters, 18) )
        encounter_count_id = encounter.encounters_count

        # msg_telegram( "Execução dos arquivos AIS finalizada com sucesso!" )
    except Exception as e   :
        print("Ocorreu um erro!")
        traceback.print_exc()
        msg_telegram( "Execução finalizada com erro!" )
        erro=True
        break

if not erro:
    msg_telegram( "Execução dos arquivos AIS Outubro finalizada com sucesso!" )

# %%

prefix_filename = "dados_ais_novembro/AIS_2015_11_"
n_files = 30

erro=False
for i in tqdm( range(1, n_files+1) ):
    filename = prefix_filename + "{:02}".format(i) + ".csv"
    encounter = Encounter(encounter_count_id=encounter_count_id)
    encounter.built_trajectories_from_scratch( filename )

    try: 
        gdf_filtered = encounter.get_gdf( )[:]
        number_ais_messages_processed += len(gdf_filtered)
        # 18km ~ 10 MN
        list_encounters.append( encounter.detect_encounters_h3(gdf_filtered, time_interval_m=120, min_distance_from_coast=18, min_distance_between_ships_km=1) )
        print("SUSPECTED CLUSTERS: ", len(list_encounters))
        list_not_encouters.append( encounter.get_not_encounters( gdf_filtered, list_encounters, 18) )
        encounter_count_id = encounter.encounters_count

        # msg_telegram( "Execução dos arquivos AIS finalizada com sucesso!" )
    except Exception as e   :
        print("Ocorreu um erro!")
        traceback.print_exc()
        msg_telegram( "Execução finalizada com erro!" )
        erro=True
        break

if not erro:
    msg_telegram( "Execução dos arquivos AIS Novembro finalizada com sucesso!" )

# %%

prefix_filename = "dados_ais_dezembro/AIS_2015_12_"
n_files = 31
#AFAZER: Erro no arquivo 31DEZ

erro=False
for i in tqdm( range(1, n_files+1) ):
    filename = prefix_filename + "{:02}".format(i) + ".csv"
    encounter = Encounter(encounter_count_id=encounter_count_id)
    encounter.built_trajectories_from_scratch( filename )

    try: 
        gdf_filtered = encounter.get_gdf( )[:]
        number_ais_messages_processed += len(gdf_filtered)
        # 18km ~ 10 MN
        list_encounters.append( encounter.detect_encounters_h3(gdf_filtered, time_interval_m=120, min_distance_from_coast=18, min_distance_between_ships_km=1) )
        print("SUSPECTED CLUSTERS: ", len(list_encounters))
        list_not_encouters.append( encounter.get_not_encounters( gdf_filtered, list_encounters, 18) )
        encounter_count_id = encounter.encounters_count

        # msg_telegram( "Execução dos arquivos AIS finalizada com sucesso!" )
    except Exception as e   :
        print("Ocorreu um erro!")
        traceback.print_exc()
        msg_telegram( "Execução finalizada com erro!" )
        erro=True
        break

if not erro:
    msg_telegram( "Execução dos arquivos AIS Dezembro finalizada com sucesso!" )


# %%

# encounter between 2 trajectories
gdf_1 = encounter.get_concat_gdf( list_encounters, 1 )
gdf_2 = encounter.get_concat_gdf( list_encounters, 2 )

# write gdf encounter points in pickle file
encounter.write_pickle_obj( pd.concat( [gdf_1, gdf_2], ignore_index=False), 'gdf_encounters_12meses_len20.pickle'  )

# %%

# # create moving pandas trajectories to encounters
# trajs_enc = encounter.create_trajectory( pd.concat( [gdf_1, gdf_2], ignore_index=False) )

# %%

# write trajectories em pickle file
# encounter.write_pickle_obj( trajs_enc, 'collection_trajectories_encounters_12meses.pickle'  )

# %%

# transform a list of gdf not encounters points in a only one
flat_no_encounter = []
for i in range(len(list_not_encouters)):
    for j in range(len(list_not_encouters[i])):
        flat_no_encounter.append( list_not_encouters[i][j] )

concat_gdf_no_encounters = pd.concat( flat_no_encounter, ignore_index=False) 

# write gdf not encounters in pickle file
encounter.write_pickle_obj( concat_gdf_no_encounters, 'gdf_not_encounters_12meses_len20.pickle'  )

# %%

concat_gdf_no_encounters = concat_gdf_no_encounters.drop( ['direction', 'speed'], axis=1 )

# create moving pandas trajectories to not encounter trajectories
# trajs_no_encounters = encounter.create_trajectory( concat_gdf_no_encounters )

# write trajectories in pickle file
# encounter.write_pickle_obj( trajs_no_encounters, 'collection_trajectories_no_encounters_12meses.pickle'  )

# %%

print("Mensagens AIS processadas: ", number_ais_messages_processed )
msg_telegram( "PROCESSAMENTO DOS ARQUIVOS AIS FINALIZADOS! MSG AIS processadas: " + str(number_ais_messages_processed) )

# %%

count_trajectories_enc = 0
for i in range(len(list_encounters)):
    count_trajectories_enc += len( list_encounters[i] )
# each encouter has 2 trajectories
count_trajectories_enc *= 2

count_trajectories_nenc = 0
for i in range(len(list_not_encouters)):
    count_trajectories_nenc += len( list_not_encouters[i] )
# each encouter has 2 trajectories
count_trajectories_nenc *= 2


print("Mensagens AIS processadas: ", number_ais_messages_processed )
print( f"Trajetorias de encontros: {count_trajectories_enc}" )
print( f"Trajetorias normais: {count_trajectories_nenc}" )

# %%

encounter.write_pickle_obj( list_encounters, 'list_encounters_12meses.pickle'  )



# %%
# #####################################
#
# Tests codes
# ######################################


# Read test
# Em 2.313.354.048 de linhas, temos 38.113 trajetorias
enc = Encounter( )
# oficial
# gdf_enc_test = enc.read_pickle_obj('gdf_encounters_12meses.pickle')
gdf_enc_test = enc.read_pickle_obj('gdf_encounters_12meses_len20.pickle')


# %%
# Filter by encounter_id first
encounter_id = 60000
filtered_gdfs = [
    gdf_enc_test[(gdf_enc_test['encounter_id'] == encounter_id) & (gdf_enc_test['MMSI'] == mmsi)] 
    for mmsi in gdf_enc_test.loc[gdf_enc_test['encounter_id'] == encounter_id, 'MMSI'].unique()
]
# %%
# plot all encounters
m = None
for encounter_id in gdf_enc_test['encounter_id'].unique():
    filtered_gdfs = [
        gdf_enc_test[(gdf_enc_test['encounter_id'] == encounter_id) & (gdf_enc_test['MMSI'] == mmsi)] 
        for mmsi in gdf_enc_test.loc[gdf_enc_test['encounter_id'] == encounter_id, 'MMSI'].unique()
    ]
    m = enc.plot_encounter(filtered_gdfs[0], filtered_gdfs[1], m)
m

# %%
# Create H3 index for resolutions 11 and 10.

# resolution 11
gdf_enc_test['h3_reso11_index'] = [
    h3.latlng_to_cell(lat, lon, 11) for lat, lon in zip(gdf_enc_test['LAT'], gdf_enc_test['LON'])
]

# resolution 9
gdf_enc_test['h3_reso9_index'] = [
    h3.latlng_to_cell(lat, lon, 9) for lat, lon in zip(gdf_enc_test['LAT'], gdf_enc_test['LON'])
]

# %%

# Calculate for how much time the encounter happened
# How much time do the ships spend 50 meters distance. 
# If the AIS timestamp difference greater than 10 minutes or distance greater than 50 m, the proximity_duration_minutes is NaN
# Calc all distance between ship points; detect the oldest time and newest time for calc the time.
import numpy as np
from geopy.distance import geodesic

# Definir uma distância de proximidade (em metros, por exemplo)
threshold_distance = 50  # distância de 50 metros

def calculate_proximity_duration(encounter_df):
    # Obter os MMSIs distintos
    mmsi_list = encounter_df['MMSI'].unique()
    if len(mmsi_list) != 2:
        return 0  # Ignorar encontros com menos ou mais de 2 embarcações

    # Separar as duas trajetórias
    traj_1 = encounter_df[encounter_df['MMSI'] == mmsi_list[0]].sort_values('BaseDateTime').reset_index(drop=True)
    traj_2 = encounter_df[encounter_df['MMSI'] == mmsi_list[1]].sort_values('BaseDateTime').reset_index(drop=True)

    start_time = None
    end_time = None

    try:
        # Calcular distância ponto a ponto e registrar o tempo de proximidade
        for i in range(len(traj_1)):
            # # Tratar caso a indexação seja diferente entre as duas trajetórias
            # if i >= len(traj_2):
            #     raise IndexError(f"Ponto {i} não existe na segunda trajetória para encounter_id: {encounter_df['encounter_id'].iloc[0]}")

            for j in range(len(traj_2)):
                t1 = traj_1.loc[i, 'BaseDateTime']
                t2 = traj_2.loc[j, 'BaseDateTime']
                # Calcular a diferença em minutos
                time_difference = abs((t2 - t1).total_seconds()) / 60  # abs() garante que a diferença seja sempre positiva
                # se a diferenca for menor que 10 minutos
                if time_difference < 10:
                    point1 = (traj_1.loc[i, 'LAT'], traj_1.loc[i, 'LON'])
                    point2 = (traj_2.loc[j, 'LAT'], traj_2.loc[j, 'LON'])
                    distance = geodesic(point1, point2).meters

                    if distance <= threshold_distance:
                        if traj_1.loc[i, 'BaseDateTime'] < traj_1.loc[j, 'BaseDateTime']:
                            if start_time is None or traj_1.loc[i, 'BaseDateTime'] < start_time:
                                start_time = traj_1.loc[i, 'BaseDateTime']
                            if end_time is None or traj_1.loc[i, 'BaseDateTime'] > end_time:
                                end_time = traj_1.loc[i, 'BaseDateTime']
                        else:
                            if start_time is None or traj_1.loc[j, 'BaseDateTime'] < start_time:
                                start_time = traj_1.loc[j, 'BaseDateTime']
                            if end_time is None or traj_1.loc[i, 'BaseDateTime'] > end_time:
                                end_time = traj_1.loc[i, 'BaseDateTime']                       

    except KeyError as e:
        print(f"KeyError na linha {i} com encounter_id: {encounter_df['encounter_id'].iloc[0]}, MMSI: {traj_1.loc[i, 'MMSI']} ou {traj_2.loc[i, 'MMSI']}. Erro: {e}")
        return 0  # Retornar 0 para lidar com o erro sem interromper o loop
    except IndexError as e:
        print(f"IndexError: {e}")
        return 0

    # Calcular a duração em minutos
    if start_time and end_time:
        duration = (end_time - start_time).total_seconds() / 60  # duração em minutos
        return duration
    else:
        return 0  # Caso não fiquem próximas por tempo relevante

# Aplicar a função em cada 'encounter_id'
gdf_consolidation = gdf_enc_test.reset_index()
encounter_durations = gdf_consolidation.groupby('encounter_id').apply(calculate_proximity_duration).reset_index(name='proximity_duration_minutes')
# encounter_durations = gdf.groupby('encounter_id').apply(calculate_proximity_duration)

encounter_durations = encounter_durations[encounter_durations['proximity_duration_minutes'] > 0]
# Mesclar com o GeoDataFrame original, se necessário
gdf_consolidation = gdf_enc_test.reset_index().merge(encounter_durations, on='encounter_id', how='left')

# Exibir resultados
print(encounter_durations.head())

# %%
# Detect who is Waiting and who is Arriving using Timestamp
# The first arrives in the cell is the waiting and the last is the arriving.
traj_data = []
for eid in gdf_consolidation['encounter_id'].unique():

    try:
        mmsis = gdf_consolidation[ gdf_consolidation['encounter_id'] == eid ]['MMSI'].unique()

        gdf1 = gdf_consolidation[ (gdf_consolidation['MMSI'] == mmsis[0]) & (gdf_consolidation['encounter_id'] == eid) ]
        gdf2 = gdf_consolidation[ (gdf_consolidation['MMSI'] == mmsis[1]) & (gdf_consolidation['encounter_id'] == eid) ]

        if not (len(gdf1) == 21 and len(gdf2) == 21):
            continue

        # I'm using resolution 10 because I already know that is a encounter, and using resolution 11 in this
        # case It's very small to capture the both trajectories.
        # Find the common rows based on the 'h3_reso9_index' column
        common_rows = gdf1[gdf1['h3_reso9_index'].isin(gdf2['h3_reso9_index'])].reset_index()
        all_rows_in_cells = gpd.GeoDataFrame(pd.concat( [ 
            gdf1[ gdf1['h3_reso9_index'].isin( common_rows['h3_reso9_index'] ) ], 
            gdf2[ gdf2['h3_reso9_index'].isin( common_rows['h3_reso9_index'] ) ] 
            ],
            ignore_index=False)).reset_index( )
        # Encontrando a linha com a data mais antiga
        oldest_row = all_rows_in_cells.loc[all_rows_in_cells['BaseDateTime'].idxmin()]
        
        if oldest_row['MMSI'] == mmsis[0]:
            traj_data.append( [eid, mmsis[0], 'waiting' ] )    
            traj_data.append( [eid, mmsis[1], 'arriving'] )
        else:
            traj_data.append( [eid, mmsis[0], 'arriving'] )
            traj_data.append( [eid, mmsis[1], 'waiting' ] )    
            
   
    except ValueError as e:
        print(f"ValueError na eid {eid} com rows: {common_rows}")
        print(f"len gdf1 = {len(gdf1)} e len gdf2 = {len(gdf2)}")

    except IndexError as e:
        print(f"IndexError: {e}")

df_data = pd.DataFrame(traj_data, columns=['encounter_id', 'MMSI', 'event'])
gdf_consolidation = gdf_consolidation.merge(df_data, on=['encounter_id', 'MMSI'], how='left')

# %%
## Statistics

# Number of encounters
n_encounters = gdf_consolidation['encounter_id'].iloc[-1] + 1
print(f"Number of encounters: {n_encounters}")
# Number of trajectories
print(f"Number of trajectories: {n_encounters*2}")

# %%
# Group and count occurrencies by Vessel Type

# gdf_consolidation['Status'] = gdf_consolidation['VesselType'].fillna('Unknown')
# df_grouped = gdf_consolidation.groupby(['encounter_id', 'MMSI', 'VesselType']).count().reset_index(name='Count')
unique_count_vesseltype = gdf_consolidation[['encounter_id', 'VesselType','MMSI']].drop_duplicates()
# count_by_vessel_type = df_grouped['VesselType'].map(vessel_type_mapping).value_counts(dropna=False)
count_by_vessel_type = unique_count_vesseltype['VesselType'].map(vessel_type_mapping).value_counts(dropna=False)
df_result = count_by_vessel_type.reset_index()
df_result.columns = ['VesselTypeName', 'Count']
print(df_result)
df_result.iloc[:,1].sum()
# %%
# In this case, we can have more distinct rows than n_trajectories 
# because the ship status can change during the trajectories;

# gdf_consolidation['Status'] = gdf_consolidation['Status'].fillna('Unknown')
# df_grouped = gdf_consolidation.groupby(['encounter_id', 'MMSI', 'Status']).size().reset_index(name='Count')
unique_count_status = gdf_consolidation[['encounter_id', 'Status','MMSI']].drop_duplicates()
# Map STATUS codes to descriptions and count occurrences
# df_status_counts = df_grouped['Status'].map(status_mapping).value_counts(dropna=False)
count_by_vessel_status = unique_count_status['Status'].map(status_mapping).value_counts(dropna=False)
df_status_counts = count_by_vessel_status.reset_index()
df_status_counts.columns = ['STATUS_DESCRIPTION', 'Count']
print(df_status_counts)
df_status_counts.iloc[:,1].sum()

# # %%

# # Dictionary mapping Cargo codes to descriptions
# cargo_mapping = {
#     0: "Undefined",
#     10: "General Cargo",
#     20: "Chemical Tanker",
#     30: "Oil Tanker",
#     40: "Liquefied Gas Tanker",
#     50: "Bulk Carrier",
#     60: "Container Ship",
#     70: "Refrigerated Cargo",
#     80: "Roll-on/Roll-off",
#     90: "Passenger Ship"
# }

# unique_count_cargo = gdf_consolidation[['encounter_id', 'Cargo','MMSI']].drop_duplicates()
# # df_grouped = gdf_consolidation.groupby(['encounter_id', 'MMSI', 'Cargo']).count().reset_index()
# # Map Cargo codes to descriptions and count occurrences
# df_cargo_counts = unique_count_cargo['Cargo'].map(cargo_mapping).value_counts(dropna=False)
# df_cargo_counts = df_cargo_counts.reset_index()
# df_cargo_counts.columns = ['Cargo_Description', 'Count']
# print(df_cargo_counts)
# df_cargo_counts.iloc[:,1].sum()

# %%
# Pair of Type vessels encounter;
unique_pairs = gdf_consolidation[['encounter_id', 'VesselType', 'MMSI' ]].drop_duplicates()
l_eid = []
l_t1 = []
l_t2 = []

try:
    for eid in gdf_consolidation['encounter_id'].unique():
        vt = unique_pairs[ unique_pairs['encounter_id'] == eid ]
        l_eid.append( eid )
        l_t1.append( vt['VesselType'].iloc[0] )
        l_t2.append( vt['VesselType'].iloc[1] )

except ValueError as e:
    print(f"ValueError na eid {eid} com rows: {common_rows}")
    print(f"len gdf1 = {len(gdf1)} e len gdf2 = {len(gdf2)}")

except IndexError as e:
    print(f"IndexError: {e}")
    print(f"ValueError na eid {eid} vt: {vt}")

mapped_values_t1 = [vessel_type_mapping.get(value, "Unknown") for value in l_t1]    
mapped_values_t2 = [vessel_type_mapping.get(value, "Unknown") for value in l_t2]    

df_pairs = pd.DataFrame({
    'eid': l_eid,
    't1': mapped_values_t1,
    't2': mapped_values_t2
})

# Substituir valores para tratar equivalências
df_pairs['t1'] = df_pairs['t1'].replace(["Not available (default)", "Other sea-going vessel"], "Unknown")
df_pairs['t2'] = df_pairs['t2'].replace(["Not available (default)", "Other sea-going vessel"], "Unknown")

# Recalcular os pares ordenados considerando as novas substituições
df_pairs['pair'] = df_pairs.apply(lambda row: tuple(sorted([row['t1'], row['t2']])), axis=1)
pair_counts_combined_updated = df_pairs['pair'].value_counts().reset_index()
pair_counts_combined_updated.columns = ['pair', 'count']

# Exibir os resultados atualizados
pair_counts_combined_updated[:10]


# %%
# mean of duration
m = gdf_consolidation[['encounter_id', 'proximity_duration_minutes']][ gdf_consolidation['proximity_duration_minutes'].notna()].groupby('encounter_id').mean()
m['proximity_duration_minutes'].mean()

# %%
# Draft change during encounter
#

eid_draft_change = []
for eid in gdf_consolidation['encounter_id'].unique():
    mmsis = gdf_consolidation[ gdf_consolidation['encounter_id'] == eid ]['MMSI'].unique()

    df1 = gdf_consolidation[ (gdf_consolidation['MMSI'] == mmsis[0]) & (gdf_consolidation['encounter_id'] == eid) ]
    df2 = gdf_consolidation[ (gdf_consolidation['MMSI'] == mmsis[1]) & (gdf_consolidation['encounter_id'] == eid) ]

    draft_std_1 = round( df1['Draft'].std(), 2)
    draft_std_2 = round( df2['Draft'].std(), 2)

    if draft_std_1 > 0:
        eid_draft_change.append( [eid, mmsis[0], df1['VesselType'].iloc[0]] ) 
    
    if draft_std_2 > 0:
        eid_draft_change.append( [eid, mmsis[1], df2['VesselType'].iloc[0]] ) 

print(f"Trajectories with Draft change during encounter: {len(eid_draft_change)}")

# %%

vessel_drafts = np.array( eid_draft_change )
df_drafts = pd.DataFrame({
    'eid': vessel_drafts[:, 0],
    'mmsi': vessel_drafts[:, 1],
    'vessel_type': vessel_drafts[:, 2]
})

# %%
# Pairs of Draft Change
#

eid_daft_changes = list(vessel_drafts[:, 0].astype(int))
unique_pairs = gdf_consolidation[['encounter_id', 'VesselType', 'MMSI' ]].drop_duplicates()
l_eid = []
l_t1 = []
l_t2 = []

try:
    for eid in eid_daft_changes:
        vt = unique_pairs[ unique_pairs['encounter_id'] == eid ]
        l_eid.append( eid )
        l_t1.append( vt['VesselType'].iloc[0] )
        l_t2.append( vt['VesselType'].iloc[1] )

except ValueError as e:
    print(f"ValueError na eid {eid} com rows: {common_rows}")
    print(f"len gdf1 = {len(gdf1)} e len gdf2 = {len(gdf2)}")

except IndexError as e:
    print(f"IndexError: {e}")
    print(f"ValueError na eid {eid} vt: {vt}")

mapped_values_t1 = [vessel_type_mapping.get(value, "Unknown") for value in l_t1]    
mapped_values_t2 = [vessel_type_mapping.get(value, "Unknown") for value in l_t2]    

df_pairs = pd.DataFrame({
    'eid': l_eid,
    't1': mapped_values_t1,
    't2': mapped_values_t2
})

# Substituir valores para tratar equivalências
df_pairs['t1'] = df_pairs['t1'].replace(["Not available (default)", "Other sea-going vessel"], "Unknown")
df_pairs['t2'] = df_pairs['t2'].replace(["Not available (default)", "Other sea-going vessel"], "Unknown")

# Recalcular os pares ordenados considerando as novas substituições
df_pairs['pair'] = df_pairs.apply(lambda row: tuple(sorted([row['t1'], row['t2']])), axis=1)
pair_counts_combined_updated = df_pairs['pair'].value_counts().reset_index()
pair_counts_combined_updated.columns = ['pair', 'count']

# Exibir os resultados atualizados
pair_counts_combined_updated[:10]

# %%

#Write  consolidated dataset
enc.write_pickle_obj( gdf_consolidation, 'gdf_12months_encounters_len20.pickle'  )

# %%
# read no encounter trajectories
gdf_nenc_test = enc.read_pickle_obj('gdf_not_encounters_12meses.pickle')

# %%
# Write no encounter trajectories
enc.write_pickle_obj( gdf_nenc_test, 'gdf_not_encounters_12meses.pickle'  )

# %%

gdf_enc_test[ gdf_enc_test['encounter_id'] == 3 ][['MMSI', 'Draft']]
# %%

g = gdf_enc_test[ (gdf_enc_test['encounter_id'] == 23443) & (gdf_enc_test['MMSI'] == 538090163)  ]
g
# %%

# create moving pandas trajectories to encounters
trajs_enc = enc.create_trajectory( gdf_enc_test )

# %%

gdf_ne_test = enc.read_pickle_obj('gdf_not_encounters_12meses.pickle')

# %%

trajs_ne = enc.create_trajectory( gdf_ne_test )

# %%
# # Detect who is Waiting and who is Arriving using SOG
traj_data = []
for eid in gdf_enc_test['encounter_id'].unique():

    mmsis = gdf_enc_test[ gdf_enc_test['encounter_id'] == eid ]['MMSI'].unique()

    df1 = gdf_enc_test[ (gdf_enc_test['MMSI'] == mmsis[0]) & (gdf_enc_test['encounter_id'] == eid) ]
    df2 = gdf_enc_test[ (gdf_enc_test['MMSI'] == mmsis[1]) & (gdf_enc_test['encounter_id'] == eid) ]

    sog_mean_1 = df1['SOG'][:10].mean()
    sog_mean_2 = df2['SOG'][:10].mean()
    # heading_1 = df1['Heading'][:10].diff(1).fillna(0).mean()
    
    # heading_2 = df2['Heading'][:10].diff(1).fillna(0).mean()
    
    # delta_cog = df1['COG'][:10] - df1['COG'][:10].shift(1)
    # # Normalizar delta_cog para o intervalo [-180, 180]
    # delta_cog = ((delta_cog + 180) % 360) - 180
    # delta_cog.fillna(0)
    # cog_var_1 = delta_cog.var()

    # delta_cog = df2['COG'][:10] - df2['COG'][:10].shift(1)
    # # Normalizar delta_cog para o intervalo [-180, 180]
    # delta_cog = ((delta_cog + 180) % 360) - 180
    # delta_cog.fillna(0)
    # cog_var_2 = delta_cog.var()

    if sog_mean_1 < sog_mean_2:
        traj_data.append( [eid, mmsis[0], 'waiting' ] )    
        traj_data.append( [eid, mmsis[1], 'arriving'] )
    else:
        traj_data.append( [eid, mmsis[0], 'arriving'] )
        traj_data.append( [eid, mmsis[1], 'waiting' ] )    
        


    # traj_data.append( [eid, mmsis[0], sog_mean_1, heading_1, cog_var_1] )
#     traj_data.append( [eid, mmsis[1], sog_mean_2, heading_2, cog_var_2] )



# %%


# %%
## Plot the encounter
encounter_id = 10930
gdf = gdf_enc_test[gdf_enc_test['encounter_id']==encounter_id].reset_index()
g = calculate_proximity_duration(gdf)

m = None
mmsis = gdf['MMSI'].unique()
gdf1 = gdf[(gdf['encounter_id'] == encounter_id) & (gdf['MMSI'] == mmsis[0])] 
gdf2 = gdf[(gdf['encounter_id'] == encounter_id) & (gdf['MMSI'] == mmsis[1])] 
m = enc.plot_encounter(gdf1, gdf2, m)
m

# %%
import folium
from shapely.geometry import Polygon
from folium import GeoJson
import h3

# Função para desenhar hexágono no mapa existente
def draw_h3_hexagon(h3_index, map_object):
    # Obter as coordenadas do hexágono
    hex_boundary = h3.cell_to_boundary(h3_index)  # Use h3_to_boundary
    
    # Criar o polígono no formato aceito pelo Folium
    folium.Polygon(
        locations=hex_boundary,  # Lista de coordenadas
        color='blue',  # Cor das bordas
        fill=True,  # Preencher o polígono
        fill_color='orange',  # Cor de preenchimento
        fill_opacity=0.5  # Transparência do preenchimento
    ).add_to(map_object)
    
    return map_object

# %%

h3_index = "89447397417ffff"
# m = folium.Map(location=[-22.9068, -43.1729], zoom_start=12)  # Rio de Janeiro como exemplo
draw_h3_hexagon(h3_index, m)

# %%

