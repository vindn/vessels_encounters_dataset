import geopandas as gpd
from geopy.distance import geodesic
from shapely.geometry import Point
from shapely.ops import nearest_points
import numpy as np
from tqdm import tqdm
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas(desc="Progress")

class CalcDistanceToCoast:
    def __init__(self):
        # Path to the GeoPackage file
        self.brazil_gpkg = 'data/gadm41_BRA.gpkg'
        self.eez_shp = 'data/zee/EEZ_land_union_v3_202003/EEZ_Land_v3_202030.shp'
        # Load the specific layer from the GeoPackage
        self.gdf_brazil = gpd.read_file(self.brazil_gpkg)
        self.gdf_eez_brazil = gpd.read_file(self.eez_shp)
        self.gdf_eez_brazil = self.gdf_eez_brazil[self.gdf_eez_brazil["UNION"] == "Brazil"]

    def show_map_brazil(self):
        return self.gdf_brazil.plot()
    
    def show_eez_map_brazil(self):
        return self.gdf_eez_brazil.plot()
    
    # Function to calculate geodesic distance in nautical miles
    def calculate_geodesic_distance(self, point1, point2):
        return geodesic(point1, point2).nautical

    def distance_to_brazil_coast(self, lon, lat):
        # Find the nearest point on the coastline of the MULTIPOLYGON
        nearest_point = nearest_points(Point(lon, lat), self.gdf_brazil.geometry[0])[1]

        # Calculate the geodesic distance in miles and convert to nautical miles
        distance_miles = geodesic((lat, lon), (nearest_point.y, nearest_point.x)).miles
        distance_nautical = distance_miles * 0.868976
        
        return distance_nautical
    
    def distance_to_brazil_coast_array(self, trajs):
        distances = []
        print("Calculating distances between trajectories to coast ...")
        for traj in tqdm(trajs.trajectories[:]):
            d1 = self.distance_to_brazil_coast(traj.df.iloc[0]["lon"], traj.df.iloc[0]["lat"])
            d2 = self.distance_to_brazil_coast(traj.df.iloc[-1]["lon"], traj.df.iloc[-1]["lat"])
            # Average between the greatest and smallest distance
            distances.append((d1 + d2) / 2)

        return np.array(distances)

    def detect_arriving(self, trajs):
        arriving = []
        print("Calculating distances between trajectories to coast ...")
        for traj in tqdm(trajs.trajectories[:]):
            d1 = self.distance_to_brazil_coast(traj.df.iloc[0]["lon"], traj.df.iloc[0]["lat"])
            d2 = self.distance_to_brazil_coast(traj.df.iloc[-1]["lon"], traj.df.iloc[-1]["lat"])
            
            diff = d1 - d2
            dist = self.calculate_geodesic_distance((traj.df.iloc[0]["lat"], traj.df.iloc[0]["lon"]), (traj.df.iloc[-1]["lat"], traj.df.iloc[-1]["lon"]))
            if dist == 0:
                arriving.append(0)
            else:
                arriving.append(diff / dist)
            # if d2 < d1:
            #     # If the last position is less than the first position, the vessel may be arriving
            #     arriving.append(diff / dist)
            # else:
            #     # If the last position is greater than the first position, the vessel may be leaving
            #     arriving.append(0)           

        return np.array(arriving)

    def get_gdf_brazil(self):
        return self.gdf_brazil
    
    def check_trajectory_crossed_coast(self, traj):       
        return self.gdf_brazil.geometry.crosses(traj.to_linestring())
    
    def calc_point_inside_eez(self, lon, lat):
        point = Point(lon, lat)
        # This returns a Series of boolean values
        is_inside = self.gdf_eez_brazil.contains(point)

        # To check if the point is inside at least one polygon
        return is_inside.any()

    def calc_traj_inside_eez(self, traj):
        is_inside = traj.df.apply(lambda row: self.gdf_eez_brazil.contains(row['geometry']).any(), axis=1)
        return is_inside

    def calc_trajs_inside_eez(self, trajs):
        print("Calculating trajectories inside Brazil EEZ ...")
        t = []
        for traj in tqdm(trajs.trajectories[:]):
            is_inside = traj.df.apply(lambda row: self.gdf_eez_brazil.contains(row['geometry']).any(), axis=1)
            d = is_inside.any()
            t.append(int(d))

        return np.array(t)

    def filter_gdf_inside_eez(self, gdf):        
        dist = CalcDistanceToCoast()
        is_inside = gdf.reset_index().progress_apply(lambda row: self.gdf_eez_brazil.contains(row['geometry']).any(), axis=1)
        return gdf.reset_index()[is_inside]
