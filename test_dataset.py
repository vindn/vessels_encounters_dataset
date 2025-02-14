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


# %%
# #####################################
#
# Tests codes
# ######################################


# Read test
# Em 2.313.354.048 de linhas, temos 38.113 trajetorias
try:
    with open('dataset_git/gdf_12months_encounters_len20.pickle', 'rb') as data_file:
        gdf_enc_test = pickle.load(data_file)
except Exception as e:
    print(e, "File not Found!")

# oficial
# gdf_enc_test = enc.read_pickle_obj('gdf_encounters_12meses.pickle')
# gdf_enc_test = enc.read_pickle_obj('gdf_encounters_12meses_len20.pickle')

# %%
## Statistics

# Number of encounters
n_encounters = gdf_enc_test['encounter_id'].iloc[-1] + 1
print(f"Number of encounters: {n_encounters}")
# Number of trajectories
print(f"Number of trajectories: {n_encounters*2}")

# %%
# Group and count occurrencies by Vessel Type

# gdf_enc_test['Status'] = gdf_enc_test['VesselType'].fillna('Unknown')
# df_grouped = gdf_enc_test.groupby(['encounter_id', 'MMSI', 'VesselType']).count().reset_index(name='Count')
unique_count_vesseltype = gdf_enc_test[['encounter_id', 'VesselType','MMSI']].drop_duplicates()
# count_by_vessel_type = df_grouped['VesselType'].map(vessel_type_mapping).value_counts(dropna=False)
count_by_vessel_type = unique_count_vesseltype['VesselType'].map(vessel_type_mapping).value_counts(dropna=False)
df_result = count_by_vessel_type.reset_index()
df_result.columns = ['VesselTypeName', 'Count']
print(df_result)
df_result.iloc[:,1].sum()

# %%
# In this case, we can have more distinct rows than n_trajectories 
# because the ship status can change during the trajectories;

# gdf_enc_test['Status'] = gdf_enc_test['Status'].fillna('Unknown')
# df_grouped = gdf_enc_test.groupby(['encounter_id', 'MMSI', 'Status']).size().reset_index(name='Count')
unique_count_status = gdf_enc_test[['encounter_id', 'Status','MMSI']].drop_duplicates()
# Map STATUS codes to descriptions and count occurrences
# df_status_counts = df_grouped['Status'].map(status_mapping).value_counts(dropna=False)
count_by_vessel_status = unique_count_status['Status'].map(status_mapping).value_counts(dropna=False)
df_status_counts = count_by_vessel_status.reset_index()
df_status_counts.columns = ['STATUS_DESCRIPTION', 'Count']
print(df_status_counts)
df_status_counts.iloc[:,1].sum()

# %%
# Pair of Type vessels encounter;
unique_pairs = gdf_enc_test[['encounter_id', 'VesselType', 'MMSI' ]].drop_duplicates()
l_eid = []
l_t1 = []
l_t2 = []

try:
    for eid in gdf_enc_test['encounter_id'].unique():
        vt = unique_pairs[ unique_pairs['encounter_id'] == eid ]
        l_eid.append( eid )
        l_t1.append( vt['VesselType'].iloc[0] )
        l_t2.append( vt['VesselType'].iloc[1] )

except ValueError as e:
    print(f"ValueError na eid {eid} com vt: {vt}")
    print(f"len l_t1 = {len(l_t1)} e len gdl_t2f2 = {len(l_t2)}")

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
m = gdf_enc_test[['encounter_id', 'proximity_duration_minutes']][ gdf_enc_test['proximity_duration_minutes'].notna()].groupby('encounter_id').mean()
m['proximity_duration_minutes'].mean()

# %%
# Draft change during encounter
#

eid_draft_change = []
for eid in gdf_enc_test['encounter_id'].unique():
    mmsis = gdf_enc_test[ gdf_enc_test['encounter_id'] == eid ]['MMSI'].unique()

    df1 = gdf_enc_test[ (gdf_enc_test['MMSI'] == mmsis[0]) & (gdf_enc_test['encounter_id'] == eid) ]
    df2 = gdf_enc_test[ (gdf_enc_test['MMSI'] == mmsis[1]) & (gdf_enc_test['encounter_id'] == eid) ]

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
unique_pairs = gdf_enc_test[['encounter_id', 'VesselType', 'MMSI' ]].drop_duplicates()
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
    print(f"ValueError na eid {eid} com vt: {vt}")
    print(f"len l_t1 = {len(l_t1)} e len l_t2 = {len(l_t2)}")

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
