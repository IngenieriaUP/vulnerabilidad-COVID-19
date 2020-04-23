import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shapely
import json
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import requests
from h3 import h3
from scipy.spatial import cKDTree
from tqdm import tqdm


# - Descargar datos de cualquier ciudad (límites, ubicación de puntos de servicio en distintas categorías como hospitales o bancos) en base a OSM
# - Descargar datos demográficos actualizados de la ciudad desde HDX
# - Particionar la superficie de la ciudad en celdas regulares
# - Generar métricas de acceso
# - Generar mapas y visualizaciones con los resultados

class BIDDataDownloader:
    def __init__():
        self.hdx_url = 'https://data.humdata.org/dataset/{}'
        self.overpass_url = "http://overpass-api.de/api/interpreter"
        self.osm_url = 'https://nominatim.openstreetmap.org/search.php'
        self.osm_parameters = {
            'polygon_geojson': '1',
            'format': 'geojson'
        }

    def download_osm(self, expected_position, query):
        '''
        Download OpenStreetMaps data for a specific city

        Parameters
        ----------

        expected_position: int
                           Expected position of the polygon data within the Nominatim results

        query: str
               Query for city polygon data to be downloaded

        Returns
        -------

        city: GeoDataFrame
              GeoDataFrame with the city's polygon as its geometry column


        Example
        -------

        >> lima = download_osm(2, "Lima, Peru")
        >> lima.head()
        geometry	 | place_id	 | osm_type	| osm_id     | display_name	| place_rank  |  category | type	       | importance	| icon
        MULTIPOLYGON | 235480647 | relation	| 1944670.0  | Lima, Peru	| 12	      |  boundary |	administrative | 0.703484	| https://nominatim.openstreetmap.org/images/map...


        '''
        parameters['q'] = query

        response = requests.get(self.url, params=self.parameters)
        all_results = response.json()
        gdf = gpd.GeoDataFrame.from_features(all_results['features'])
        city = gdf.iloc[expected_position:expected_position+1, :]

        return city

    def merge_geom_downloads(self, gdfs):
        '''
        Merge several GeoDataFrames from OSM download_osm

        Parameters
        ----------

        dfs: array_like
             Array of GeoDataFrames to merge

        Returns
        -------

        concat: GeoDataFrame
                Output from concatenation and unary union of geometries, providing
                a single geometry database for the city

        Example
        -------

        >> lima = download_osm(2, "Lima, Peru")
        >> callao = download_osm(1, "Lima, Peru")
        >> lima_ = merge_geom_downloads([lima, callao])
        >> lima_.head()
        geometry
        MULTIPOLYGON (((-76.80277 -12.47562, -76.80261...

        '''

        concat = gpd.GeoDataFrame(geometry=[pd.concat(gdfs).unary_union])
        return concat

    def download_hdx(self, resource):
        '''
        Download the High Resolution Population Density maps from HDX.

        Parameters
        ----------

        resource: str
                  Specific address to the resource for each city. Since every dataset
                  is referenced to a diferent resource id, only the base url can be provided
                  by the library

        Returns
        -------
        population: DataFrame
                    DataFrame with lat, lon, and population columns. Coordinates
                    are in EPSG 4326.


        Example
        -------

        >> pop_lima = download_hdx_population_data("4e74db39-87f1-4383-9255-eaf8ebceb0c9/resource/317f1c39-8417-4bde-a076-99bd37feefce/download/population_per_2018-10-01.csv.zip")
        >> pop_lima.head()

        latitude   | longitude  | population_2015 |	population_2020
        -18.339306 | -70.382361 | 11.318147	      | 12.099885
        -18.335694 | -70.393750 | 11.318147	      | 12.099885
        -18.335694 | -70.387361	| 11.318147	      | 12.099885
        -18.335417 | -70.394028	| 11.318147	      | 12.099885
        -18.335139 | -70.394306	| 11.318147	      | 12.099885

        '''

        population = pd.read_csv(self.hdx_url.format(resource))
        return population

    def filter_population(self, pop_df, polygon_gdf):
        '''
        Filter an HDX database download to the polygon bounds

        Parameters
        ----------

        pop_df: DataFrame
                Result from download_hdx

        polygon_gdf: GeoDataFrame
                     Result from download_osm or merge_geom_downloads

        Returns
        -------

        filtered_points_gdf: GeoDataFrame
                         Population DataFrame filtered to polygon bounds

        Example
        -------

        >> lima_ = merge_geom_downloads([lima, callao])
        >> pop = pop_lima = download_hdx_population_data("4e74db39-87f1-4383-9255-eaf8ebceb0c9/resource/317f1c39-8417-4bde-a076-99bd37feefce/download/population_per_2018-10-01.csv.zip")
        >> filter_population(pop, lima_)
        	latitude   | longitude  | population_2015 | population_2020 | geometry
            -12.519861 | -76.774583 | 2.633668        | 2.644757        | POINT (-76.77458 -12.51986)
            -12.519861 | -76.745972 | 2.633668        | 2.644757        | POINT (-76.74597 -12.51986)
            -12.519861 | -76.745694 | 2.633668        | 2.644757        | POINT (-76.74569 -12.51986)
            -12.519861 | -76.742639 | 2.633668        | 2.644757        | POINT (-76.74264 -12.51986)
            -12.519861 | -76.741250 | 2.633668        | 2.644757        | POINT (-76.74125 -12.51986)

        '''

        minx, miny, maxx, maxy = polygon_gdf.geometry.total_bounds
        limits_filter = pop_df['longitude'].between(minx, maxx) & pop_df['latitude'].between(miny, maxy)
        filtered_points = pop_df[limits_filter]

        geometry_ = gpd.points_from_xy(filtered_points['longitude'], filtered_points['latitude'])
        filtered_points_gdf = gpd.GeoDataFrame(filtered_points, geometry=geometry_, crs='EPSG:4326')

        return filtered_points_gdf

    def remove_features(self, gdf, bounds):
        '''
        Remove a set of features based on bounds

        Parameters
        ----------

        gdf: GeoDataFrame
             Input GeoDataFrame containing the point features filtered with filter_population

        bounds: array_like
                Array input following [miny, maxy, minx, maxx] for filtering


        Returns
        -------
        gdf: GeoDataFrame
             Input DataFrame but without the desired features

        Example
        -------

        >> lima = filter_population(pop_lima, poly_lima)
        >> removed = remove_features(lima, [-12.2,-12, -77.2,-77.17]) #Remove San Lorenzo Island
        >> print(lima.shape, removed.shape)
        (348434, 4) (348427, 4)
        '''

        filter = gdf['latitude'].between(miny,maxy) & gdf['longitude'].between(minx,maxx)
        drop_ix = gdf[filter].index

        return gdf.drop(drop_ix)

    def download_overpass_poi(self, bounds, est_type):
        '''
        Download POIs using Overpass API

        Parameters
        ----------

        bounds: array_like
                Input bounds for query. Follows [minx,miny,maxx,maxy] pattern.

        est_type: {'food_supply', 'healthcare_facilities', 'parks_pitches'}
                  Type of establishment to download. Based on this a different type of query
                  is constructed.

        Returns
        -------

        gdf: GeoDataFrame containing all de POIs from the desired query

        gdf_nodes: Only if 'parks_pitches' selected. Returns point geometry POI GeoDataFrame
        gdf_ways: Only if 'parks_pitches' selected. Returns polygon geometry POI GeoDataFrame

        Example
        -------

        '''
        minx, miny, maxx, maxy = bounds

        bbox_string = f'{minx},{miny},{maxx},{maxy}'

        # Definir consulta para instalaciones de oferta de alimentos en Lima
        overpass_url = "http://overpass-api.de/api/interpreter"

        if est_type == 'food_supply':
            overpass_query = f"""
                [timeout:120][out:json][bbox];
                (
                  node["amenity"="market_place"];
                  node["shop"~"supermarket|kiosk|mall|convenience|butcher|greengrocer"];
                );
                out body geom;
                """
        elif est_type == 'healthcare_facilities':
            overpass_query = f"""
                [timeout:120][out:json][bbox];
                (
                  node["amenity"~"clinic|hospital"];
                );
                out body geom;
                """
        else:
            overpass_query = f"""
                [timeout:120][out:json][bbox];
                (
                  way["leisure"~"park|pitch"];
                  node["leisure"="pitch"];
                );
                out body geom;
                """

        # Request data
        response = requests.get(overpass_url, params={'data': overpass_query,
                                                      'bbox': bbox_string})
        data = response.json()

        if est_type != 'parks_pitches':
            df = pd.DataFrame.from_dict(data['elements'])
            df_geom = gpd.points_from_xy(df['lon'], df['lat'])
            gdf = gpd.GeoDataFrame(df, geometry=df_geom)

            return gdf

        else:
            df = pd.DataFrame.from_dict(data['elements'])

            #Process nodes
            nodes = df[df['type'] == 'node'].drop(['bounds', 'nodes', 'geometry'], axis=1)
            node_geom = gpd.points_from_xy(nodes['lon'], nodes['lat'])
            node_gdf = gpd.GeoDataFrame(nodes, geometry=node_geom)

            #Process ways
            ways = df[df['type'] == 'way'].drop(['lat', 'lon'], axis=1)
            ways['shell'] = ways['geometry'].apply(shell_from_geometry)
            way_geom = ways['shell'].apply(shapely.geometry.Polygon)
            way_gdf = gpd.GeoDataFrame(ways, geometry=way_geom)

            return node_gdf, way_gdf

    def shell_from_geometry(geometry):
        '''
        Util function for park and pitch processing.
        '''

        shell = []
        for record in geometry:
            shell.append([record['lon'], record['lat']])
        return shell

    
