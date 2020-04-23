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

    def download_hdx_population_data(self, resource):
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

    
