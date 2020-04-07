import time
import pickle
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point


print("Loading graph")
init = time.time()
graph = pickle.load(open("data/input/lima_graph_proj.pk","rb"))
wait = time.time() - init
print("graph loaded in", wait)

# walking speed in km/hour
travel_speed = 4.5
# add an edge attribute for time in minutes required to traverse each edge
meters_per_minute = travel_speed * 1000 / 60 #km per hour to m per minute
for u, v, k, data in G.edges(data=True, keys=True):
    data['time'] = data['length'] / meters_per_minute

print("Loading nodes")
init = time.time()
nodes = ox.graph_to_gdfs(graph, nodes=True, edges=False)
wait = time.time() - init
print("nodes loaded in", wait)
print('#'*40)
print('n_nodes:',len(nodes))
print('#'*40)

## Helpers

def get_boundingbox(x, y, margin):
    x = Point(x)
    y = Point(y)
    xy = gpd.GeoDataFrame(geometry=[x, y], crs=proj_utm)
    xmin, ymin, xmax, ymax = xy.unary_union.bounds
    xmin -= margin
    ymin -= margin
    xmax += margin
    ymax += margin
    print('bbox:', xmin, xmax, ymin, ymax)
    return xmin, ymin, xmax, ymax

def get_subgraph(graph, nodes, source, target, margin):
    xmin, ymin, xmax, ymax = get_boundingbox(source, target, margin=margin)
    subgraph_nodes_ix = nodes.cx[xmin:xmax, ymin:ymax].index
    print('n_subgraph_nodes:',len(subgraph_nodes_ix))
    print("Getting subgraph")
    init = time.time()
    subgraph = graph.subgraph(subgraph_nodes_ix)
    wait = time.time() - init
    print("subgraph loaded in", wait)
    return subgraph, subgraph_nodes_ix

def get_nearest_nodes(graph, source, target):
    print("Getting nearest nodes")
    init = time.time()
    origin_node = ox.get_nearest_node(G=graph, point=(source[1],source[0]), method='euclidean')
    target_node = ox.get_nearest_node(G=graph, point=(target[1], target[0]), method='euclidean')
    wait = time.time() - init
    print("nereast nodes loaded in", wait)
    return origin_node, target_node

def get_route_data(route, nodes):
    route_nodes = nodes.loc[route]
    print('n_route_nodes:',len(route_nodes))
    route_line = list(zip(route_nodes.lat, route_nodes.lon))
    route_linestr = LineString(route_nodes.geometry.values.tolist())

    route_geom = gpd.GeoDataFrame(crs=nodes.crs)
    route_geom['geometry'] = None
    route_geom['osmids'] = None
    route_geom.loc[0,'geometry'] = route_linestr
    route_geom.loc[0,'osmids'] = str(list(route_nodes['osmid'].values))
    route_geom['length_m'] = route_geom.length
    return route_geom, route_line

## Main function

def get_scattermap_lines(source, target):
    '''
    Obtain route from graph
    '''
    # Filter graph to reduce time
    subgraph, subgraph_nodes_ix = get_subgraph(graph, nodes, source, target, 5000)
    # Get nearest nodes in the subgraph
    source_node_id, target_node_id = get_nearest_nodes(subgraph, source, target)

    print('#'*20,'source and target nodes')
    print(source_node_id)
    print(target_node_id)
    print('#'*30)
    # Get shortest_path (list of nodes)
    print("Getting shortest path")
    init = time.time()
    opt_route = nx.shortest_path(G=subgraph, source=source_node_id,
                                 target=target_node_id, weight='length')
    wait = time.time() - init
    print("shortest path in", wait)
    print('#'*40)
    print(opt_route)
    print('#'*40)

    # Get route data
    route_df, route_line = get_route_data(opt_route, nodes)

    return route_line

###################################
