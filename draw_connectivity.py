import networkx as nx
from networkx.drawing.nx_agraph import write_dot
from graph_tool.all import *



def create_glomerulus_graph(network, glomerulus_idx, outname):
    offset = {}
    glomerulus_offset = glomerulus_idx*network.neuron_populations['or'].size()
    last_offset = glomerulus_offset
    for i in network.neuron_populations:
        offset[i] = last_offset
        last_offset += network.neuron_populations[i].size()

    nodes = {}


    res = nx.DiGraph()
    connectivity = network.get_connectivity()

    nodes = {}

    order = {'or' : 0, 'orn' : 1, 'ln' : 2, 'pn' : 3 }

    for i in offset:
        nodes[i] = [(x + offset[i], {'height [z]' : order[i], 'pop' : i} ) for x in range(network.neuron_populations[i].size()//network.neuron_populations['or'].size())]


    for i in offset:
        res.add_nodes_from(nodes[i])

    for i in connectivity:
        if i['pre_id'] + offset[i['pre_population']] in res.nodes and i['post_id'] + offset[i['post_population']] in res.nodes:
            res.add_edge(i['pre_id']+offset[i['pre_population']], i['post_id']+offset[i['post_population']])

    print("Writing")
    write_dot(res, outname)

def create_network_graph(network, outname):
    offset = {}
    last_offset = 0
    for i in network.neuron_populations:
        offset[i] = last_offset
        last_offset += network.neuron_populations[i].size()

    res = Graph()
    connectivity = network.get_connectivity()

    nodes = {}

    order = {'or' : 0, 'orn' : 1, 'ln' : 2, 'pn' : 3 }

    for i in offset:
        order = {'or' : 0, 'orn' : 1, 'ln' : 2, 'pn' : 3 }
        nodes[i] = [(x + offset[i], {'height [z]' : order[i], 'pop' : i} ) for x in range(network.neuron_populations[i].size())]

    for i in offset:
        res.add_vertex_list(nodes[i])

    for i in connectivity:
        res.add_edge(i['pre_id']+offset[i['pre_population']], i['post_id']+offset[i['post_population']])

    write_dot(res, outname)
