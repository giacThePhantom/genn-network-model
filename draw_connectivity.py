from networkx.drawing.nx_agraph import write_dot
import networkx as nx



def create_glomerulus_graph(network, glomerulus_idx, outname):
    """From a network creates the dot file necessary for drawing the graph of the
    glomerulus with index glomerulus_idx. This can be later visualized with an
    application like Gephi with the Network Splitter 3D plugin
    Parameters
    ----------
    network : NeuronalNetwork
        The network for which to draw the glomerulus
    glomerulus_idx : int
        Which glomerulus to draw
    outname : string
        The file for which to save the dot file for the graph
    """

    # Get the offset of each population so that later on a neuron id is unique across
    # populations
    offset = {}
    glomerulus_offset = glomerulus_idx*network.neuron_populations['or'].size()
    last_offset = glomerulus_offset
    for i in network.neuron_populations:
        offset[i] = last_offset
        last_offset += network.neuron_populations[i].size()

    res = nx.DiGraph()
    connectivity = network.get_connectivity()

    nodes = {}
    # The order in which the different populations are displayed using the
    # Network Splitter 3D plugin for Gephi
    order = {'or' : 0, 'orn' : 1, 'ln' : 2, 'pn' : 3 }

    # Each node is a neuron, with the unique id transformed thanks to offset
    for i in offset:
        nodes[i] = [(x + offset[i], {'height [z]' : order[i], 'pop' : i} ) for x in range(network.neuron_populations[i].size()//network.neuron_populations['or'].size())]


    for i in offset:
        res.add_nodes_from(nodes[i])

    # Add the edges to the graph, considering the transformation of a glomerulus
    for i in connectivity:
        if i['pre_id'] + offset[i['pre_population']] in res.nodes and i['post_id'] + offset[i['post_population']] in res.nodes:
            res.add_edge(i['pre_id']+offset[i['pre_population']], i['post_id']+offset[i['post_population']])

    print("Writing")
    write_dot(res, outname)
