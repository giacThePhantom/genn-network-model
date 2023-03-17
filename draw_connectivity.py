import networkx as nx
import sys
from networkx.drawing.nx_agraph import write_dot
import matplotlib.pyplot as plt
import json
import numpy as np


def inside_ellipse(point, width, height):
    point_in_func = point[0]**2/(width/2)**2 + point[1]**2/(height/2)**2
    return point_in_func <= 1

def enough_space(points, new_point, min_dist):
    res = True
    for i in points:
        if np.sqrt((i[0]+new_point[0])**2 + (i[1]+new_point[1])**2) < min_dist:
            res = False
    return res

def all_nodes_evenly_in_rectangle(nodes, width_factor, height_factor, midpoints, name, color_options):
    res = ""
    points = []
    for i in nodes:
        x = np.random.uniform(low = midpoints[0]-(width_factor/2), high = midpoints[0] + (width_factor/2))
        y = np.random.uniform(low = midpoints[1]-(height_factor/2) , high = midpoints[1] + (height_factor/2))

        while not enough_space(points, [x, y], 0.5):
            x = np.random.uniform(low = midpoints[0]-(width_factor/2), high = midpoints[0] + (width_factor/2))
            y = np.random.uniform(low = midpoints[1]-(height_factor/2) , high = midpoints[1] + (height_factor/2))

        points.append([x,y])
        res += f"({x}, {y}) node{color_options} ({i}) {{{name}}}\n"

    return res





def create_nodes(nodes):
    res = ""
    textheight = 20.0
    textwidth = 14.0
    res += f"({textwidth/2 - 4}, {textheight}) node[in front of path,circle,fill=red!50] ({nodes['or'][0]}) {{or}}\n"



    res += all_nodes_evenly_in_rectangle(nodes['orn'], textwidth, 10, (textwidth/2-4,textheight/2+4), 'orn', '[in front of path,circle,fill=yellow!50]')
    res += all_nodes_evenly_in_rectangle(nodes['pn'], textwidth/2-2, 2, (textwidth/4-4, 2), 'pn', '[in front of path,circle,fill=green!50]')
    res += all_nodes_evenly_in_rectangle(nodes['ln'], textwidth/2, 4, (textwidth/4+4,2), 'ln', '[in front of path,circle,fill=blue!50]')
    return res

def create_edges(nodes, connectivity, offset):
    res = ""
    list_nodes = []


    for i in nodes:
        list_nodes += nodes[i]
    print(np.unique([[x['pre_population'], x['post_population']] for x in connectivity]))
    for i in connectivity:
        if i['pre_id'] + offset[i['pre_population']] in list_nodes and i['post_id'] + offset[i['post_population']] in list_nodes:
            res += f"\path [->] ({i['pre_id'] + offset[i['pre_population']]}) edge node {{}} ({i['post_id'] + offset[i['post_population']]});\n"
    return res





def get_glomerulus(network, glomerulus_idx):
    offset = {}
    glomerulus_offset = glomerulus_idx*network.neuron_populations['or'].size()
    last_offset = glomerulus_offset
    for i in network.neuron_populations:
        offset[i] = last_offset
        last_offset += network.neuron_populations[i].size()

    nodes = {}


    for i in offset:
        nodes[i] = [x + offset[i] for x in range(network.neuron_populations[i].size()//network.neuron_populations['or'].size())]

    connectivity = network.get_connectivity()


    with open('output.tex', 'w') as f:
        print("""\documentclass{report}
            \\usepackage{tikz}
            \\usepackage{subcaption}
            \\pagenumbering{gobble}

            \\begin{document}
            \\begin{figure}
              \\begin{tikzpicture}
                  \draw
                    """, file = f)
        print(create_nodes(nodes) + ';', file = f)
        print(create_edges(nodes, connectivity, offset), file = f)
        print("""
        \\end{tikzpicture}
        \\end{figure}
        \\end{document}""", file = f)







def create_graph(network, glomerulus_idx, outname):
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

    order = {'or' : 0, 'orn' : 1, 'ln' : 2, 'pn' : 2 }

    for i in offset:
        nodes[i] = [(x + offset[i], {'column [z]' : order[i], 'pop' : i} ) for x in range(network.neuron_populations[i].size()//network.neuron_populations['or'].size())]


    for i in offset:
        res.add_nodes_from(nodes[i])

    for i in connectivity:
        if i['pre_id'] + offset[i['pre_population']] in res.nodes and i['post_id'] + offset[i['post_population']] in res.nodes:
            res.add_edge(i['pre_id']+offset[i['pre_population']], i['post_id']+offset[i['post_population']])

    print("Writing")
    write_dot(res, outname)
