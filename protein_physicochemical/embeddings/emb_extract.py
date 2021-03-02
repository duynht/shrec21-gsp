import karateclub as kc
import trimesh as tm
import networkx as nx
from torch_geometric import utils as tgu
from torch_geometric import transforms as tgt
from torch_cluster import knn_graph
import os
import os.path as osp
import numpy as np
from glob import glob

from utils import off
import embeddings.gpu_karateclub as gkc


def tg_off2networkx(filepath, k=16):
    mesh = off.read_off(filepath)
    # mesh.edge_index = knn_graph(mesh.pos, k=k)
    mesh = tgt.FaceToEdge()(mesh)
    mesh = tgu.to_networkx(mesh, to_undirected=True)

    return [nx.convert_node_labels_to_integers(mesh.subgraph(c).copy()) for c in nx.connected_components(mesh)]

def trimesh_off2networkx(filepath):
    mesh = off.read_off(filepath)
    mesh = tgu.to_trimesh(mesh)
    mesh = mesh.vertex_adjacency_graph

    return [nx.convert_node_labels_to_integers(mesh.subgraph(c).copy()) for c in nx.connected_components(mesh)]


def dask_read_dataset(raw_dir):
    graph_list = []
    subgraph_list = []
    file_dict = {}

    for filepath in sorted(glob(raw_dir + '/*', recursive=True)):
        if os.path.isdir(filepath): continue
        subgraph_list.append(dask.delayed(trimesh_off2networkx)(filepath))

    subgraph_list = dask.compute(*subgraph_list)

    for i, filepath in enumerate(sorted(glob(raw_dir + '/*', recursive=True))):
        if os.path.isdir(filepath): continue
        filename = osp.basename(filepath).split('.')[0]
        graph_list.extend(subgraph_list[i])
        file_dict[filename] = (len(graph_list) - len(subgraph_list[i]), len(graph_list))

    return file_dict, graph_list

def _read_dataset(raw_dir):
    graph_list = []
    file_dict = {}

    for filepath in sorted(glob(raw_dir + '/*', recursive=True)):
        if os.path.isdir(filepath): continue
        filename = osp.basename(filepath).split('.')[0]
        subgraphs = trimesh_off2networkx(filepath)
        graph_list.extend(subgraphs)
        file_dict[filename] = (len(graph_list) - len(subgraphs), len(graph_list))

    print('finished read graphs')

    return file_dict, graph_list

def _generate_graph_embeddings(raw_dir, algo, file_dict, graph_list):
    algo = algo.lower()

    if algo == 'ige':
        model = kc.IGE()
    elif algo == 'geoscattering':
        model = kc.GeoScattering()
    elif algo == 'gl2vec':
        model = kc.GL2Vec()
    elif algo == 'netlsd':
        model = kc.NetLSD()
    elif algo == 'sf':
        model = kc.SF()
    elif algo == 'fgsd':
        model = kc.FGSD()
    elif algo == 'graphwave':
        model = kc.GraphWave()
    elif algo == 'feathergraph':
        model = kc.FeatherGraph()
    elif algo == 'gpu_geo':
        model = gkc.GeoScattering()
    elif algo == 'gpu_fgsd':
        model = gkc.FGSD()
    elif algo == 'gpu_netlsd':
        model = gkc.NetLSD()
    else:
        raise NotImplementedError("Unknown graph embedding")

    output_dir = '_'.join([raw_dir,'embeddings', algo])

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    
    model.fit(graph_list)
    emb = model.get_embedding()

    print('finished embeddings')

    agg_emb = []
    for filename, (start, end) in file_dict.items():
        total_size = sum([len(c) for c in graph_list[start:end]])
        weights = np.array([len(c)/total_size for c in graph_list[start:end]])
        agg_emb.append(np.average(emb[start:end], weights=weights, axis=0))

    print("finished aggregating embeddings")

    for i, filename in enumerate(file_dict.keys()):
        np.save(osp.join(output_dir, '.'.join([filename,'npy'])), emb[i])

    return #list(file_dict.keys()), np.array(new_emb)

def extract_dir(raw_dir, algo_list):
    model_list = []
    for algo in algo_list:
        algo = algo.lower()

        if algo == 'ige':
            model = kc.IGE()
        elif algo == 'geoscattering':
            model = kc.GeoScattering()
        elif algo == 'gl2vec':
            model = kc.GL2Vec()
        elif algo == 'netlsd':
            model = kc.NetLSD()
        elif algo == 'sf':
            model = kc.SF()
        elif algo == 'fgsd':
            model = kc.FGSD()
        elif algo == 'graphwave':
            model = kc.GraphWave()    
        elif algo == 'feathergraph':
            model = kc.FeatherGraph()    
        elif algo == 'gpu_geo':
            model = gkc.GeoScattering()
        elif algo == 'gpu_fgsd':
            model = gkc.FGSD()
        elif algo == 'gpu_netlsd':
            model = gkc.NetLSD()
        else:
            raise NotImplementedError("Unknown graph embedding")
        
        model_list.append(('_'.join([raw_dir,'embeddings', algo]), model))
        # output_dir_list.append('_'.join([raw_dir,'embeddings', algo]))

    for filepath in sorted(glob(raw_dir + '/*', recursive=True)):
        if os.path.isdir(filepath): continue

        filename = osp.basename(filepath).split('.')[0]

        subgraphs = trimesh_off2networkx(filepath)

        for output_dir, model in model_list:
            if not osp.exists(output_dir):
                os.makedirs(output_dir)    
                
            # model.fit(subgraphs)
            # emb = model.get_embedding()

            # total_size = sum([len(c) for c in subgraphs])
            # weights = np.array([len(c)/total_size for c in subgraphs])
            # emb = np.average(emb, weights=weights, axis=0)

            model.fit([max(subgraphs, key=len)])
            emb = model.get_embedding()

            out_filepath = osp.join(output_dir, '.'.join([filename,'npy']))
            np.save(out_filepath, emb)