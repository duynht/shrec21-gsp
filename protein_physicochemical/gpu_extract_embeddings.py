import argparse
import ast
import dask
__spec__ = None

import os.path as osp

from embeddings import gpu_emb_extract

# def dask_wrapper(raw_dir, algo, file_dict, graph_list):
#     print('_'.join([dir,'embeddings', algo]))
#     try:
#         generate_graph_embeddings(raw_dir, algo, file_dict, graph_list)
#     except Exception as e:
#         print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',
                        default='/home/nhtduy/SHREC21/protein-physicochemical/DATASET',
                        type=str,
                        help='root path of data directories')
    parser.add_argument('--dir_list', 
                        default=None,
                        type=str,
                        help='data directories to be extracted')
    parser.add_argument('--algo_list',
                        default="['fgsd', 'sf', 'netlsd', 'geoscattering', 'ige']",
                        type=str,
                        help='list of embedding algos')

    args = parser.parse_args()

    algo_list = ast.literal_eval(args.algo_list)

    if not args.dir_list:
        dir_list = [
            'OFF_training_anonym_v2',
            'OFF_test_anonym_v2',
            'OFF_train_preprocess',
            'OFF_test_preprocess'
        ]
    else:
        dir_list = ast.literal_eval(args.dir_list)

    root = args.data_root

    dask.config.set(scheduler='processes')
    works = []

    for dir in dir_list:
        raw_dir = osp.join(root, dir)
        print(dir)
        gpu_emb_extract.extract_dir(raw_dir, algo_list)
        # file_dict, graph_list = read_dataset(raw_dir)
        # for algo in algo_list:
            # works.append(dask.delayed(dask_wrapper)(raw_dir, algo, file_dict, graph_list))
            # dask_wrapper(raw_dir, algo, file_dict, graph_list)
            # emb_extract.extract_dir(raw_dir, algo)

    # dask.compute(*works)
