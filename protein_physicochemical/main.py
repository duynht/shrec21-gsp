import torch
import gc
import torch_geometric.data as tgd
import torch_geometric.transforms as tgt
from torch.nn import functional as F
import random
import argparse
import configparser

from dataset.protein import InMemoryProteinSurfaceDataset, ProteinSurfaceDataset
from models.chebynet.chebynet import ChebyNet

from sklearn.model_selection import train_test_split 
import os.path as osp
import pandas as pd 

@torch.no_grad()
def test(model, loader, args):
    model.eval()
    correct = 0
    loss = 0.
    criterion = torch.nn.CrossEntropyLoss()
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.argmax(dim=1)
        batch_loss = criterion(out, data.y - 1)
        correct += int((pred == data.y - 1).sum())
        # loss += F.nll_loss(out,data.y,reduction='sum').item()
        loss += batch_loss
    return correct / len(loader.dataset), loss / len(loader.dataset)

def main():
    root = '/home/nhtduy/SHREC21/protein-physicochemical/DATASET/'


    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=167,
                        help='seed')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='weight decay')
    parser.add_argument('--nhid', type=int, default=256,
                        help='hidden size')
    parser.add_argument('--pooling-ratio', type=float, default=0.5,
                        help='pooling ratio')
    parser.add_argument('--dropout-ratio', type=float, default=0.5,
                        help='dropout ratio')
    parser.add_argument('--epochs', type=int, default=100000,
                        help='maximum number of epochs')
    parser.add_argument('--patience', type=int, default=500,
                        help='patience for earlystopping')
    # parser.add_argument('--num-examples', type=int, default=-1,
    #                     help='number of examples, all examples by default')
    # parser.add_argument('--meshes-to-points', type=int, default=0,
    #                     help='convert the initial meshes to points cloud')
    # parser.add_argument('--face-to-edge', type=int, default=1,
    #                     help='convert the faces to edge index')
    parser.add_argument('--model', default="gnn",
                        help='main model')
    # parser.add_argument('--layer', default="gnn",
    #                     help='layer to use if you are using simple_edge_conv or edge_conv')
    parser.add_argument('--set-x', default=1, type=int,
                        help='set x features during data processing')
    # parser.add_argument("--num-instances", type=int, default=-1,
    #                     help="Number of instances per class")
    # parser.add_argument("--num-sample-points", type=int, default=-1,
    #                     help="Number of points to sample when convert from meshes to points cloud")
    parser.add_argument("--load-latest", action="store_true",
                        help="Load the latest checkpoint")
    parser.add_argument("--num-classes", type=int, default=144,
                        help="Number of classes")
    # parser.add_argument("--random-rotate", action="store_true",
    #                     help="Use random rotate for data augmentation")
    # parser.add_argument("--k", type=int, default=16,
    #                     help="Number of nearest neighbors for constructing knn graph")
    parser.add_argument("--in-memory-dataset", action="store_true",
                        help="Load the whole dataset into memory (faster but use more memory)")
    parser.add_argument('--use-txt', action="store_true",
                        help='whether to use physicochemical information')
    parser.add_argument('--graph_pool', type=str, default='mean',
                        help='Global node pooling method')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--cheb_poly', type=int, default=5,
                        help='Chebyshev polynomial order')
    parser.add_argument('--num_conv', type=int, default=6,
                        help='Number of convolutional layers')

    args = parser.parse_args()
    random.seed(args.seed)

    config = configparser.ConfigParser()
    config.read("config.ini")
    config_paths = config["PATHS"]
    base_path = config_paths["base_path"]
    classes_path = base_path + config_paths["classes_path"]
    off_train_folder_path = base_path + config_paths["off_train_folder_path"]
    txt_train_folder_path = base_path + config_paths["txt_train_folder_path"]
    off_final_test_folder_path = base_path + config_paths["off_final_test_folder_path"]
    txt_final_test_folder_path = base_path + config_paths["txt_final_test_folder_path"]
    configuration = "off"
    if args.use_txt:
        configuration += "-txt"

    #TODO: currently fixed ratio
    df = pd.read_csv(osp.join(base_path, classes_path))
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

    size_sr = train_df.pivot_table(index=['label'], aggfunc='size')
    total_size = size_sr.sum()
    class_weights = size_sr.to_numpy() / total_size

    transform_list = [
        tgt.FaceToEdge(True),
        tgt.Distance()
    ]
    transforms = tgt.Compose(transform_list)

    if args.in_memory_dataset:
        DatasetType = InMemoryProteinSurfaceDataset
    else:
        DatasetType = ProteinSurfaceDataset

    print("Preparing data...")

    dataset_path = osp.join(base_path, '-'.join([off_train_folder_path,'full-with-txt']))
    print(f"Dataset path: {dataset_path}")
    
    train_off_dataset = DatasetType(dataset_path, 
                                    train_df.to_records(index=False), 
                                    off_train_folder_path, 
                                    txt_train_folder_path, 
                                    args, 
                                    "train", 
                                    transform=transforms)
    val_off_dataset = DatasetType(dataset_path, 
                                val_df.to_records(index=False), 
                                off_train_folder_path, 
                                txt_train_folder_path, 
                                args, "val", transform=transforms)

    test_off_dataset = DatasetType(dataset_path, 
                                test_df.to_records(index=False),
                                off_train_folder_path,
                                txt_train_folder_path,
                                args, "test", transform=transforms)
    
    train_off_loader = tgd.DataLoader(train_off_dataset, batch_size=args.batch_size, shuffle=True)
    val_off_loader = tgd.DataLoader(val_off_dataset, batch_size=args.batch_size, shuffle=True)
    test_off_loader = tgd.DataLoader(test_off_dataset, batch_size=args.batch_size, shuffle=True)

    print("Loaders are ready")

    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(args.device)

    if args.use_txt:
        args.num_features = 6
    else:
        args.num_features = 3
    print("Number of features dimension:", args.num_features)
    print("Number of classes:", args.num_classes)

    print(args)

    # if args.model == "pointnet":
    #     model = PointNet(args).to(args.device)
    # elif args.model == "simple_edge_conv":
    #     model = SimpleEdgeConvModel(args).to(args.device)
    # elif args.model == "edge_conv":
    #     model = EdgeConvModel(args).to(args.device)
    # else:
    #     model = GNN(args).to(args.device)
    
    # print(model)

    if args.model == 'chebynet':
        model = ChebyNet(
            node_input_dim = args.num_features,
            node_hidden_dim = args.nhid,
            output_dim = args.num_classes,
            polynomial_order=args.cheb_poly,
            num_step_prop=args.num_conv,
            dropout_rate=args.dropout_ratio,
            graph_pool = args.graph_pool
        ).to(args.device)
    else:
        raise NotImplementedError("Model not available")
    
    model_save_path = osp.join(config_paths['model_dir'], f'{args.model}-{configuration}-{"inmem" if args.in_memory_dataset else "outmem"}-nhid={args.nhid}-latest.pth')

    if args.load_latest:
        model.load_state_dict(torch.load(model_save_path))
        val_acc, val_loss = test(model, val_off_loader, args)
        print("Validation loss: {}\taccuracy:{}".format(val_loss, val_acc))
        test_acc, test_loss = test(model, test_off_loader, args)
        print("Test loss:{}\taccuracy:{}".format(test_loss, test_acc))
        torch.save(model.state_dict(), model_save_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(args.device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    torch.cuda.empty_cache()
    gc.collect()

    min_loss = 1e10
    patience = 0
    epoch = 0
    for epoch in range(args.epochs):
        model.train()
        training_loss = 0
        for i, data in enumerate(train_off_loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            target = data.y.long()
            out = model(data)
            loss = criterion(out, target - 1)
            training_loss += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        training_loss /= len(train_off_loader.dataset)
        print("Training loss: {}".format(training_loss))
        val_acc, val_loss = test(model, val_off_loader, args)
        print("Validation loss: {}\taccuracy:{}".format(val_loss, val_acc))
        if val_loss < min_loss:
            torch.save(model.state_dict(), model_save_path)
            print("Model saved at epoch{}".format(epoch))
            min_loss = val_loss
            patience = 0
        else:
            patience += 1
        if patience > args.patience:
            break 

    if epoch:
        print("Last epoch before stopping:", epoch)

    test_acc, test_loss = test(model, test_off_loader, args)
    print("Test loss:{}\taccuracy:{}".format(test_loss, test_acc))

if __name__ == "__main__":
    main()