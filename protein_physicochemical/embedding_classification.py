from sys import stdin, stdout
from sklearn.model_selection import train_test_split 
from utils import cla
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os.path as osp
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/nhtduy/SHREC21/protein-physicochemical/DATASET/',
                        help='Root data path')
    parser.add_argument('--generate_csv', action='store_true',
                        help='Generate the CSV from classTraining.cla')
    parser.add_argument('--emb_dir', type=str,
                        help='Path to embeddings')
    parser.add_argument('--submit', action='store_true')

    args = parser.parse_args()

    if not args.submit:
        if args.generate_csv:
            df = read_labels(osp.join(args.root, 'classTraining.cla'), write_file=True)
        else:
            df = pd.read_csv(osp.join(args.root, 'classTraining.csv'))

        df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
        # truth_dict = {str(key): value for key, value in truth_df.to_records(index=False)}

        # size_sr = train_df.pivot_table(index=['label'], aggfunc='size')
        # # size_dict = {str(key): value for key, value in size_sr.items()}
        # total_size = size_sr.sum()
        # weights = size_sr.to_numpy() / total_size

        df_train.X = df_train.off_file.apply(lambda f: np.load(osp.join(args.root, args.emb_dir, f'{f}.npy')))
        df_test.X = df_test.off_file.apply(lambda f: np.load(osp.join(args.root, args.emb_dir, f'{f}.npy')))

        # for filepath in sorted(glob(emb_dir + '/*', recursive=True)):
        #     if os.path.isdir(filepath): continue
        #     X.append(np.load(filepath))
        #     filename = osp.basename(filepath).split('.')[0]
        #     y.append(truth_dict[filename])

        X_train = np.array([x for _, x in df_train.X.items()]).squeeze()
        X_test = np.array([x for _, x in df_test.X.items()]).squeeze()

        y_train = np.array([x for _, x in df_train.label.items()]).squeeze()
        y_test = np.array([x for _, x in df_test.label.items()]).squeeze()

        model = SVC(kernel='sigmoid', class_weight='balanced')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(accuracy_score(y_test, y_pred))



