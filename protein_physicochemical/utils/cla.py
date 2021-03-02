import pandas as pd
import os.path as osp

def read_labels(filepath, write_file=False):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    num_classes, num_sample = tuple(lines[1].split())
    num_classes, num_sample = int(num_classes), int(num_sample)
    lines = lines[2:]

    n = 0
    label = 0
    curr = 0

    data = []

    for line in lines:
        if not line or line == '\n': 
            continue
        elems = line.split()
        if len(elems) == 3:
            assert curr == n
            curr = 0
            label += 1
            n = int(elems[-1])
        else:
            curr += 1
            data.append((int(elems[0]), label))

    assert num_classes == label

    df = pd.DataFrame.from_records(data, columns=['off_file', 'label'])
    if write_file:
        df.to_csv('.'.join([filepath.split('.')[-2], 'csv']), index=False)
    return df

if __name__ == "__main__":
    df = read_labels('/home/nhtduy/SHREC21/protein-physicochemical/DATASET/classTraining.cla', write_file=True)


