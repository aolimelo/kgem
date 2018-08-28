import numpy as np
from argparse import ArgumentParser
from scipy.sparse import coo_matrix
import os
import unicodedata


def load_entities_dict(inputDir):
    dataset = np.load(inputDir)
    return dataset["entities_dict"].item() if "entities_dict" in dataset else None


def load_relations_dict(inputDir):
    dataset = np.load(inputDir)
    return dataset["relations_dict"].item() if "relations_dict" in dataset else None


def loadGraphNpz(inputDir):
    return np.load(inputDir)["data"]


def split_coo(X, test_valid_size=0.1):
    dataset_size = len(X.row)
    test_size = valid_size = int(dataset_size * test_valid_size)
    np.random.seed(0)

    split_indices = np.zeros(dataset_size)
    split_indices[0:test_size] = np.ones((test_size))
    split_indices[test_size:test_size + valid_size] = np.full((valid_size), 2)
    np.random.shuffle(split_indices)

    train_indices = np.where(split_indices == 0)[0]
    test_indices = np.where(split_indices == 1)[0]
    valid_indices = np.where(split_indices == 2)[0]

    train = coo_matrix((X.data[train_indices], (X.row[train_indices], X.col[train_indices])), shape=X.shape)
    test = coo_matrix((X.data[test_indices], (X.row[test_indices], X.col[test_indices])), shape=X.shape)
    valid = coo_matrix((X.data[valid_indices], (X.row[valid_indices], X.col[valid_indices])), shape=X.shape)

    return train, test, valid


if __name__ == '__main__':
    parser = ArgumentParser("Converts npz tensor to txt format")
    parser.add_argument("input", type=str, help="path to npz file")
    parser.add_argument("output", type=str, help="path to directory with the txt files")

    args = parser.parse_args()

    train_path = os.path.join(args.output, "train.txt")
    test_path = os.path.join(args.output, "test.txt")
    valid_path = os.path.join(args.output, "valid.txt")
    entity2id_path = os.path.join(args.output, "entity2id.txt")
    relation2id_path = os.path.join(args.output, "relation2id.txt")

    try:
        os.stat(args.output)
    except:
        print("creating %s since it does not exist"%args.output)
        os.mkdir(args.output)

    X = loadGraphNpz(args.input)
    ents_dict = {k:v for v,k in load_entities_dict(args.input).items()}
    rels_dict = {k:v for v,k in load_relations_dict(args.input).items()}

    ents,rels = [], []
    for i,k in enumerate(sorted(ents_dict.keys())):
        if i!=k:
            raise Exception("entitiy id %d missing"%i)
        ents.append(unicodedata.normalize("NFKD",ents_dict[k]).encode("ascii","ignore"))
    for i,k in enumerate(sorted(rels_dict.keys())):
        if i!=k:
            raise Exception("relation id %d missing"%i)
        rels.append(unicodedata.normalize("NFKD",rels_dict[k]).encode("ascii","ignore"))


    with open(entity2id_path,"wb") as f:
        for i,ent in enumerate(ents):
            f.write(ent+"\t"+str(i)+"\n")
    with open(relation2id_path,"wb") as f:
        for i,rel in enumerate(rels):
            f.write(rel+"\t"+str(i)+"\n")

    print("Splitting train/test/valid")
    X_train, X_test, X_valid = [],[],[]
    for p in xrange(len(X)):
        X_p_train, X_p_test, X_p_valid = split_coo(X[p])
        X_train.append(X_p_train)
        X_test.append(X_p_test)
        X_valid.append(X_p_valid)

    with open(train_path,"wb") as f:
        for p in xrange(len(X_train)):
            for i in xrange(X_train[p].nnz):
                s = X_train[p].row[i]
                o = X_train[p].col[i]
                f.write(ents[s]+"\t"+ents[o]+"\t"+rels[p]+"\n")

    with open(test_path,"wb") as f:
        for p in xrange(len(X_test)):
            for i in xrange(X_test[p].nnz):
                s = X_test[p].row[i]
                o = X_test[p].col[i]
                f.write(ents[s]+"\t"+ents[o]+"\t"+rels[p]+"\n")

    with open(valid_path,"wb") as f:
        for p in xrange(len(X_valid)):
            for i in xrange(X_valid[p].nnz):
                s = X_valid[p].row[i]
                o = X_valid[p].col[i]
                f.write(ents[s]+"\t"+ents[o]+"\t"+rels[p]+"\n")





