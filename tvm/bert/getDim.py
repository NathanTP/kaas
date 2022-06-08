import json
def getDim(index):
    f = open('bert/bert_graph.json', 'r')
    graph = json.load(f)
    dim = graph["attrs"]["shape"][1][index]
    storage_id = graph["attrs"]["storage_id"][1][index]
    f.close()
    return dim, storage_id


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=int)

    args = parser.parse_args()
    print(getDim(args.index))
