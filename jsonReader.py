# read and show a json file

import json

import csv


length = []

def read_json(path):

    with open(path, 'r') as f:
        data = json.load(f)
        for sample in data:
            length.append(sample[1])
                # length.append(len(data[vname][relation][0]['sub'].keys()))

    # save the length list to a .csv file
    with open('length.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(length)
    print(len(length))


if __name__ == "__main__":
    # path = "../ground_data/results/kl-warmup-klrct-vidvrd_batch.json"
    path = "dataset/vidvrd/vrelation_val.json"
    read_json(path)