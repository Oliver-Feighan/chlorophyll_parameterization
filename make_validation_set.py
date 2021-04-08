import json
import random
import pprint

data = open("tddft_data/tddft_data.json")

data = json.load(data)

points = list(data.keys())

validation_set = random.sample(points, k=int(len(points)/2))

pp = pprint.PrettyPrinter(width=128, compact=True)

pp.pprint(validation_set)
