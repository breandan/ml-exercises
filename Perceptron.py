threshold = 0
RATE = .1

def classify(datapoint, weights):
    prediction = sum(x * y for x, y in zip(datapoint, weights))
    if prediction < 0:
        return 0
    else:
        return 1

def train(data):
    weights = [0] * len(data[0].features)
    total_error = threshold + 1
    while total_error > threshold:
        total_error = 0
        for item in data:
            error = item.label - classify(item.features, weights)
            weights = [w + RATE * error * i for w, i in
                       zip(weights, item.features)]
            total_error += abs(error)
    return weights

class Datum:
    def __init__(self, features, label):
        self.features = [1] + features
        self.label = label

weights = train([Datum([0, 0], 1),
                 Datum([1, 1], 1),
                 Datum([2, 1], 0)])

print("Weights: " + str(weights))

print(classify([0, 0], weights))
print(classify([0, 1], weights))
print(classify([2, 1], weights))
print(classify([3, 1], weights))
print(classify([1, 1], weights))
