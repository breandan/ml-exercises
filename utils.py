def pop_all():
    #bring all the figures hiding in the background to the foreground
    all_figures=[manager.canvas.figure for manager in matplotlib.\
        _pylab_helpers.Gcf.get_all_fig_managers()]
    [fig.canvas.manager.show() for fig in all_figures]
    return len(all_figures)

import os
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/breandan/ml-exercises/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + "/housing_train.csv"
train = "housing_train.csv"
test = "housing_test.csv"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    train_path = os.path.join(housing_path, train)
    test_path = os.path.join(housing_path, test)
    urllib.request.urlretrieve(DOWNLOAD_ROOT + train, train_path)
    urllib.request.urlretrieve(DOWNLOAD_ROOT + test, test_path)