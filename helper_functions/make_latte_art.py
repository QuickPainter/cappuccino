# tasseography is the art of reading the coffee grounds --> in this case plotting the candidates and examining them by eye

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import h5py



def main(csv_file):
    table = pd.read_csv(csv_file)


def plot_candidate():
    pass

if __name__ == '__main__':
    csv_file_path= sys.argv[1]
    main(csv_file_path)
