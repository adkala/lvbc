import matplotlib.pyplot as plt
import numpy as np

def visualize_single_trajectory(y, pred_y, tm, ax=None):
    if ax is None:
        _, ax = plt.subplots() 
    

def visualize_trajectory(config, test=False, max_plots=100, pause=0.1):
    if not config['delta_p']:
        raise NotImplementedError

    datasets = config['train_datasets' if not test else 'test_datasets']
    for _ in range(max_plots):
        i = np.random.randint(len(datasets))
        j = np.random.randint(len(datasets[i]))

        x, y, _ = datasets[i][j]