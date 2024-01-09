import numpy as np
from data_related import get_all_data, load_data, separate_data, normalize_data, split_data, save_data
from visualization import visualize_data


def main():
    # Import all the data
    d = get_all_data(seed=42)
    save_data(d, "datasets/dict_data.pkl")

    # Visualize data
    do_visualize = False
    if do_visualize:
        idx = np.random.randint(0, len(d['x']), size=4, dtype=int)
        visualize_data(d['x'], np.argmax(d['y'], axis=1), idx)


if __name__ == '__main__':
    main()
