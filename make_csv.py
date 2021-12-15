import pandas as pd
import os

def main():
    root = 'PokemonData'
    dirs = os.listdir(root)

    to_id = pd.read_csv('map.csv').set_index("Name").to_dict()

    data = []
    for d in dirs:
        if d != ".DS_Store":
            subdir = os.path.join(root, d)
            imgs = os.listdir(subdir)

            for i in imgs:
                img_path = os.path.join(subdir, i)
                data.append({'path': img_path, 'label_full': d, 'label': to_id['#'][d]})

    data = pd.DataFrame(data)
    data.to_csv('data.csv')


if __name__ == "__main__":
    main()
