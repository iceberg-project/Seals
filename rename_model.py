import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='predict new images using a previously trained model')
parser.add_argument('--folder', type=str)
parser.add_argument('--target_name', type=str)

args = parser.parse_args()
target = args.target_name
folder = args.folder


def main():
    csv_files = [filename for filename in os.listdir(folder) if filename.endswith('.csv')]
    for filename in csv_files:
        path = os.path.join(folder, filename)
        target_name = os.path.join(folder, target) + '.csv'
        df = pd.read_csv(path)
        df['model_name'] = target * len(df)
        df.to_csv(target_name, index=False)


if __name__ == '__main__':
    main()
