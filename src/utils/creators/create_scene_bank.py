import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='creates a scene bank to validate sealnet instances at the scene level')
parser.add_argument('--positive_classes', type=str, help='classes of interest, separated by spaces. The script will'
                                                         'loop through database imagery searching for intances of'
                                                         'the positive classes, if there is at least one element of '
                                                         'those classes in a raster, it will be flagged as a positive')
parser.add_argument('--out_file', type=str, help='name of the csv file where the scene bank will be saved to')


args = parser.parse_args()


def create_scene_bank(shape_file, positive_classes):
    """
    Creates a scene bank to perform sealnet validation at the scene level.

    Input:
        shape_file: path to .csv shape_file with training points with latitude, longitude, classification label and
            source raster layer.
        positive_classes: list of classes that will count as a positive -- scenes will be checked for the presence of
            those classes, assigned a "YES" in case they are present and a "NO" otherwise.

    Output:
        python dictionary listing scenes as positive or negative and storing the count of elements in each positive
        class inside scenes.
    """
    # read shapefile with seal points as pandas dataframe
    shp_file = pd.read_csv(shape_file)

    # get a list of scenes inside shapefile
    scenes = pd.unique(shp_file['scene'])

    # create an empty dictionary to store scene_bank
    scene_bank = {}

    # iterate over scenes inside shapefile looking for seals, saving seal counts
    for idx, scene in enumerate(scenes):
        scene_data = shp_file.loc[shp_file['scene'] == scene]

        labels = {label: sum(scene_data['label'] == label) for label in positive_classes}
        scn = {'scene': scene}
        scene_bank[idx] = {**scn, **labels}
        if sum(labels.values()) > 0:
            scene_bank[idx]['present'] = 1
        else:
            scene_bank[idx]['present'] = 0

    return pd.DataFrame(scene_bank)


def main():
    positive_classes = args.positive_classes.split()
    scene_bank = create_scene_bank(shape_file='seal_points_espg3031.csv', positive_classes=positive_classes)
    scene_bank = scene_bank.transpose()
    scene_bank.to_csv('./training_sets/{}'.format(args.out_file), index=False)


if __name__ == '__main__':
    main()
