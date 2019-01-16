"""
Test sealnet
==========================================================

Testing script for ICEBERG seals use case. Compares merged predicted shapefile from 'merge_shapefiles.py' with
ground-truth shapefile, retrieving test precision and recall for a given model.

Author: Bento Goncalves
License: MIT
Copyright: 2018-2019
"""
import geopandas as gpd
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='compares model results to groundtruth to extract precision and recall')
parser.add_argument('--shp_prediction', type=str, help='path to shapefile with model predictions')
parser.add_argument('--shp_ground_truth', type=str, help='path to shapefile with ground-truths')
parser.add_argument('--tolerance', type=float, help='maximum distance to consider two points a match')
parser.add_argument('--output_file', type=str, help='desired name of output file')


def test_cnn(shp_pred: str, shp_gt: str, tolerance: float, output: str):
    """
    :param shp_pred: path to prediction shapefile
    :param shp_gt: path to ground-truth shapefile
    :param tolerance: maximum distance in pixels to consider two points a match
    :param output: path to output file
    :return: creates a .csv file with test statistics
    """

    # read csvs
    preds = gpd.read_file(shp_pred)
    ground_truth = gpd.read_file(shp_gt)

    # store results
    stats = pd.DataFrame()

    # loop over scenes
    for scene in pd.unique(ground_truth.scene):
        # store stats
        true_positives = 0

        # restrict databases
        preds_scene = preds.loc[preds.scene == scene]
        ground_truth_scene = ground_truth.loc[ground_truth.scene == scene]

        # check for matches
        matched = set([])
        for gt_point in ground_truth_scene.iterrows():
            filtered = preds_scene.iloc[list(set([idx for idx in range(len(preds_scene))]) - matched)]
            for idx, pred_point in enumerate(filtered.iterrows()):
                # get euclidean distance
                if gt_point[1]['geometry'].distance(pred_point[1]['geometry']) < tolerance:
                    matched.add(idx)
                    true_positives += 1
                    continue

        # add false positives
        false_negatives = len(ground_truth_scene) - len(matched)
        false_positives = len(preds_scene) - len(matched)

        # store precision and recall
        stats = stats.append({'precision': true_positives / max(true_positives + false_positives, 1),
                              'recall': true_positives / max(true_positives + false_negatives, 1),
                              'scene': scene,
                              'ground-truth_count': len(ground_truth_scene),
                              'predicted_count': len(preds_scene)}, ignore_index=True)
    stats.to_csv(output)


def main():
    args = parser.parse_args()
    test_cnn(shp_pred=args.shp_prediction,
             shp_gt=args.shp_ground_truth,
             tolerance=args.tolerance,
             output=args.output_file)


if __name__ == '__main__':
    main()
