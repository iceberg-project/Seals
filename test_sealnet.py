"""
Test sealnet
==========================================================

Testing script for ICEBERG seals use case. Compares merged predicted shapefile from 'merge_shapefiles.py' with
ground-truth shapefile, retrieving test precision and recall for a given model.

Author: Bento Goncalves
License: MIT
Copyright: 2018-2019
"""
import argparse

import geopandas as gpd
import pandas as pd
from fiona.crs import from_epsg
from scipy.sparse.csgraph import connected_components

parser = argparse.ArgumentParser(description='compares model results to groundtruth to extract precision and recall')
parser.add_argument('--shp_prediction', type=str, help='path to shapefile with model predictions')
parser.add_argument('--shp_ground_truth', type=str, help='path to shapefile with ground-truths')
parser.add_argument('--tolerance_seal', type=float, help='maximum distance to consider two points a match')
parser.add_argument('--tolerance_haul', type=float, help='maximum distance to consider two seals members'
                                                         'of the same haul out')
parser.add_argument('--output_file', type=str, help='desired name of output file')


def test_cnn(shp_pred, shp_gt, tolerance_seal, tolerance_haul, output):
    """
    :param shp_pred: path to prediction shapefile
    :param shp_gt: path to ground-truth shapefile
    :param tolerance_seal: maximum distance in meters to consider two points a match
    :param tolerance_haul: maximum distance in meters to consider two seals members of the
    same haul out
    :param output: path to output file
    :return: creates a .csv file with test statistics
    """

    # read csvs
    preds = gpd.read_file(shp_pred)
    ground_truth = gpd.read_file(shp_gt)

    # store results
    stats_seal = pd.DataFrame()
    stats_haul = pd.DataFrame()

    # store haul out shapefile
    hauls_shp = gpd.GeoDataFrame(crs=from_epsg(3031))
    hauls_shp_gt = gpd.GeoDataFrame(crs=from_epsg(3031))

    # loop over scenes common to predicted and ground-truth shapefiles
    for scene in set(preds.scene).intersection(set(ground_truth.scene)):
        # store stats
        true_positives_seal = 0
        true_positives_haul = 0

        # restrict databases
        preds_scene = preds.loc[preds.scene == scene]
        ground_truth_scene = ground_truth.loc[ground_truth.scene == scene]

        # check for individual seal matches
        matched = set([])
        for gt_point in ground_truth_scene.iterrows():
            filtered = preds_scene.iloc[list(set([idx for idx in range(len(preds_scene))]) - matched)]
            for idx, pred_point in enumerate(filtered.iterrows()):
                # get euclidean distance
                if gt_point[1]['geometry'].distance(pred_point[1]['geometry']) < tolerance_seal:
                    matched.add(idx)
                    true_positives_seal += 1
                    continue

        # add false positives
        false_negatives_seal = len(ground_truth_scene) - len(matched)
        false_positives_seal = len(preds_scene) - len(matched)

        buffer_gt = ground_truth_scene.buffer(tolerance_haul)
        buffer_preds = preds_scene.buffer(tolerance_haul)

        # find overlaps
        overlap_matrix_gt = buffer_gt.apply(lambda x: buffer_gt.overlaps(x)).values.astype(int)
        overlap_matrix_preds = buffer_preds.apply(lambda x: buffer_preds.overlaps(x)).values.astype(int)

        # store overlap groups
        n_hauls_gt, groups_gt = connected_components(overlap_matrix_gt)
        n_hauls_preds, groups_preds = connected_components(overlap_matrix_preds)

        # dissolve points that overlap groundtruth and save groundtruth shapefile
        hauls_gt = gpd.GeoDataFrame({'geometry': buffer_gt, 'group': groups_gt}, crs=from_epsg(3031))
        hauls_gt = hauls_gt.dissolve(by='group')
        hauls_gt['haulout_size'] = [sum(groups_gt == grp) for grp in hauls_gt.index]
        hauls_gt['scene'] = [scene] * len(hauls_gt)

        # dissolve points that overlap predicted and save haulout shapefile
        hauls_preds = gpd.GeoDataFrame({'geometry': buffer_preds, 'group': groups_preds}, crs=from_epsg(3031))
        hauls_preds = hauls_preds.dissolve(by='group')
        hauls_preds['haulout_size'] = [sum(groups_preds == grp) for grp in hauls_preds.index]
        hauls_preds['scene'] = [scene] * len(hauls_preds)

        # add haulout shapes to shapefile
        hauls_shp = hauls_shp.append(hauls_preds, ignore_index=True)
        hauls_shp_gt = hauls_shp_gt.append(hauls_gt, ignore_index=True)

        # find overlaps between prediction and groundtruth
        for _, gt_haul in hauls_gt.iterrows():
            for _, pred_haul in hauls_preds.iterrows():
                true_positives_haul += int(gt_haul['geometry'].overlaps(pred_haul['geometry']))
        false_negatives_haul = n_hauls_gt - true_positives_haul
        false_positives_haul = n_hauls_preds - true_positives_haul

        # store precision and recall 
        stats_seal = stats_seal.append({'precision': true_positives_seal / max(true_positives_seal +
                                                                               false_positives_seal, 1),
                                        'recall': true_positives_seal / max(true_positives_seal + false_negatives_seal,
                                                                            1),
                                        'scene': scene,
                                        'ground-truth_count': len(ground_truth_scene),
                                        'predicted_count': len(preds_scene)}, ignore_index=True)

        stats_haul = stats_haul.append({'precision': true_positives_haul / max(true_positives_haul +
                                                                               false_positives_haul, 1),
                                        'recall': true_positives_haul / max(true_positives_haul + false_negatives_haul,
                                                                            1),
                                        'scene': scene,
                                        'ground-truth_count': n_hauls_gt,
                                        'predicted_count': n_hauls_preds}, ignore_index=True)

    # loop through scenes entirely missed by one observer (i.e. model or ground-truth)
    for scene in set(pd.unique(preds.scene)).difference(set(pd.unique(ground_truth.scene))):
        # restrict to scene
        preds_scene = preds.loc[preds.scene == scene]
        ground_truth_scene = ground_truth.loc[ground_truth.scene == scene]

        # present in predictions and absent in groundtruth
        if len(preds_scene) > 0:
            # create buffer
            buffer_preds = preds_scene.buffer(tolerance_haul)

            # find overlaps
            overlap_matrix_preds = buffer_preds.apply(lambda x: buffer_preds.overlaps(x)).values.astype(int)

            # store overlap groups
            n_hauls_preds, groups_preds = connected_components(overlap_matrix_preds)

            # dissolve points that overlap predicted and save haulout shapefile
            hauls_preds = gpd.GeoDataFrame({'geometry': buffer_preds, 'group': groups_preds}, crs=from_epsg(3031))
            hauls_preds = hauls_preds.dissolve(by='group')
            hauls_preds['haulout_size'] = [sum(groups_preds == grp) for grp in hauls_preds.index]
            hauls_preds['scene'] = [scene] * len(hauls_preds)

            # add haulout shapes to shapefile
            hauls_shp = hauls_shp.append(hauls_preds, ignore_index=True)

            # store precision and recall
            stats_seal = stats_seal.append({'precision': 0,
                                            'recall': 0,
                                            'scene': scene,
                                            'ground-truth_count': 0,
                                            'predicted_count': len(preds_scene)}, ignore_index=True)

            stats_haul = stats_haul.append({'precision': 0,
                                            'recall': 0,
                                            'scene': scene,
                                            'ground-truth_count': 0,
                                            'predicted_count': n_hauls_preds}, ignore_index=True)

        # present in ground_truth and absent in predictions
        else:
            # create buffer
            buffer_gt = ground_truth_scene.buffer(tolerance_haul)

            # find overlaps
            overlap_matrix_gt = buffer_gt.apply(lambda x: buffer_gt.overlaps(x)).values.astype(int)

            # store overlap groups
            n_hauls_gt, groups_gt = connected_components(overlap_matrix_gt)
             
            # store haulouts
            hauls_gt = gpd.GeoDataFrame({'geometry': buffer_gt, 'group': groups_gt}, crs=from_epsg(3031))
            hauls_gt = hauls_gt.dissolve(by='group')
            hauls_gt['haulout_size'] = [sum(groups_gt == grp) for grp in hauls_gt.index]
            hauls_gt['scene'] = [scene] * len(hauls_gt)

            # save stats
            stats_seal = stats_seal.append({'precision': 0,
                                            'recall': 0,
                                            'scene': scene,
                                            'ground-truth_count': len(ground_truth_scene),
                                            'predicted_count': 0}, ignore_index=True)

            stats_haul = stats_haul.append({'precision': 0,
                                            'recall': 0,
                                            'scene': scene,
                                            'ground-truth_count': n_hauls_gt,
                                            'predicted_count': 0}, ignore_index=True)

            hauls_shp_gt = hauls_shp_gt.append(hauls_gt, ignore_index=True)

    # save output
    stats_seal.to_csv(f"{output}prec_recall_seal.csv")
    stats_haul.to_csv(f"{output}prec_recall_haul.csv")
    hauls_shp.to_file(f"{output}pred_haulouts.shp")
    hauls_shp_gt.to_file(f"{output}gt_haulouts.shp")


def main():
    args = parser.parse_args()
    test_cnn(shp_pred=args.shp_prediction,
             shp_gt=args.shp_ground_truth,
             tolerance_seal=args.tolerance_seal,
             tolerance_haul=args.tolerance_haul,
             output=args.output_file)


if __name__ == '__main__':
    main()
