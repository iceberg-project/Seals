import pandas as pd
import time
import argparse
import os

parser = argparse.ArgumentParser(description='validates a CNN at the scene level')

parser.add_argument('--model_name', type=str, help='name of input model file from training, this name will also be used'
                                                   'in subsequent steps of the pipeline')

args = parser.parse_args()


def main():
    # accumulate stats
    prec_recall_model = pd.DataFrame()
    # get model name
    model_name = args.model_name
    # find scene banks
    scene_banks = [ele for ele in os.listdir('./training_sets') if 'scene_bank' in ele]
    # loop over scene banks
    for bank in scene_banks:
        scene_bank = pd.read_csv('./training_sets/{}'.format(bank))
        # store model stats
        model_stats = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        positive_classes = [label for label in scene_bank.columns if label not in ['scene', 'present']]
        # load classifications
        val_scene_model = pd.read_csv('./saved_models/haulout/{}/{}_scene_val.csv'.format(model_name, model_name))
        # find scenes
        scenes = os.listdir('./tiled_images')
        # loop through scenes
        for scene in scenes:
            val_scene_labels = pd.unique(val_scene_model.loc[val_scene_model['scene'] == scene, 'label'])
            # record if scene is TP, TN, FP or FN
            detected = False
            for label in positive_classes:
                if label in val_scene_labels:
                    detected = True
            # if scene is occupied
            if int(scene_bank.loc[scene_bank['scene'] == scene]['present']):
                if detected:
                    model_stats['TP'] += 1
                else:
                    model_stats['FN'] += 1
            # if scene is empty
            else:
                if detected:
                    model_stats['FP'] += 1
                else:
                    model_stats['TN'] += 1

        # record precision and recall for model (max to avoid division by zero in case of no true positives)
        precision = model_stats['TP'] / (max([model_stats['TP'], 1E-8]) + model_stats['FP'])
        recall = model_stats['TP'] / (max([model_stats['TP'], 1E-8]) + model_stats['FN'])
        # append precision and recall from scene bank to prec_reacall_model
        prec_recall_model = prec_recall_model.append({'model_name': model_name, 'precision': precision, 'recall': recall,
                                                      'label': '-'.join(positive_classes)}, ignore_index=True)
    # save model stats to csv
    prec_recall_model.to_csv('./saved_models/haulout/{}/{}_scene_prec_recall.csv'.format(model_name, model_name))


if __name__ == '__main__':
    main()
