import pandas as pd
from glob import glob
import argparse


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='validates a CNN at the haul out level')
    parser.add_argument('filename', type=str, help='Ouput filename')
    args = parser.parse_args()
    predictions = glob('*.csv')
    
    aggregate_pred = pd.DataFrame(columns=['predictions','filenames'])
    
    for pred in predictions:
        temp = pd.DataFrame.from_csv(pred,index_col=None)
        aggregate_pred = aggregate_pred.append(temp)
        
    aggregate_pred.reset_index(inplace=True,drop=['index'])
    aggregate_pred.to_csv('%s'% args.filename,index=False)
    
    
