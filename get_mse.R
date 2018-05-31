# Script to plot a confusion matrix and get refined stats for seal detection models

# load packages
library(ggplot2)
library(reshape2)
library(scales)
library(glue)
library(argparse)

# define arg-parser 
parser = ArgumentParser(description="R script to get validation stats for counting CNNs")
parser$add_argument("--input_file", type="character", help="model name generated during training")
parser$add_argument("--pipeline", type="character", help='model pipeline')
parser$add_argument("--dest_folder", type="character", default="saved_models", help="folder where pipeline is located")
args = parser$parse_args()


# plot confusion matrix
get_mse = function(csv_file){

    # read validation output from validate 
    rows = read.csv(csv_file, stringsAsFactors=FALSE)
    
    # get running time and parameters (last row)
    total_params = as.integer(rows[dim(rows)[1], 1])
    running_time = as.double(rows[dim(rows)[1], 2])
    
    # remove last row
    rows = rows[-dim(rows)[1],]
    
    # get mse
    mse = mean(apply(rows, 1, function(x) (x[1] - x[2])^2))
    
    # get false precision and recall
    TP = 0
    FP = 0
    FN = 0
    for(i in 1:dim(rows)[1]){
        TP = TP + min(rows[i, ])
        difference = rows[i, 'ground_truth'] - rows[i, 'predicted']
        FP = FP + max(0, difference * -1)
        FN = FN + max(0, difference)
    }
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    # create DataFrame for MSE
    mse = data.frame('n_parameters'=total_params, 'running_time'=running_time, 'MSE'=mse, 'model_name'=model_name,
                     'precision'=precision, 'recall'=recall, 'total_predicted'=sum(rows[,'predicted']), 
                     'total_ground_truth'=sum(rows[,'ground_truth']))
    
    write.csv(mse, glue("./{dest_folder}/{pipeline}/{model_name}/{model_name}_mse.csv"))
}



# run for validation data
model_name = args$input_file
pipeline = args$pipeline
dest_folder = args$dest_folder
get_mse(csv_file=glue('./{dest_folder}/{pipeline}/{model_name}/{model_name}_validation.csv'))




