# Script to plot a confusion matrix and get refined stats for seal detection models

# load packages
library(ggplot2)
library(reshape2)
library(scales)
library(glue)
library(argparse)

# define arg-parser 
parser = ArgumentParser(description="R script to get benchmarks for counting CNNs")
parser$add_argument("--pipeline", type="character", help='model pipeline')
parser$add_argument("--input_file", type="character", help="model name generated during training")
args = parser$parse_args()


# get benchmarks
generate_benchmark = function(csv_file){
    
    # extract ground-truth
    rows = read.csv(csv_file, stringsAsFactors=FALSE)
    
    # get mean and median 
    mean_count = mean(rows[,'ground_truth'])
    median_count = median(rows[,'ground_truth'])
    
    # get mse
    mse = c()
    mse = c(mse, mean(apply(rows, 1, function(x) (x['ground_truth'])^2)))
    mse = c(mse, mean(apply(rows, 1, function(x) (x['ground_truth'] - mean_count)^2)))
    mse = c(mse, mean(apply(rows, 1, function(x) (x['ground_truth'] - median_count)^2)))
    
    # get false precision and recall
    
    precision = c(0)
    recall = c(0)
    
    for(ele in c(mean_count, median_count)){
        TP = 0
        FP = 0
        FN = 0
        for(i in 1:dim(rows)[1]){
            TP = TP + min(rows[i, ])
            difference = rows[i, 'ground_truth'] - ele
            FP = FP + max(0, difference * -1)
            FN = FN + max(0, difference)
        }
        
        precision = c(precision, TP / (TP + FP))
        recall = c(recall, TP / (TP + FN))
    }
    
    # create DataFrame for MSE
    mse = data.frame('n_parameters'=rep(0,3), 'running_time'=rep(0,3), 'MSE'=mse, 'model_name'=c('Always-zero', 'Always-mean', 'Always-median'),
                     'precision'=precision, 'recall'=recall, 'total_predicted'=c(0, mean_count * dim(rows)[1], median_count * dim(rows)[1]), 
                     'total_ground_truth'=rep(sum(rows[,'ground_truth'])), 3)
    
    write.csv(mse, glue("./saved_models/{pipeline}/benchmarks.csv"))
}



# run for validation data
model_name = args$input_file
pipeline = args$pipeline
generate_benchmark(csv_file=glue('./saved_models/{pipeline}/{model_name}/{model_name}_validation.csv'))




