# Script to plot a confusion matrix and get refined stats for seal detection models

# load packages
library(ggplot2)
library(reshape2)
library(scales)
library(glue)
library(argparse)

# define arg-parser 
parser = ArgumentParser(description="R script to get validation stats and plot a confusion matrix")
parser$add_argument("--input_file", type="character", help="model name generated during training")
parser$add_argument("--pipeline", type="character", help='model pipeline')
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
    
    # create DataFrame for MSE
    mse = data.frame('n_parameters'=total_params, 'running_time'=running_time, 'MSE'=mse, 'model_name'=model_name, 
                     'total_predicted'=sum(rows[,'predicted']), 'total_ground_truth'=sum(rows[,'ground_truth']))
    
    write.csv(mse, glue("./saved_models/{pipeline}/{model_name}/{model_name}_mse.csv"))
}



# run for validation data
model_name = args$input_file
pipeline = args$pipeline
get_mse(csv_file=glue('./saved_models/{pipeline}/{model_name}/{model_name}_validation.csv'))




