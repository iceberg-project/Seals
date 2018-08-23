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
parser$add_argument("--dest_folder", type="character", default="saved_models", help="folder where pipeline is located")
args = parser$parse_args()


# plot confusion matrix
get_confusion_matrix = function(csv_file, labels, pos_classes){
    # read validation output from validate 
    rows = read.csv(csv_file, stringsAsFactors=FALSE)
    
    # get running time and parameters (last row)
    total_params = as.integer(rows[dim(rows)[1], 1])
    running_time = as.double(rows[dim(rows)[1], 2])
    
    # remove last row
    rows = rows[-dim(rows)[1],]
    
    # reformat as confusion matrix
    n_labels = length(labels)
    conf_matrix = data.frame(matrix(0, nrow=n_labels, ncol=n_labels), row.names=labels)
    colnames(conf_matrix) = labels
    for(row in 1:dim(rows)[1]){
        conf_matrix[rows[row, "ground_truth"], rows[row, "predicted"]] = conf_matrix[rows[row, "ground_truth"], 
                                                                                     rows[row, "predicted"]] + 1
    }
    
    # get balanced validation accuracy 
    accs = c()
    for(lbl in labels){
        accs = c(accs, conf_matrix[lbl, lbl] / sum(conf_matrix[lbl,]))
    }
    # fill nas in case there are now samples in one class
    accs[is.na(accs)] = 0
    balanced_acc = sum(accs) / length(accs)
    
    # get detailed statistics on positive classes
    pos_classes_stats = data.frame()
    for(class in pos_classes){
        # class-specific precision
        class_prec = conf_matrix[class, class] / sum(conf_matrix[,class])
        # class-specific recall
        class_rec = conf_matrix[class, class] / sum(conf_matrix[class,])
        # add to dataframe
        pos_classes_stats = rbind(pos_classes_stats, c(class_prec, class_rec))
    }
    colnames(pos_classes_stats) = c('precision', 'recall')
    pos_classes_stats['label'] = pos_classes
    pos_classes_stats['balanced_accuracy'] = rep(balanced_acc, dim(pos_classes_stats)[1])
    pos_classes_stats['model_name'] = rep(model_name, dim(pos_classes_stats)[1])
    pos_classes_stats['n_parameters'] = rep(total_params, dim(pos_classes_stats)[1])
    pos_classes_stats['running_time'] = rep(running_time, dim(pos_classes_stats)[1])
    
    # melt dataframe for plotting
    plot_df = melt(as.matrix(conf_matrix))
    
    # plot confusion matrix
    plot = ggplot(plot_df, aes(Var2, Var1)) + # x and y axes => Var1 and Var2
        geom_tile(aes(fill = value)) + # background colours are mapped according to the value column
        geom_text(aes(fill = plot_df$value, label = round(plot_df$value, 2))) + # write the values
        scale_fill_gradient2(low = muted("darkred"), 
                             mid = "white", 
                             high = muted("midnightblue"), 
                             midpoint = mean(plot_df$value)) + # determine the colour
        theme(panel.grid.major.x=element_blank(), #no gridlines
              panel.grid.minor.x=element_blank(), 
              panel.grid.major.y=element_blank(), 
              panel.grid.minor.y=element_blank(),
              panel.background=element_rect(fill="white"), # background=white
              axis.text.x = element_text(angle=90, hjust = 1,vjust=1,size = 12,face = "bold"),
              plot.title = element_text(size=20,face="bold"),
              axis.text.y = element_text(size = 12,face = "bold")) + 
        theme(legend.title=element_text(face="bold", size=14)) + 
        scale_x_discrete(name="") +
        scale_y_discrete(name="") +
        theme(legend.position="none")
    
    # save confusion_matrix as a png figure
    png(glue("./{dest_folder}/{pipeline}/{model_name}/{model_name}_conf_matrix.png"))
    print(plot)
    dev.off()
    
    # write performance metrics to csv
    write.csv(pos_classes_stats, glue("./{dest_folder}/{pipeline}/{model_name}/{model_name}_prec_recall.csv"))
}



# unroll inputs
model_name = args$input_file
pipeline = args$pipeline
dest_folder = args$dest_folder

# define positive class labels, checking for binary
if(grepl('binary', model_name) == TRUE){
    pos_classes = c('seal', 'non-seal')
    # define labels
    labels = c('seal', 'non-seal')
    
} else{
    pos_classes = c('crabeater', 'weddell', 'emperor', 'marching-emperor')
    # define labels
    labels = c('crabeater', 'weddell', 'pack-ice', 'other', 'emperor', 'open-water', 'ice-sheet',
               'marching-emperor', 'crack', 'glacier', 'rock')
    
}

# run validation
get_confusion_matrix(csv_file=glue('./{dest_folder}/{pipeline}/{model_name}/{model_name}_validation.csv'), labels=labels, pos_classes=pos_classes)




