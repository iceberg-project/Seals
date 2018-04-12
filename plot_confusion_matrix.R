library(ggplot2)
library(reshape2)
library(scales)

# plot confusion matrix
get_confusion_matrix = function(csv_file, labels, pos_classes){
    # read validation output from validate 
    rows = read.csv(csv_file, stringsAsFactors=FALSE)
    
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
    png("conf_matrix.png")
    print(plot)
    dev.off()
    return(list('conf_matrix'=conf_matrix, 'pos_class_stats'=pos_classes_stats, 
                'balanced_accuracy'=balanced_acc))
}

# define labels
labels = c('crabeater', 'weddell', 'pack-ice', 'other', 'emperor', 'open-water', 'ice-sheet',
           'marching-emperor', 'crack', 'glacier', 'rock')

# define positive class labels
pos_classes = c('crabeater', 'weddell', 'emperor', 'marching-emperor')

# run for validation data
output = get_confusion_matrix(csv_file='training_set_12_Apr_2018_17_15_conf_matrix.csv', labels=labels,
                          pos_classes=pos_classes)



