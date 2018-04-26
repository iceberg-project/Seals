# Script to plot a comparison plot across seal detection models

# load packages
library(ggplot2)
library(reshape2)
library(gridExtra)

# read csv table from all 
pooled_data = read.csv('./saved_models/pooled_prec_recall.csv', stringsAsFactors=FALSE)

# reshape for plotting
pooled_data = melt(pooled_data)
pooled_data = pooled_data[c(which(pooled_data['variable'] == 'precision'),
                            which(pooled_data['variable'] == 'recall')),]
colnames(pooled_data)[c(1, 3)] = c('label', 'metric')

# function for plotting precision/recall of a single label
label_plot = ggplot(data=pooled_data, 
                    mapping=aes(x=model_name, y=value * 100, fill=metric)) +
    theme_minimal(base_size=15) +
    geom_col(position='dodge', width=0.7) +
    ylim(c(0,100)) +
    labs(x='model name', y='%') +
    theme(axis.title = element_text(face="bold", size=18),
          strip.text.x = element_text(size=18, face="italic"))+
    scale_fill_brewer(palette = "Set2") +
    facet_wrap(~label) 

# save confusion_matrix as a png figure
png("./saved_models/model_comparison_plot.png", width=800, height=800)
print(label_plot)
dev.off()
