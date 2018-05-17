# Script to plot a comparison plot across seal detection models

# load packages
library(ggplot2)
library(reshape2)
library(argparse)

# define arg-parser 
parser = ArgumentParser(description="R script to get pooled data csv and plot model comparisons")
parser$add_argument("--input_file", type="character", help=".csv file with precision and recall")
parser$add_argument("--output_file", type="character", help='filename for the .png plot')
parser$add_argument("--x", type="character", help="name of the column which will be used as the X axis")
parser$add_argument("--y", type="character", help='name of the column which will be used as the Y axis')
parser$add_argument("--facet", type="character", help='name of the column which will be used to determine facets')
args = parser$parse_args()

# unroll arguments
args = parse_args(parser=parser)
inp_file = args$input_file
out_file = args$output_file
x_label = args$x
y_label = args$y
facet = args$facet

# read csv table from all 
pooled_data = read.csv(inp_file, stringsAsFactors=FALSE)

# add columns for plot axes and facet wrap
pooled_data['x'] = pooled_data[, which(colnames(pooled_data) == x_label)]
pooled_data['y'] = pooled_data[, which(colnames(pooled_data) == y_label)]
pooled_data['facet'] =pooled_data[, which(colnames(pooled_data) == facet)]

# function for plotting precision/recall of a single label
label_plot = ggplot(data=pooled_data, 
                    mapping=aes(x=x, y=y, color=model_name)) +
    theme_minimal(base_size=15) +
    geom_point(size=4, alpha=0.8) +
    ylim(c(0,100)) +
    xlim(c(0,100)) +
    labs(x='Recall', y='Precision') +
    theme(axis.title = element_text(face="bold", size=18),
          strip.text.x = element_text(size=18, face="italic"))+
    scale_colour_brewer(palette="Set1") +
    facet_wrap(~facet, ncol=2, scales="free") 

# save confusion_matrix as a png figure
png(out_file, width=800, height=800)
print(label_plot)
dev.off()
