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
parser$add_argument("--facet", default='NULL', type="character", help='name of the column which will be used to determine facets')

# unroll arguments
args = parser$parse_args()
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

# find x and y limits
x_lim = c(0, max(pooled_data['x']))
y_lim = c(0, max(pooled_data['y']))

# function for plotting precision/recall of a single label
label_plot = ggplot(data=pooled_data, 
                    mapping=aes(x=x, y=y, color=model_name)) +
    theme_minimal(base_size=15) +
    geom_point(size=11, alpha=0.5) +
    geom_point(size=12, shape=21, color='black') +
    labs(x=x_label, y=y_label) +
    xlim(x_lim) +
    ylim(y_lim) +
    theme(axis.title = element_text(face="bold", size=22),
          strip.text.x = element_text(size=18, face="italic"))+
    scale_colour_brewer(palette="Set1") 

if(facet != 'NULL'){
    label_plot = label_plot + facet_wrap(~facet, ncol=2, scales="free")
}

# save confusion_matrix as a png figure
png(out_file, width=1200, height=800)
print(label_plot)
dev.off()
