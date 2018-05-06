# Script to plot a comparison plot across seal detection models

# load packages
library(ggplot2)
library(reshape2)
library(argparser)

# define arg-parser 
parser = arg_parser("R script to get validation stats and plot a confusion matrix")
parser = add_argument(parser, "input_file", help="csv file with precision and recall")
inp_file = parse_args(parser=parser)

# read csv table from all 
pooled_data = read.csv(inp_file, stringsAsFactors=FALSE)
colnames(pooled_data)[2] = 'label'

# function for plotting precision/recall of a single label
label_plot = ggplot(data=pooled_data, 
                    mapping=aes(x=recall * 100, y=precision * 100, color=model_name)) +
    theme_minimal(base_size=15) +
    geom_point(size=4, alpha=0.8) +
    ylim(c(0,100)) +
    xlim(c(0,100)) +
    labs(x='Recall', y='Precision') +
    theme(axis.title = element_text(face="bold", size=18),
          strip.text.x = element_text(size=18, face="italic"))+
    scale_colour_brewer(palette="Set1") +
    facet_wrap(~label, ncol=2, scales="free") 

# save confusion_matrix as a png figure
png("comparison_plot.png", width=800, height=800)
print(label_plot)
dev.off()
