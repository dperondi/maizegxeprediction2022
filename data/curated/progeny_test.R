### Author: Sebastiano Busato
### Date: 12/15/2022 and onward
### What this code does:
###     1) Read in the vcf file 
###     2) Prefiltering based on quality: drop underrepresented SNPs and samples with low coverage
###     3) Compare between bioprojects to drop crappy data
###     4) Impute missing 
###     5) -- add more here --

########## SOME DEPENDENCIES AND USEFUL STUFF BELOW ##########

#ipak checks if needed packages are present, if not, installs 
#note: direct package install from cloudy is a big mess, so install missing from command line - interactive terminal instead
ipak <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}

#ignore this chunk - keeping it only cause I keep forgetting how to install stuff from bioconductor
# if (!require("BiocManager", quietly = TRUE))
#     install.packages("BiocManager")
# BiocManager::install("snpStats")

#list of required dependencies
packages <- c("asreml", "tidyverse", "RColorBrewer", "LDheatmap", 
            "BGLR", "asremlPlus", "GGally", "Matrix", "AGHmatrix", 
            "gplots", "psych", "cowplot", "rrBLUP", "Hmisc","ASRgenomics", 
            "reshape2", "knitr", "cowplot", "data.table", "ellipse", 
            "factoextra", "scattermore", "superheat", "mappoly", "vcfR")

#check, install and all that
ipak(packages)

####part 1: reading vcf file using read.delim and tab as a delimiter
start_time = Sys.time() #takes about 8 minutes on cloudy 
#note the check.names = F statement here: R will try to override varnames, we don't want that
genotypes <- read.delim(file = "/home/sbusato/G2F/Data/Training_Data/5_Genotype_Data_All_Years.vcf", skip = 22, sep = "\t", , check.names = F)
end_time = Sys.time()
readtime_baseR <- end_time - start_time
dim(genotypes)

##read experiment information from provided summary 
exp_metadata <- read.delim("/home/sbusato/G2F/Data/Training_Data/GenoDataSources.txt")
exp_metadata$Bioproject[exp_metadata$Bioproject==""] <- "missing"
length(unique(exp_metadata$Bioproject)) # 18 unique bioprojects
length(unique(exp_metadata$Dataset)) # 7 unique datasets

###how many cultivars per bioproject?
meta_bioproject <- exp_metadata %>% group_by(Bioproject) %>% count(cultivar) %>% summarise(num_cultivars = sum(n))

bp_stats <- ggplot() + geom_col(data = meta_bioproject, aes(x = Bioproject, y = num_cultivars)) +
theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
#ggsave("bp_stats.png",bp_stats)

###how many cultivars per dataset source
meta_datasource <- exp_metadata %>% group_by(Dataset) %>% count(cultivar) %>% summarise(num_cultivars = sum(n))

datasource_stats <- ggplot() + geom_col(data = meta_datasource, aes(x = Dataset, y = num_cultivars)) +
theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
ggsave("ds_stats.png",datasource_stats)

#with datasource strategy info only 
splitto <- strsplit(exp_metadata$Dataset,"_")
splitto2 <- 0
for (i in 1:length(splitto)) {
  splitto2[i] <- splitto[[i]][1]
}
exp_metadata$dataset_trim <- splitto2

meta_datasource2 <- exp_metadata %>% group_by(dataset_trim) %>% count(cultivar) %>% summarise(num_cultivars = sum(n))

datasource_stats2 <- ggplot() + geom_col(data = meta_datasource2, aes(x = dataset_trim, y = num_cultivars)) +
theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
#ggsave("ds_stats_short.png",datasource_stats2)

#extract parent1 and parent2 from vcf file 
parents <- colnames(genotypes[,-c(1:9)]) #should be 4928 individuals
parent1 <- ""
parent2 <- ""
for (i in 1:length(parents)){
  parent1[i] <- strsplit(parents[i], split = "/")[[1]][1]
  parent2[i] <- strsplit(parents[i], split = "/")[[1]][2]
}

#build better metadata table 
bettermeta <- data.frame(parents, parent1, parent2)
colnames(bettermeta) <- c("hybrid", "cultivar", "parent2")
#merge with old table for metadata
a <- merge(bettermeta, exp_metadata, by="cultivar")
colnames(a) <- c("parent1", "hybrid", "cultivar", "Dataset_p1", "SourceName_p1", "Bioproject_p1", "dataset_trim_p1")
bettermeta <- merge(a, exp_metadata, by="cultivar")

colnames(bettermeta) <- c("parent2","parent1","hybrid","Dataset_p1","SourceName_p1","Bioproject_p1","dataset_trim_p1",
                          "Dataset_p2","SourceName_p2","Bioproject_p2","dataset_trim_p2")   

#reorganize columns tidy
bettermeta <- bettermeta[,c(3,2,1,4,5,6,8,9,10,7,11)]
#check if parents belong to same bioproject, dataset or at least strategy
bettermeta$match_bioproject <- bettermeta$Bioproject_p1==bettermeta$Bioproject_p2
bettermeta$match_dataset <- bettermeta$Dataset_p1==bettermeta$Dataset_p2
bettermeta$match_ds_short <- bettermeta$dataset_trim_p1==bettermeta$dataset_trim_p2

match_ds_short <- ggplot() + geom_col(aes(x = c("true","false"), y = c(sum(bettermeta$match_ds_short), length(bettermeta$hybrid)-sum(bettermeta$match_ds_short))))
match_bp <- ggplot() + geom_col(aes(x = c("true","false"), y = c(sum(bettermeta$match_bioproject), length(bettermeta$hybrid)-sum(bettermeta$match_bioproject))))
#ggsave("match_ds_stats_short.png",match_ds_short)
#write bettermeta to file 
#write.csv(bettermeta, "bettermeta_geno.csv", row.names=FALSE)

#manipulating vcf file to work with mappoly: format is snp name, p1, p2, seq, seq_position, then all samples
#also shortened for test

genotypes_mappoly <- genotypes[1:50000,]
colnames(genotypes_mappoly)[1:15]
genotypes_mappoly <- genotypes_mappoly[,-c(6,7,8,9)]
genotypes_mappoly <- genotypes_mappoly[,c(3,4,5,1,2,6:4933)]
#write.csv(genotypes_mappoly, "genotypes_mappoly.csv", row.names=FALSE)

 

a <- (geno_short)
#b <- a %>% summarise_all(n_distinct)

b <- apply(a, 2, function(x) (unique(x)))
unique(unlist(b))



# start_time = Sys.time()
# geno_vcfR <- read.vcfR(file = "/home/sbusato/G2F/Data/Training_Data/5_short.vcf")
# end_time = Sys.time()
# readtime_vcfR <- end_time - start_time ## 11 minutes on cloudy

# geno_vcfR@gt <- geno_vcfR@gt[,-1]

# geno_vcfR_tidy <- vcfR2tidy(geno_vcfR)
# test <- vcfR2genind(geno_vcfR)
# test$[1:20]





##for testing purposes: shorten the DF to 20k SNPs, 2000 individuals 
geno_short <- genotypes
#assign row names 
rownames(geno_short) <- geno_short$ID
#drop useless columns
geno_short <- geno_short[, -c(1,2,3,4,5,6,7,8,9)]

#b <- apply(geno_short, 2, function(x) (unique(x)))
#d <- apply(geno_short,MARGIN=1,table)

#c <- unique(unlist(b))


refs <- "0/0"
alts <- c("1/1","2/2","3/3")
hets <- c("1/0","0/1","2/0","0/2","2/1","1/2","1/3","3/1","2/3","3/2","3/0","0/3")
missing  <- "./."

#length(refs)+length(alts)+length(hets)+1

start_time = Sys.time()

geno_short_mat <- as.matrix(geno_short)
geno_short_mat[geno_short_mat %in% refs] <- 0
geno_short_mat[geno_short_mat %in% alts] <- 2
geno_short_mat[geno_short_mat %in% hets] <- 1
geno_short_mat[geno_short_mat %in% missing] <- NA
geno_short_mat <- matrix(as.numeric(geno_short_mat), ncol = ncol(geno_short))
rownames(geno_short_mat) <- rownames(geno_short)
colnames(geno_short_mat) <- colnames(geno_short)

missing_per_snp <- apply(geno_short_mat, 1, function(x) as.numeric(sum(is.na(x))))
missing_per_sample <- apply(geno_short_mat, 2, function(x) as.numeric(sum(is.na(x))))

end_time = Sys.time()
processtime_wholemat <- end_time - start_time ##15.627 mins




#write.csv(geno_short_mat, "justincase.csv", row.names=TRUE)
#justin <- read.csv(file = "/home/sbusato/justincase.csv")

png("missing_per_sample.png")
miss_sample <-hist(missing_per_sample)
dev.off()
png("missing_per_snp.png")
miss_snp <- hist(missing_per_snp)
dev.off()

#filter out samples with more than 75000 SNPs as NA & SNPs missing from 1500 samples or more 
sample_rem <- data.frame(missing_per_sample[missing_per_sample>=75000]) #55 samples
colnames(sample_rem) <- c("countmissing")
SNP_rem <- data.frame(missing_per_snp[missing_per_snp>=1500]) #4486 SNPs
colnames(SNP_rem) <- c("countmissing")

`%nin%` = Negate(`%in%`)
geno_short_mat_filt <- geno_short_mat[rownames(geno_short_mat) %nin% rownames(SNP_rem), colnames(geno_short_mat) %nin% rownames(sample_rem)]

dim(geno_short_mat) - dim(geno_short_mat_filt)



#####PROGENY CALCULATION

#A matrix - expected relationship matrix 

families <- data.frame(parents, parent1, parent2)
colnames(families) <- c("ID","Par1","Par2")
families$ID <- paste("IND",as.character(1:4928),sep="_")
families$Par1 <- as.factor(families$Par1)
families$Par2 <- as.factor(families$Par2)
Amat <- Amatrix(families[,1:3], ploidy=2, verify = FALSE)
Amat <- createA(families)

datatreat(families)

test <- families$ID
test[(test)]






a <- which(geno_short == "./.", arr.ind=TRUE)
ind <- a[,1]
ind <- unique(ind)
snps <- 1:20000
snps <- snps[!snps %in% ind]

geno_short <- geno_short[snps,]

geno_short2 <- lapply(geno_short, as.numeric)
geno_short2 <- as.data.frame(geno_short2)
rownames(geno_short2) <- geno_short$ID

gs.mat <- as.matrix(geno_short2)
gs.mat.inv <- t(gs.mat)
colnames(gs.mat.inv) <- rownames(geno_short)

G <- Gmatrix(gs.mat.inv, method = "VanRaden", ploidy = 2)
heatmap(G)


hist(G[row(G)==col(G)])
mean(G[row(G)==col(G)])
hist(G[row(G)!=col(G)])
mean(G[row(G)!=col(G)])


#garbagee

###########
dat <- read_vcf(file.in = geno_vcfR, ploidy = 2)
dat <- filter_individuals(dat)
dat <- filter_missing(dat, type = "marker", filter.thres = .05)
dat <- filter_missing(dat, type = "individual", filter.thres = .05)
##########
