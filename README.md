# maizegxeprediction2022

This repository hosts the data and code used by the EnBiSys team in the Genomes to Fields (G2F) Genotype by Environment Prediction Competition ([www.maizegxeprediction2022.org](https://www.maizegxeprediction2022.org)).

## Overview

The following structure of directories was created to organize the project.

 * [data/](./data)
   * [raw/](./data/raw)
     * [COMPETITION_DATA_README.docx](./data/raw/COMPETITION_DATA_README.docx)
     * [Training_Data/](./data/raw/Training_Data)
       * [1_Training_Trait_Data_2014_2021.csv](./data/raw/Training_Data/1_Training_Trait_Data_2014_2021.csv)
       * [2_Training_Meta_Data_2014_2021.csv](./data/raw/Training_Data/2_Training_Meta_Data_2014_2021.csv)
       * [3_Training_Soil_Data_2015_2021.csv](./data/raw/Training_Data/3_Training_Soil_Data_2015_2021.csv)
       * [4_Training_Weather_Data_2014_2021.csv](./data/raw/Training_Data/4_Training_Weather_Data_2014_2021.csv)
       * [5_Genotype_Data_All_Years.vcf.zip](./data/raw/Training_Data/5_Genotype_Data_All_Years.vcf.zip)
       * [6_Training_EC_Data_2014_2021.csv](./data/raw/Training_Data/6_Training_EC_Data_2014_2021.csv)
       * [All_hybrid_names_info.csv](./data/raw/Training_Data/All_hybrid_names_info.csv)
       * [GenoDataSources.txt](./data/raw/Training_Data/GenoDataSources.txt)
     * [Testing_Data/](./data/raw/Testing_Data)
       * [1_Submission_Template_2022.csv](./data/raw/Testing_Data/1_Submission_Template_2022.csv)
       * [2_Testing_Meta_Data_2022.csv](./data/raw/Testing_Data/2_Testing_Meta_Data_2022.csv)
       * [3_Testing_Soil_Data_2022.csv](./data/raw/Testing_Data/3_Testing_Soil_Data_2022.csv)
       * [4_Testing_Weather_Data_2022.csv](./data/raw/Testing_Data/raw/4_Testing_Weather_Data_2022.csv)
       * [6_Testing_EC_Data_2022.csv](./data/raw/Testing_Data/6_Testing_EC_Data_2022.csv)
   * [curated/](./data/curated)
 * [submit/](./submit)
   * [v1/](./submit/v1)
   * [v2/](./submit/v2)
   * [v3/](./submit/v3)
   * [v4/](./submit/v4)
   * [v5/](./submit/v5)
 * [workspace/](./workspace)
   * [Daniel/](./workspace/Daniel)
   * [Kirtley/](./workspace/Kirtley)
   * [Max/](./workspace/Max)
   * [Peiran/](./workspace/Peiran)
   * [Sebastiano/](./workspace/Sebastiano)
 * [README.md](./README.md)

## Usage
The folder [workspace/](./workspace) contains folders for every team member where he/she can create Jupiter Notebooks, python files, folders, datasets, etc. This approach allows each one of us to try things without disturbing the code of others. At same time, it allow us to contribute to each other in a more organized way.

The folder [submit/](./submit) is strictly used by the team to submit each version to the website [eval.ai](https://eval.ai/web/challenges/challenge-page/1878/submission). We only have 5 submissions allowed.

The folder [data/](./data) contains the raw folder where the shared datasets by the organizers are located. A folder curated can be used to generate "curated" datasets.

## Contributing

Pull requests are welcome. For major updates, please open an issue first
to discuss what you would like to change.
