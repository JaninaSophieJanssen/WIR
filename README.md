# WIR Repo
We use https://github.com/irgroup/topic-gen in the folder topic-gen-main.
The code we added is in the file 'Qrel_LLM_Judgements', all relevant documents are also in the folder. 
This leads to doubling of some files, as they are also in the folder 'Qrels updated'.

This is due to us wanting the steps to be clearly seperated, for easier reuse of the step by step process. 

## Run and Qrel Evaluation
Run_Qrel_Compare has the script for the Run and Qrel comparision. The baseline runs need to be unzipped inside the folder before running code for everything to work as well as have the zipped version still in the folder. While the evaluation will work with just the zipped folders, the script also compares the qrels to the runs, for which unzipped txt files where used. 
