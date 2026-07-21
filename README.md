# WIR Repo
We use https://github.com/irgroup/topic-gen in the folder topic-gen-main.
The code we added is in the file 'Qrel_LLM_Judgements', all relevant documents are also in the folder. 
This leads to doubling of some files, as they are also in the folder 'Qrels updated'.

This is due to us wanting the steps to be clearly seperated, for easier reuse of the step by step process. 

## Steps taken

1. Corpus_Filtering
2. Human relabeling (no code for this)
3. LLM relabeling
4. Qrel Updating
5. Rejudged Analysis
6. Run Evaluation based of different Qrels

### Run Evaluation based of different Qrels
Run_Qrel_Compare has the script for the Run and Qrel comparision. The baseline runs need to be unzipped inside the folder before running code for everything to work as well as have the zipped version still in the folder. While the evaluation will work with just the zipped folders, the script also compares the qrels to the runs, for which unzipped txt files where used. 
