import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell
def _(pd):
    Janina = pd.read_excel("Relabel_Janina_translated.xlsx", sheet_name='Relabel_Janina_translated')
    Muhammed = pd.read_excel("relabel_Muhammed.xlsx", sheet_name='relabel_Muhammed')
    Maryam = pd.read_excel("Relabel_Maryam_translated.xlsx", sheet_name='Relabel_Maryam_translated')

    translated_and_judged = pd.concat([Janina, Muhammed, Maryam], ignore_index=True)
    llm_judged = pd.read_csv("qrels_rejudged_LLM.txt", sep=" ", header=None, names=['qid', 'zero', 'docid', 'new relevance'])

    qrels = pd.read_csv("qrels.txt", sep=" ", header=None, names=['qid', 'zero', 'docid', 'jugement'])
    return llm_judged, qrels, translated_and_judged


@app.cell
def _(pd):
    def update_jud_column(df1, df2):
        #code fixed with help of mistral 

        df1["qid"] = df1["qid"].astype('str')
        df1["docid"] = df1["docid"].astype('int')
        df1["jugement"] = df1["jugement"].astype('int')
        df2["qid"] = df2["qid"].astype('str')
        df2["docid"] = df2["docid"].astype('int')

        for col in ["maryam_relevance", "relevance", "relevance "]:
            if col in df2.columns:
                df2[col] = pd.to_numeric(df2[col], errors='coerce').astype('Int64')

        # Merge df1 and df2 on "id" and "doc" to find matches
        merged = pd.merge(df1, df2, on=["qid", "docid"], how="left")

        merged["jud_update"] = merged.apply(
            lambda row: row.get("maryam_relevance") if pd.notna(row.get("maryam_relevance"))
            else (row.get("new relevance") if pd.notna(row.get("new relevance"))
                  else (row.get("new relevance ") if pd.notna(row.get("new relevance "))
                        else None)),
            axis=1
        )

        # Update the "jud" column in df1 where there is a match
        df1.loc[merged["jud_update"].notna(), "jugement"] = merged.loc[merged["jud_update"].notna(), "jud_update"]

        return df1

    return (update_jud_column,)


@app.cell
def _(pd):
    def create_human_judgments(df2):

    
        df = pd.DataFrame()
        df["qid"] = df2["qid"]
        df["docid"] = df2["docid"]

        # Add the 'zero' column filled with 0
        df["zero"] = 0

        df["jugement"] = df2.apply(
            lambda row: row.get("maryam_relevance") if pd.notna(row.get("maryam_relevance"))
            else (row.get("new relevance") if pd.notna(row.get("new relevance"))
                  else (row.get("new relevance ") if pd.notna(row.get("new relevance "))
                        else None)),
            axis=1
        )

        return df

    return (create_human_judgments,)


@app.cell
def _(
    create_human_judgments,
    llm_judged,
    qrels,
    translated_and_judged,
    update_jud_column,
):
    qrel_rejudged_human = create_human_judgments(translated_and_judged)
    qrel_human_judgments = update_jud_column(qrels, translated_and_judged)
    qrel_llm_judgments = update_jud_column(qrels, llm_judged)
    return qrel_human_judgments, qrel_llm_judgments, qrel_rejudged_human


@app.cell
def _(qrel_human_judgments, qrel_llm_judgments, qrel_rejudged_human):
    qrel_rejudged_human.to_csv("qrel_rejudged_human.txt", sep=' ', index=False, na_rep='NaN')
    qrel_human_judgments.to_csv("qrel_human.txt", sep=' ', index=False, na_rep='NaN')
    qrel_llm_judgments.to_csv("qrel_llm.txt", sep=' ', index=False, na_rep='NaN')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
