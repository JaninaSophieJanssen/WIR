import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell
def _():
    from langchain.chat_models import init_chat_model
    from topic_gen.generate import Generator
    from topic_gen.models import MTO_responds, Qrel, Qrels
    import pandas as pd
    from langchain_mistralai import ChatMistralAI

    #code is based in https://github.com/irgroup/topic-gen/tree/main "Generate relevance judgments (qrels)" it uses the new relabeled data and the topics, description and narritive created through WIR


    # Load data
    Janina = pd.read_excel("Relabel_Janina_translated.xlsx", sheet_name='Relabel_Janina_translated')
    Muhammed = pd.read_excel("relabel_Muhammed.xlsx", sheet_name='relabel_Muhammed')
    Maryam = pd.read_excel("Relabel_Maryam_translated.xlsx", sheet_name='Relabel_Maryam_translated')
    topics = pd.read_excel("topics.xlsx")

    combined_for_sorting = pd.concat([Janina, Muhammed, Maryam], ignore_index=True)



    combined_for_sorting[['query', 'description', 'narrative']] = pd.DataFrame([[None, None, None] for _ in range(len(combined_for_sorting))])


    # Step 2: Fill the columns by matching qid and query_id
    for _, row in topics.iterrows():
        query_id = row['query_id']
        query = row['query']
        description = row['description']
        narrative = row['narrative']

        # Update all rows in combined_for_sorting where qid matches query_id
        combined_for_sorting.loc[combined_for_sorting['qid'] == query_id, 'query'] = query
        combined_for_sorting.loc[combined_for_sorting['qid'] == query_id, 'description'] = description
        combined_for_sorting.loc[combined_for_sorting['qid'] == query_id, 'narrative'] = narrative


    combined_for_sorting.sort_values('qid')
    return (
        Generator,
        MTO_responds,
        Qrel,
        Qrels,
        combined_for_sorting,
        init_chat_model,
    )


@app.cell
def _(
    Generator,
    MTO_responds,
    Qrel,
    Qrels,
    combined_for_sorting,
    init_chat_model,
):
    #Get Document and Query Ids from the non-englishtext
    combined_for_sorting['docid'] = combined_for_sorting['docid'].astype(str)
    did = combined_for_sorting['docid'].values.tolist()  
    qid = combined_for_sorting['qid'].values.tolist() #

    #Get the filtered queries, descriptions, and narratives
    queries = combined_for_sorting['query'].values.tolist()
    descriptions = combined_for_sorting['description'].values.tolist()
    narratives = combined_for_sorting['narrative'].values.tolist()

    #get our translated documents
    documents = combined_for_sorting['abstract - English'].values.tolist()


    llm = init_chat_model(
        model="mistral-medium",  
        model_provider="mistralai",
        temperature=0,
        api_key="YOUR_MISTRAL_API_KEY"
    )

    generator = Generator(
        llm=llm,
        prompt="-DNA-zero-shot",
        output_class=MTO_responds,
        config={"max_concurrency": 8},
    )

    results = generator.generate(
        query=queries,           # list of query strings
        description=descriptions,
        narrative=narratives,
        document=documents,      # list of document texts to judge
    )
    print(results)
    qrels = Qrels[Qrel](qrels=[
        Qrel(query_id=qid, doc_id=did, relevance=res.O)
        for qid, did, res in zip(qid, did, results)
        if isinstance(res, MTO_responds)
    ])

    print(qrels)

    qrels.to_trec("qrels.txt")
    return


if __name__ == "__main__":
    app.run()
