import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    return pd, plt, sns


@app.cell
def _(pd):
    Janina = pd.read_excel("Relabel_Janina_translated.xlsx", sheet_name='Relabel_Janina_translated')
    Muhammed = pd.read_excel("relabel_Muhammed.xlsx", sheet_name='relabel_Muhammed')
    Maryam = pd.read_excel("Relabel_Maryam_translated.xlsx", sheet_name='Relabel_Maryam_translated')

    Qrels = pd.concat([Janina, Muhammed, Maryam], ignore_index=True)

    bm25_different_relevance = pd.read_csv("bm25_different_relevance.txt")
    dense_different_relevance = pd.read_csv("dense_different_relevance.txt")
    return Qrels, bm25_different_relevance, dense_different_relevance


@app.cell
def _(Qrels, bm25_different_relevance, dense_different_relevance, pd):
    anl_dens = pd.merge(dense_different_relevance, Qrels, on=['qid', 'docid'], how="inner")
    anl_bm = pd.merge(bm25_different_relevance, Qrels, on=['qid', 'docid'], how="inner")
    dfs = [anl_bm, anl_dens]
    custom_colors = ["#3eaee0", "#84ba5a", "#ea94a8", "#fed10a"]
    return custom_colors, dfs


@app.cell
def _(custom_colors, dfs, plt, sns):
    for _df in dfs:
        # Clean and process data
        _df['titel nonEN but abstractEN'] = _df['titel nonEN but abstractEN'].astype(str).str.strip().str.lower()
        titel_auswertung = _df['titel nonEN but abstractEN'].value_counts()

        # Print results
        anzahl_englisch = titel_auswertung.get('no', 0)
        anzahl_nicht_englisch = titel_auswertung.get('yes', 0)
        gesamt_titel = anzahl_englisch + anzahl_nicht_englisch

        print("--- AUSWERTUNG DER ENGLISCHEN TITEL ---")
        if gesamt_titel > 0:
            prozent_englisch = (anzahl_englisch / len(_df)) * 100
            print(f"Dokumente mit bereits englischem Titel ('no'): {anzahl_englisch} ({prozent_englisch:.1f}%)")
            print(f"Dokumente mit fremdsprachigem Titel ('yes'): {anzahl_nicht_englisch} ({(anzahl_nicht_englisch / len(_df)) * 100:.1f}%)")
        else:
            print("Spaltenwerte konnten nicht eindeutig als 'yes' oder 'no' identifiziert werden. Gefundene Werte:")
            print(titel_auswertung)

        # Plot
        plt.figure(figsize=(7, 5))
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(x=titel_auswertung.index, y=titel_auswertung.values, palette=custom_colors)

        plt.title('Was the Title Non-EN Although the Abstract Was EN?', fontsize=13, fontweight='bold', pad=15)
        plt.xlabel('Response (no = Title was already English | yes = Title was non-English)', fontsize=11)
        plt.ylabel('Number of Documents', fontsize=11)

        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points',
                        fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.show()
    return


@app.cell
def _(custom_colors, dfs, plt, sns):
    for df in dfs:
        sprach_mapping_en = {
            'indonesian': 'Indonesian', 'indonesisch': 'Indonesian', 'indonesian ': 'Indonesian',
            'spanisch': 'Spanish', 'spanish': 'Spanish',
            'portugisisch': 'Portuguese', 'portuguese': 'Portuguese',
            'slovak': 'Slovak',
            'ukranian': 'Ukrainian', 'ukrainisch': 'Ukrainian',
            'russian': 'Russian',
            'finish': 'Finnish',
            'basque': 'Basque',
            'croatian': 'Croatian', 'kroatisch': 'Croatian',
            'polish': 'Polish',
            'hindi': 'Hindi',
            'catalan': 'Catalan',
            'swedish': 'Swedish',
            'french': 'French',
            'italian': 'Italian',
            'japanisch': 'Japanese',
            'czech': 'Czech',
            'arabisch': 'Arabic',
            'thai': 'Thai',
            'malayalam': 'Malayalam',
            'indonesian/ arabic': 'Indonesian / Arabic',
            'indonesian  /arabic': 'Indonesian / Arabic'
        }

        df['language_clean_en'] = df['language'].astype(str).str.strip().str.lower()
        df['language_clean_en'] = df['language_clean_en'].map(sprach_mapping_en).fillna(df['language_clean_en'])
        sprachen_verteilung_en = df['language_clean_en'].value_counts()

        print("--- BEREINIGTE SPRACHVERTEILUNG (ENGLISCHE BEZEICHNUNGEN) ---")
        print(sprachen_verteilung_en)

        plt.figure(figsize=(10, 8))
        sns.set_theme(style="whitegrid")
        axs = sns.barplot(x=sprachen_verteilung_en.values, y=sprachen_verteilung_en.index, palette=custom_colors)

        plt.title('Distribution of Foreign Languages in the Dataset', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Number of Documents', fontsize=12)
        plt.ylabel('Language', fontsize=12)

        plt.tight_layout()
        plt.show()
    return


@app.cell
def _(custom_colors, dfs, plt, sns):
    for _df in dfs:
        df_with_translation = _df[_df['English where (top - bottom)'].str.strip().str.lower() != 'n.a']
        position_counts = df_with_translation['English where (top - bottom)'].str.strip().str.lower().value_counts()

        print("Position der Übersetzung (ohne n.a.):")
        print(position_counts)

        plt.figure(figsize=(6, 5))
        _ax = sns.barplot(x=position_counts.index, y=position_counts.values, palette=custom_colors)

        plt.title("Wo befand sich die englische Übersetzung im Text?", fontsize=14, pad=15)
        plt.xlabel("Position im Dokument")
        plt.ylabel("Anzahl Dokumente")
        plt.xticks([0, 1], ['Unten (bottom)', 'Oben (top)'])

        plt.tight_layout()
        plt.savefig('plot_2_position.png', dpi=300)
        plt.show()
    return


@app.cell
def _(custom_colors, dfs, plt, sns):
    for _df in dfs:
        translation_counts = _df['has english partial (yes / no)'].str.strip().str.lower().value_counts()
        print("Absolute Zahlen:")
        print(_df['has english partial (yes / no)'].value_counts())

        plt.figure(figsize=(6, 5))
        aax = sns.barplot(x=translation_counts.index, y=translation_counts.values, palette=custom_colors)

        plt.title("Hatten die Dokumente bereits eine englische Teilübersetzung?", fontsize=14, pad=15)
        plt.xlabel("Englische Teilübersetzung vorhanden?")
        plt.ylabel("Anzahl Dokumente")
        plt.xticks([0, 1], ['Ja (yes)', 'Nein (no)'])

        plt.tight_layout()
        plt.savefig('plot_1_teiluebersetzung.png', dpi=300)
        plt.show()
    return


@app.cell
def _(dfs, pd):
    for _df in dfs:
        df1 = pd.read_csv("new_human_llm_relabel.csv", sep=";", encoding="utf-8-sig")
        df1.columns = df1.columns.str.strip()
        df2 = pd.merge(_df, df1, on=['qid', 'docid'], how="inner")
        print(df2[f"origianl title english (yes/no)"].value_counts())
    return


if __name__ == "__main__":
    app.run()
