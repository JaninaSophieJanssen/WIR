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

    Qrels = pd.concat([Janina, Muhammed, Maryam], ignore_index=True)

    bm25_different_relevance = pd.read_csv("bm25_different_relevance.txt")
    dense_different_relevance = pd.read_csv("dense_different_relevance.txt")
    return Qrels, bm25_different_relevance, dense_different_relevance


@app.cell
def _(dense_different_relevance):
    dense_different_relevance
    return


@app.cell
def _(Qrels, bm25_different_relevance, dense_different_relevance, pd):
    anl_dens = pd.merge(dense_different_relevance, Qrels, on=['qid', 'docid'], how="inner")
    anl_bm = pd.merge(bm25_different_relevance, Qrels, on=['qid', 'docid'], how="inner")
    dfs = [anl_bm, anl_dens]
    return (dfs,)


@app.cell
def _(dfs):
    import matplotlib.pyplot as plt
    import seaborn as sns

    for _df in dfs:
        # 1. Whitespaces bei den Werten in der Spalte entfernen und alles in Kleinbuchstaben umwandeln
        _df['titel nonEN but abstractEN'] = _df['titel nonEN but abstractEN'].astype(str).str.strip().str.lower()

        # Häufigkeiten zählen
        titel_auswertung = _df['titel nonEN but abstractEN'].value_counts()

        # Berechnung für die Textausgabe
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

        # 2. Visualisierung als Balkendiagramm
        plt.figure(figsize=(7, 5))
        sns.set_theme(style="whitegrid")

        # Die Verteilung plottem
        ax = sns.barplot(x=titel_auswertung.index, y=titel_auswertung.values, palette='Set1')

        # Labeling title and axes
        plt.title('Was the Title Non-EN Although the Abstract Was EN?', fontsize=13, fontweight='bold', pad=15)
        plt.xlabel('Response (no = Title was already English | yes = Title was non-English)', fontsize=11)
        plt.ylabel('Number of Documents', fontsize=11)

        # Die exakten Zahlen oben auf die Balken schreiben
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points',
                        fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.show()
    return plt, sns


@app.cell
def _(dfs, plt, sns):
    for df in dfs:    
        # 1. Das Wörterbuch zur Vereinheitlichung auf ENGLISCH definieren
        sprach_mapping_en = {
            # Indonesian
            'indonesian': 'Indonesian', 'indonesisch': 'Indonesian', 'indonesian ': 'Indonesian', 

            # Spanish
            'spanisch': 'Spanish', 'spanish': 'Spanish', 

            # Portuguese
            'portugisisch': 'Portuguese', 'portuguese': 'Portuguese',

            # Slovakian
            'slovak': 'Slovak',

            # Ukrainian
            'ukranian': 'Ukrainian', 'ukrainisch': 'Ukrainian',

            # Russian
            'russian': 'Russian',

            # Finnish
            'finish': 'Finnish',

            # Basque
            'basque': 'Basque',

            # Croatian
            'croatian': 'Croatian', 'kroatisch': 'Croatian',

            # Polish
            'polish': 'Polish',

            # Hindi
            'hindi': 'Hindi',

            # Catalan
            'catalan': 'Catalan',

            # Swedish
            'swedish': 'Swedish',

            # French
            'french': 'French',

            # Italian
            'italian': 'Italian',

            # Japanese
            'japanisch': 'Japanese',

            # Czech
            'czech': 'Czech',

            # Arabic
            'arabisch': 'Arabic',

            # Thai
            'thai': 'Thai',

            # Malayalam
            'malayalam': 'Malayalam',

            # Kombinierte Sprachen
            'indonesian/ arabic': 'Indonesian / Arabic',
            'indonesian  /arabic': 'Indonesian / Arabic'
        }

        # Spalte in String umwandeln, Leerzeichen außen entfernen und vereinheitlichen
        df['language_clean_en'] = df['language'].astype(str).str.strip().str.lower()

        # Fehlerhafte Schreibweisen überschreiben (wir verzichten auf das störende .capitalize())
        df['language_clean_en'] = df['language_clean_en'].map(sprach_mapping_en).fillna(df['language_clean_en'])

        # Häufigkeiten der bereinigten englischen Begriffe zählen
        sprachen_verteilung_en = df['language_clean_en'].value_counts()

        print("--- BEREINIGTE SPRACHVERTEILUNG (ENGLISCHE BEZEICHNUNGEN) ---")
        print(sprachen_verteilung_en)


        # =====================================================================
        # 2. VISUALISIERUNG ALS HORIZONTALES BALKENDIAGRAMM
        # =====================================================================
        plt.figure(figsize=(10, 8))
        sns.set_theme(style="whitegrid")

        # Horizontales Balkendiagramm erzeugen
        axs = sns.barplot(x=sprachen_verteilung_en.values, y=sprachen_verteilung_en.index, palette='viridis')

        # Titel und Achsen beschriften (Beschriftung Deutsch, Kategorien Englisch)
        plt.title('Distribution of Foreign Languages in the Dataset', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Number of Documents', fontsize=12)
        plt.ylabel('Language', fontsize=12)

        plt.tight_layout()
        plt.show()
    return


@app.cell
def _(dfs, plt, sns):
    for _df in dfs:
        # n.a. rausfiltern, weil uns nur die Dokumente mit einer klaren Position interessieren
        df_with_translation = _df[_df['English where (top - bottom)'].str.strip().str.lower() != 'n.a']
        position_counts = df_with_translation['English where (top - bottom)'].str.strip().str.lower().value_counts()
    
        print("Position der Übersetzung (ohne n.a.):")
        print(position_counts)
    
        plt.figure(figsize=(6, 5))
        _ax = sns.barplot(x=position_counts.index, y=position_counts.values, palette="Purples_r")
    
        plt.title("Wo befand sich die englische Übersetzung im Text?", fontsize=14, pad=15)
        plt.xlabel("Position im Dokument")
        plt.ylabel("Anzahl Dokumente")
        plt.xticks([0, 1], ['Unten (bottom)', 'Oben (top)'])
    
    
        plt.tight_layout()
        plt.savefig('plot_2_position.png', dpi=300)
        plt.show()

    return


@app.cell
def _(dfs, plt, sns):
    for _df in dfs:
        # Wir zählen wie oft "yes" und "no" in der Spalte vorkommen
        translation_counts = _df['has english partial (yes / no)'].str.strip().str.lower().value_counts()
        print("Absolute Zahlen:")
        print(_df['has english partial (yes / no)'].value_counts())
    
        # Balkendiagramm dazu
        plt.figure(figsize=(6, 5))
        aax = sns.barplot(x=translation_counts.index, y=translation_counts.values, palette="Blues_r")
    
        plt.title("Hatten die Dokumente bereits eine englische Teilübersetzung?", fontsize=14, pad=15)
        plt.xlabel("Englische Teilübersetzung vorhanden?")
        plt.ylabel("Anzahl Dokumente")
        plt.xticks([0, 1], ['Ja (yes)', 'Nein (no)'])
    
    
        plt.tight_layout()
        plt.savefig('plot_1_teiluebersetzung.png', dpi=300)
        plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
