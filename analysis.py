
# coding: utf-8

# # Analýza dát

# Dáta sú uložené v súboroch TSV (Tab Sepparated Values), čo je v podstate iba modifikácia súboru CSV (Comma Separated Values). Dáta obsahujú testovaciu a trénovaciu časť. Trénovací dataset obsahuje 27 atribútov, kým testovací 26. Ide o klasifikačný problém a 27-mi atribút v trénovacom datasete ozačuje evergreen stránky.  
# Nasledujú samply oboch datasetoch:

# In[1]:
import pandas

TRAIN_DF = pandas.read_csv('train.tsv', sep='\t')
TRAIN_DF.head()

# In[2]:
TEST_DF = pandas.read_csv('test.tsv', sep='\t')
TEST_DF.head()

# ## Analýza atribútov trénovacieho datasetu
# ### Url
# *Typ atribútu:* nominálny  
# Atribút predstavuje URL odkaz na webovú stránku.

# In[3]:
TRAIN_DF['url'].head()

# ### Urlid
# *Typ atribútu:* numerický, celočíselný, intervalový  
# Unikátne identifikátory pre každú webovú stránku. Hodnoty sú z intervalu <1,10566>.

# In[4]:
TRAIN_DF['urlid'].head()
TRAIN_DF['urlid'].max()
TRAIN_DF['urlid'].min()

# ### boilerplate
# *Typ atribútu:* JSON
# In[5]:
TRAIN_DF['boilerplate'][0].split('",')

# #### Title
# *Typ atribútu:* nominálny  
# Atribút predstavuje titul webovej stránky. Ide o atribút variabilnej dĺžky textu.

# #### Body
# *Typ atribútu:* nominálny  
# Atribút obsahuje text ľubovoľnej dĺžky. Ide o textový obsah webovej stránky alebo jej časť.

# #### Url
# *Typ atribútu:* nominálny  
# Atribút predstavuje URL identifikátor webovej stránky. Tento atribút je totožný s atribútom URL s rozdielom, že separátory URL (napr. bodky a lomítka) sú odstránené. Informácia v atribúte je redundantná a môže byť odstránená bez straty informácií celkového datasetu.

# ### Alchemy Category
# *Typ atribútu:* nominálny, kategorický  
# Atribút predstavuje kategóriu článku na webovej stránke. Hodnoty boli získané pomocou AlchemiAPI nástroja na analýzu textu, z ktorého boli získané kategórie (napr. business, sports…). Atribút obsahuje chýbajúce hodnoty v tvare otáznikov (?).
# In[6]:
print(TRAIN_DF['alchemy_category'].head(5))
description = TRAIN_DF['alchemy_category'].describe()
print('\nStats:\n' + str(description))
print('\nMissing value percentage: ' + str(description['freq']/description['count'] * 100))

# ### Alchemy Category Score
# *Typ atribútu:* numerický, decimálny, intervalový  
# Atribút predstavuje skóre získané AlchemyAPI na analýzu textu. Hodnoty sú z intervalu <0,1>.  
# Rovnako ako 3.4 aj tuná je 31.67% chýbajúcich hodnôt.
# In[7]:
print(TRAIN_DF['alchemy_category_score'].head(5))
print('\nStats:\n' + str(pandas.to_numeric(
    TRAIN_DF['alchemy_category_score'], errors='coerce'
    ).describe()))
description = TRAIN_DF['alchemy_category_score'].describe()
print('\nTop simbol\n' + str(description['top']))
print('\nMissing value percentage\n' + str(description['freq']/description['count'] * 100))

# ### Avglinksize
# *Typ atribútu:* numerický, decimálny  
# Atribút predstavuje priemerný počet slov v odkazoch nachádzajúcich sa na stránke.
# In[8]
print(TRAIN_DF['avglinksize'].head(5))
print('\nStats:\n' + str(pandas.to_numeric(TRAIN_DF['avglinksize'], errors='coerce').describe()))

# ### Pomerové hodnoty
# *Typ atribútov:* numerické, decimálne, intervalové  
# Táto skupina atribútov vyjadruje rôzne pomery.

# #### Common Link Ratio N
# Vyjadruje pomer odkazov, ktoré majú aspoň N slov spoločných s ostatnými odkazmi voči celkovému počtu odkazov.

# #### Compression Ratio
# Vyjadruje pomer kompresie dosiahnutej na danej stránke.

# #### Embed Ratio
# Počet použitých <embed> elementov na stránke. Atribút dosahuje hodnoty z intervalu <-1,0.25> s priemernou hodnotou -0.104. Prvý a tretí kvartil majú hodnotu 0, čo naznačuje dominanciu nulových hodnôt.

# #### Frame Tag Ratio
# Vyjadruje pomer <iframe> elementov (vnorená stránka) voči celkovému počtu elementov na stránke.

# #### HTML Ratio
# Pomer html elementov voči čistému textu na stránky.

# #### Image Ratio
# Pomer obrázkov na stránke (<img> element) voči čistému textu.  
# Minimálna hodnota sa rovná -1, čo naznačuje chybu v dátach.

# #### Parametrized Link Ratio
# Pomer parametrizovaných odkazov voči normálnym. Parametrizovaný odkaz obsahuje parametre (HTTP GET) alebo onClick() event listener.

# #### Spelling Errors Ratio
# Pomer slov nenájdených vo wiki – pokladajú sa za chybové.
# In[9]
for attr in [
        'commonlinkratio_1',
        'commonlinkratio_2',
        'commonlinkratio_3',
        'commonlinkratio_4',
        'embed_ratio',
        'frameTagRatio',
        'html_ratio',
        'image_ratio',
        'parametrizedLinkRatio',
        'spelling_errors_ratio'
    ]:
    print('\n\n### ' + attr + ' ###')
    print(TRAIN_DF[attr].head(5))
    print('\nStats:\n' + str(pandas.to_numeric(TRAIN_DF[attr], errors='coerce').describe()))

# ### Binárne nominálne hodnoty
# *Typ atribútov:* nominálne, binárne  
# Táto skupina atribútov nadobúda hodnoty 0 a 1.

# #### Frame Based
# Stránka nemá <body> element, ale má <frameset>.

# #### Has Domain Link
# Stránka obsahuje odkazy s doménou.

# #### Lengthy Link Domain
# Stránka obsahuje aspoň tri odkazy s minimálne 30 alfanumerických znakov.

# #### Label
# Stránka je evergreen. Tento atribút je dostupný iba pre trénovacie dáta.

# #### Is News
# Stránka je klasifikovaná ako novinová.

# #### News Front Page
# Stránka je klasifikovaná ako titulná stránka novín.
# In[10]
print(TRAIN_DF[[
    'framebased',
    'hasDomainLink',
    'lengthyLinkDomain',
    'label'
    ]].describe())

for attr in ['is_news', 'news_front_page']:
    print('\n\n### ' + attr + ' ###')
    print(TRAIN_DF[attr].head(5))
    print('\nStats:\n' + str(pandas.to_numeric(TRAIN_DF[attr], errors='coerce').describe()))

    if attr == 'news_front_page':
        counts = TRAIN_DF[attr].value_counts()
        count = TRAIN_DF[attr].count()
        print('\nMissing value percentage\n' + str(counts['?']/count * 100))

# ### Link Word Score
# *Typ atribútu:* numerický, pomerový  
# Percento textu, ktorý je súčasťou odkazu.
# In[11]
print(TRAIN_DF['linkwordscore'].head(5))
print('\nStats:\n' + str(TRAIN_DF['linkwordscore'].describe()))

# ### Non Markup Alphanum Characters
# *Typ atribútu:* numerický, celočíselný  
# Počet alfanumerických znakov čistého textu.
# In[12]
print(TRAIN_DF['non_markup_alphanum_characters'].head(5))
print('\nStats:\n' + str(TRAIN_DF['non_markup_alphanum_characters'].describe()))

# ### Number of Links
# *Typ atribútu:* numerický, celočíselný  
# Atribút vyjadruje počet odkazov na stránke.
# In[13]
print(TRAIN_DF['numberOfLinks'].head(5))
print('\nStats:\n' + str(TRAIN_DF['numberOfLinks'].describe()))

# In[14]
print(TRAIN_DF['numwords_in_url'].head(5))
print('\nStats:\n' + str(TRAIN_DF['numwords_in_url'].describe()))

