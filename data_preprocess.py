'''Data preprocessing module.
'''

import pandas

for file_name in ['train.tsv', 'test.tsv']:
    # Load data file
    TRAIN_DF = pandas.DataFrame(pandas.read_csv(file_name, sep='\t'))
    NORMALIZED_DF = pandas.DataFrame()

    # Removed columns
    REMOVED = [
        'embed_ratio',
        'framebased'
    ]

    # Columns to be normalized
    TO_NORMALIZE = {
        'commonlinkratio_2',
        'frameTagRatio',
        'html_ratio',
        'image_ratio',
        'parametrizedLinkRatio',
        'spelling_errors_ratio',
        'linkwordscore',
        'non_markup_alphanum_characters',
        'numberOfLinks',
        'avglinksize',
        'compression_ratio'
    }

    # Renema some of the columns to be more descriptive
    NEW_COLUMN_NAME = {
        'boilerplate': 'textcontent',
        'commonlinkratio_1': 'common_link_ratio_1',
        'commonlinkratio_2': 'common_link_ratio_2',
        'commonlinkratio_3': 'common_link_ratio_3',
        'commonlinkratio_4': 'common_link_ratio_4',
        'compression_ratio': 'compression_ratio',
        'framebased': 'frameset_instead_body',
        'frameTagRatio': 'iframe_ratio',
        'hasDomainLink': 'has_domain_link',
        'lengthyLinkDomain': 'has_long_links',
        'parametrizedLinkRatio': 'parametrized_link_ratio',
        'linkwordscore': 'hyperlink_text_ratio',
        'non_markup_alphanum_characters': 'raw_text_character_count',
        'numberOfLinks': 'link_count',
        'avglinksize': 'avg_link_word_count',
        'label': 'is_evergreen'
    }

    # Modify, normalize and remove columns
    for key in TRAIN_DF:
        if key in REMOVED:
            continue

        # Change column name
        column_name = key
        if key in NEW_COLUMN_NAME:
            column_name = NEW_COLUMN_NAME[key]

        # Handle outliers
        # 'compression_ratio' has outliers: 10% of the data are above 3 * standard deviation
        #   - left untouched for now, only normalized
        # image_ratio - outliers replaced for mean value

        # Special outliers
        if column_name == 'image_ratio':
            TRAIN_DF.loc[TRAIN_DF['image_ratio'] < 0, ('image_ratio')] = 0

            # Replace outliers with mean values
            outlier_treshold = 3 * TRAIN_DF[key].std()
            TRAIN_DF.loc[
                TRAIN_DF['image_ratio'] - TRAIN_DF['image_ratio'].mean() > outlier_treshold,
                ('image_ratio')
            ] = TRAIN_DF['image_ratio'].mean()

        if column_name == 'is_news':
            TRAIN_DF.loc[TRAIN_DF['is_news'] == '?', ('is_news')] = '0'

        if column_name == 'news_front_page':
            TRAIN_DF.loc[TRAIN_DF['news_front_page'] == '?', ('news_front_page')] = '0'

        # Normalize data
        if key in TO_NORMALIZE:
            min_value = TRAIN_DF[key].min()
            max_value = TRAIN_DF[key].max()
            NORMALIZED_DF[column_name] = (TRAIN_DF[key] - min_value) / (max_value - min_value)
        else:
            NORMALIZED_DF[column_name] = TRAIN_DF[key]

    # Save data to file
    NORMALIZED_DF.to_csv('processed_%s' % file_name, sep='\t')
