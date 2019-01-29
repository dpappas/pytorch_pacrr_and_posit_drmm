import numpy as np
import tqdm


def plot_attentions(documents, word_attention_scores, section_attention_scores, hierarchical=False):
    """
    :param documents: numpy array (str)
                    (number of documents, number of words) if hierarchical is False,
                    else (number of documents, number of sections, number of words)
    :param word_attention_scores: numpy array float32 (number of documents, number of words)
    :param section_attention_scores: numpy array float32 (number of documents, number of sections, number of words)
    :param hierarchical: True if hierarchical attention network
    :return: html document
    """
    for i, document in tqdm.tqdm(enumerate(documents)):

        html_content = '<!DOCTYPE html><html><head> ' \
                       '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">' \
                       '<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>' \
                       '<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>' \
                       '<style type="text/css">body { padding: 200px}span { border: 0px solid;}</style> </head>' \
                       '<body>'
        if hierarchical:
            word_attention_scores = np.squeeze(word_attention_scores, axis=0)
            section_attention_scores = np.squeeze(np.squeeze(section_attention_scores, axis=0), axis=1)
            html_content += '<table>'
            mean_section = np.mean(section_attention_scores[:len(document)])
            std_section = np.std(section_attention_scores[:len(document)])
            for i, (section, sec_word_attention_scores, section_attention_score) in enumerate(
                    zip(documents, word_attention_scores, section_attention_scores)):
                html_content += '<tr><td style="min-width:20px;"></td><td><b>Section #{}</b></td><tr>'.format(i + 1)
                html_content += '<tr>'
                # Compute z-scores and print heat-map for section
                if section_attention_score < mean_section:
                    html_content += '<td style="min-width:20px;">'
                else:
                    color_opacity = (section_attention_score - mean_section) / (3 * std_section)
                    html_content += '<td style= "min-width:20px; background-color:rgba(255, 0, 0, {0:.1f});">  </span>'.format(
                        color_opacity)
                html_content += '</td>'
                html_content += '<td>'
                # Compute mean and standard deviation
                mean = np.mean(section_attention_scores[:len(section)])
                std = np.std(section_attention_scores[:len(section)])
                # Compute z-scores and print heat-map for words
                for attention_score, word in zip(sec_word_attention_scores, section):
                    if word.token_text == '\n':
                        html_content += '<br/>'
                    else:
                        if attention_score < mean:
                            html_content += '<span>{} </span>'.format(word.token_text)
                        else:
                            color_opacity = (attention_score - mean) / (3 * std)
                            html_content += '<span style= "background-color:rgba(255, 0, 0, {0:.1f});">{1} </span>'.format(
                                color_opacity,
                                word.token_text)
                html_content += '</td>'
                html_content += '</tr>'
            html_content += '</table>'
        else:
            # Compute mean and standard deviation
            word_attention_scores = np.squeeze(word_attention_scores, axis=(0, 2))
            mean = np.mean(word_attention_scores[:len(document)])
            std = np.std(word_attention_scores[:len(document)])
            # Compute z-scores and print heat-map for words
            for word, attention_score in zip(document, word_attention_scores):
                if word.token_text == '\n':
                    html_content += '<br/>'
                else:
                    if attention_score < mean:
                        html_content += '<span>{} </span>'.format(word.token_text)
                    else:
                        color_opacity = (attention_score - mean) / (3 * std)
                        html_content += '<span style= "background-color:rgba(255, 0, 0, {0:.1f});">{1} </span>'.format(
                            color_opacity,
                            word.token_text)
        html_content += '</body>'
        return html_content
