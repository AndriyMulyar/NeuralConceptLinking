"""
Mention linking:  25% accuracy
"""
import scispacy, spacy
from scispacy.umls_linking import UmlsEntityLinker
from clinical_data.concept_linking import load_n2c2_2019
nlp = spacy.load("en_core_sci_sm")
linker = UmlsEntityLinker(resolve_abbreviations=True)
nlp.add_pipe(linker)

correct = 0
total = 0

for file in load_n2c2_2019(partition='test'):
    text = str(file['note'])

    doc = nlp(text)

    correct_labels = [(mention['mention'][0].start_char,
                         mention['mention'][-1].end_char,
                         mention['concept']) for mention in file['mentions']]
    predicted_labels = []


    for ent in doc.ents:
        if ent._.umls_ents:
            linked_entity = tuple((ent.start_char, ent.end_char, ent._.umls_ents[0][0]))
            predicted_labels.append(linked_entity)

    for start, end, cui in correct_labels:
        total+=1
        for p_start, p_end, p_cui in predicted_labels:
            if abs(p_start- start) < 2  and abs(p_end - end) < 2 and cui == p_cui:
                correct+=1

assert total == 6925
print("Correct:", correct)
print("Total:", total)
print("Accuracy:", correct/total)



