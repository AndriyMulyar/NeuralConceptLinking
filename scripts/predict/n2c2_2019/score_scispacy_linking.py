"""
Mention finding + linking:  25% accuracy
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

    correct_mentions_text = [' '.join(span.text for span in mention['mention']) for mention in file['mentions'] ]
    batch_candidates = linker.candidate_generator(correct_mentions_text, 30)
    #doc = nlp(text)

    correct_labels = [(mention['mention'][0].start_char,
                         mention['mention'][-1].end_char,
                         mention['concept']) for mention in file['mentions']]
    predicted_labels = []

    for mention, candidates in zip(correct_labels, batch_candidates):
        predicted = []
        for cand in candidates:
            score = max(cand.similarities)
            if score > linker.threshold:
                predicted.append((cand.concept_id, score))
        sorted_predicted = sorted(predicted, reverse=True, key=lambda x: x[1])
        if not predicted:
            predicted_labels.append(-1)
        else:
            predicted_labels.append(sorted_predicted[0][0])

    assert len(correct_labels) == len(predicted_labels)
    for correct_concept, predicted_concept in zip(correct_labels, predicted_labels):
        total+=1
        if correct_concept[2] == predicted_concept:
            correct+=1

assert total == 6925
print("Correct:", correct)
print("Total:", total)
print("Accuracy:", correct/total)