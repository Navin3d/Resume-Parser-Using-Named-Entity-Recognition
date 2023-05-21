import spacy
from spacy.tokens import Doc, Span
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Step 1: Prepare annotated resume data
resume_data = pd.read_csv('annotated_resumes.csv')  # Load annotated resume data

# Step 2: Split data into training and testing sets
train_data, test_data = train_test_split(resume_data, test_size=0.2, random_state=42)

# Step 3: Train a named entity recognition (NER) model
nlp = spacy.blank('en')  # Create a blank English language model
ner = nlp.create_pipe('ner')  # Add named entity recognition pipeline
nlp.add_pipe(ner, last=True)

# Define entity labels for resume fields
labels = ['NAME', 'EMAIL', 'PHONE', 'SKILLS']
for label in labels:
    ner.add_label(label)

# Prepare training data in spaCy format
train_examples = []
for index, row in train_data.iterrows():
    text = row['text']
    entities = row['entities']
    train_examples.append((text, {'entities': entities}))

# Train the NER model
nlp.begin_training()
for epoch in range(10):
    np.random.shuffle(train_examples)
    losses = {}
    for text, annotations in train_examples:
        doc = nlp.make_doc(text)
        example = Doc(nlp.vocab, words=[t.text for t in doc])
        example.ents = [Span(example, start=e[0], end=e[1], label=e[2]) for e in annotations['entities']]
        nlp.update([example], losses=losses)

# Step 4: Test the NER model
test_results = []
for index, row in test_data.iterrows():
    text = row['text']
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    test_results.append((text, entities))

# Step 5: Evaluate the NER model
ground_truth = test_data['entities'].tolist()
predicted = [result[1] for result in test_results]
print(classification_report(ground_truth, predicted))

# Step 6: Extract information from resumes
def parse_resume(resume_text):
    doc = nlp(resume_text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    parsed_resume = {}
    for entity in entities:
        label = entity[1]
        if label not in parsed_resume:
            parsed_resume[label] = [entity[0]]
        else:
            parsed_resume[label].append(entity[0])

    return parsed_resume

# Example usage
resume_text = "John Doe\nEmail: john.doe@example.com\nPhone: 1234567890\nSkills: Python, Java, SQL, JavaScript"
parsed_resume = parse_resume(resume_text)
print(parsed_resume)
