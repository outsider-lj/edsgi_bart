# edsgi

## Usage

### Dependencies

Install the required libraries (Python 3.6 | CUDA 11.4)

### BART
https://huggingface.co/facebook/bart-base

### Knowledge
NRC_VAD http://saifmohammad.com/WebPages/nrc-vad.html

Concept_Net https://github.com/commonsense/conceptnet5/wiki/Downloads

### Dataset

The preprocessed dataset is already provided as `/data/ed/ed_with_kg.json`. However, if you want to create the dataset yourself, you need run all the files in preprocessing.

### Training
The first is to train the file train_edsgi.py;

The second step is to delete the tie_weight function and correct from_pretained method(line 1545) in modeling_edsgi_utils.py(load the step-generation model);

Next is to train the train_edsgi_integ.py.

## Testing
run test_edsgi.py
