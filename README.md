# edsgi

## Usage

### Dependencies

Install the required libraries (Python 3.6 | CUDA 11.4)

### Knowledge
NRC_VAD http://saifmohammad.com/WebPages/nrc-vad.html

Concept_Net https://github.com/commonsense/conceptnet5/wiki/Downloads

### Dataset

The preprocessed dataset is already provided as `/data/ed/ed_with_kg.json`. However, if you want to create the dataset yourself, you need run all the files in preprocessing.

### Training
main taining file is train_edsgi.py, train twice. 
in second time, modify optimizer and loss in train_edsgi.py; delete all tie_weight function and correct from_pretained method(line 1545) in modeling_edsgi_utils.py; modify model_checkpoint in train_edsgi_config.json

## Testing
run test_edsgi.py
