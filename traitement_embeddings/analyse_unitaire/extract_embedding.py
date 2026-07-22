import os
import numpy as np

emb_from_file = "./embeddings.tsv"
utterance_id = 'audio-1775033905.42249_operateur_7'

def extract_representation(file_name, speaker_id):
    """From a parsed file containing speaker name (or id) and a given representation 
    (quantized units or speaker embedding), this function allows to extract the
    representation as a numpy array."""

    rep = None
    with open(file_name) as f:
        non_empty_lines = [line for line in f if line.strip()]
        #nb_line = len(non_empty_lines)
        for line in non_empty_lines:
            if  line.find(speaker_id) != -1: 
                rep_str = line.rsplit("|")[1]
                rep = np.fromstring(rep_str, sep=' ',  dtype=np.float64) 
                print(f"Sucessfully found corresponding representation for {speaker_id}")
                break
    if rep is None:
        print(f"Representation not found for {speaker_id}")
    return rep

rep = extract_representation(emb_from_file, utterance_id)
#print(rep)
print(rep.shape)
