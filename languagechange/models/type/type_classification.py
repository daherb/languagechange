from languagechange.usages import TargetUsage
from typing import List, Union
from sentence_transformers import CrossEncoder
import numpy as np
import languagechange.logging

class ChangeTypeClassifier:

    def __init__(self, model_name = 'ChangeIsKey/change-type-classifier', max_length=512):
        self.model = CrossEncoder(model_name, max_length=max_length)    


    def predict(self, definitions : List[tuple[str]], all_scores = False, labels = True):
        '''
            Takes as input pairs of definitions and returns the types of semantic relationship (hyponymy, hypernymy, co-hyponymy, antonymy or homonymy) between the definitions of each pair.
            Args:
                definitions ([str]): a list of examples, each containing a tuple of definitions.
                all_scores (bool): determines whether to return the logit values for all classes or just the predicted class.
                labels (bool): determines whether to return the label(s) or the id(s) for each prediction.
        '''
        for example_pair in definitions:
            assert len(example_pair) == 2

        logits = self.model.predict(definitions)

        if not all_scores:
            # Choose the argmax
            predictions = np.argmax(logits,axis=1)
            # Return the label with the highest probability
            if labels:
                return self.ids2labels(predictions)
            # Get the id with the highest probability
            else:
                return predictions

        # Return raw (id) probabilities
        elif not labels:
            return logits

        # Return the probabilities of all labels
        else:
            return [{self.id2label(i) : prob for i, prob in enumerate(probs)} for probs in logits]  
        
    def id2label(self, id):
        if id in self.model.config.id2label:
            return self.model.config.id2label[id]
        else:
            logging.error(f"Id '{id}' not present in the model id:s.")
            raise KeyError
    
    def ids2labels(self, ids):
        return [self.id2label(id) for id in ids]
    
    def label2id(self, label):
        if label in self.model.config.label2id:
            return int(self.model.config.label2id[label])
        else:
            logging.error(f"Label '{label}' not present among model labels.")
            raise KeyError
    
    def labels2ids(self, labels):
        return [self.label2id(label) for label in labels]
    