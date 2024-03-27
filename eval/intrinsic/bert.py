from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

from tqdm import tqdm
import numpy as np

class NextSentenceMetric:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
        
        return
    
    def next_sentence_predict(self, str1, str2):
        '''returns the probability that BERT believes str1 follows str2'''
        encoding = self.tokenizer(str1, str2, return_tensors="pt")
        outputs = self.model(**encoding)
        logits = outputs.logits

        probs = torch.nn.functional.softmax(logits)
        return probs[0, 0]
    

    def next_sentence_predict_batch(self, strs1, strs2, reduction="mean"):
        '''pairwise probabilities of pairwise strings between two lists'''
        assert(reduction == "mean" or reduction == "sum" or reduction == None)
        assert(len(strs1) == len(strs2))

        probs = np.zeros(len(strs1))
        for i in range(len(strs1)):
            probs[i] = self.next_sentence_predict(strs1[i], strs2[i])
            
        if reduction is None:
            return probs
        elif reduction == "sum":
            return np.sum(probs)
        elif reduction == "mean":
            return np.mean(probs)
        
    def next_sentence_predict_seq(self, strs):
        '''list of strings [s1, s2, s3, ...] whose adjacent similarity will be
        calculated and the average will be returned'''
        assert(len(strs) >= 2)

        strs1 = strs[:-1]
        strs2 = strs[1:]

        return self.next_sentence_predict_batch(strs1, strs2, reduction="mean")
    

    def next_sentence_predict_seq_batch(self, batch_strs):
        '''list of list of strings where each inner list is a sequence of strings
        generated from an image sequence 
        
        Function will return the average of the next_sentence_predict_seq 
        for each sequence of strings (will be an average of averages)'''

        avg_probs = np.zeros(len(batch_strs))
        for i, strs in enumerate(batch_strs):
            avg_probs[i] = self.next_sentence_predict_seq(strs)

        return np.mean(avg_probs)


if __name__ == "__main__":

    prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    next_sentence = "The sky is blue due to the shorter wavelength of blue light."

    ns_metric = NextSentenceMetric()
    ns_metric.next_sentence_predict(prompt, next_sentence)

    avg_prob = ns_metric.next_sentence_predict_batch(
        [prompt, prompt] * 10,
        [next_sentence, next_sentence] * 10,
        reduction="mean"
    )
    print(avg_prob)