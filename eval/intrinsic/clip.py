from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

import torch
import numpy as np
from tqdm import tqdm

class CLIPSimilarityMetric:

    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        return
    
    def similarity(self, image, str):
        '''image = tensor of shape (3, H, W) or PIL Image and text is a strings
        returns the score of the associated image and string'''

        inputs = self.processor(text=[str], images=image, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        return logits_per_image.item()
    

    def similarity_seq(self, images, strs, reduction="mean"):
        '''returns representation of pair-wise scores of list of images and 
        paired strings'''
        assert(reduction == "mean" or reduction == "sum" or reduction == None)
        assert(len(images) == len(strs))

        scores= np.zeros(len(images))
        for i in range(len(images)):
            scores[i] = self.similarity(images[i], strs[i])
            
        if reduction is None:
            return scores
        elif reduction == "sum":
            return np.sum(scores)
        elif reduction == "mean":
            return np.mean(scores)
        
    def similarity_seq_batch(self, batch_images, batch_strs):
        '''list of list of images and list of list of strings
        will compute the mean similarity for each sublist and return mean

        So will compute mean of mean
        '''

        scores = np.zeros(len(batch_images))
        for i in range(len(batch_images)):
            scores[i] = self.similarity_seq(batch_images[i], batch_strs[i])
            
        return np.mean(scores)

if __name__ == "__main__":

    cs_metric = CLIPSimilarityMetric()
    
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw) 
    prompt = "a photo of a cat"
    print(cs_metric.similarity(image, prompt))
    print(cs_metric.similarity_seq([image] * 5, [prompt] * 5))