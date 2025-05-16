from transformers import pipeline
import numpy as np

classifier = pipeline("image-classification", model='Notebooks/Food_classifier/final_model')

if __name__ == '__main__':
    result = classifier('Data/Hamburger_(12164386105).jpg')
    best = max(result, key=lambda x: x['score'])
    print(best['label'], best['score'])
