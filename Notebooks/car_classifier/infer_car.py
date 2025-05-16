from transformers import pipeline
import numpy as np

classifier = pipeline("image-classification", model='E:/Github/Product_classifier/Notebooks/car_classifier/final_model')

if __name__ == '__main__':
    result = classifier(r'E:\Github\Product_classifier\Notebooks\car_classifier\download (1).jpg')
    best = max(result, key=lambda x: x['score'])
    print(best['label'], best['score'])
