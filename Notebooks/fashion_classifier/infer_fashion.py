from transformers import pipeline
import numpy as np

# Load pipeline from checkpoint
classifier = pipeline(
    "image-classification",
    model='Notebooks/fashion_classifier/final_model'
)

if __name__ == '__main__':
    image_path = r'Notebooks\fashion_classifier\Data\download.jpg'
    results = classifier(image_path)

    if results:
        top_result = max(results, key=lambda x: x['score'])
        print(f"Predicted label: {top_result['label']}, Score: {top_result['score']:.4f}")
    else:
        print("No results returned from classifier.")
