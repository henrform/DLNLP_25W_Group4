from transformers import pipeline

# Point this to the local folder where you saved BERT
model_path = "./bert_base_uncased"

# Initialize the 'fill-mask' pipeline
unmasker = pipeline('fill-mask', model=model_path)

# Test sentence: "The movie was [MASK]."
# We expect words like: great, good, bad, amazing
results = unmasker("The movie was [MASK].")

print("--- BERT Synonym Test ---")
for res in results:
    print(f"Score: {res['score']:.4f} | Prediction: {res['token_str']}")