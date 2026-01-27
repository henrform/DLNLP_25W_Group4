from transformers import AutoModel, AutoModelForSequenceClassification

model = AutoModel.from_pretrained("./distilled_tinybert_6l_rte")

print(model.encoder.layer[0].attention.self.query)
