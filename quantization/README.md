# Quantization

## Results / Issues

Same RTE accuracy, regardless of _quantization method_ and _number of bits_.

I first repurposed Henry's Jupyter notebook for quantization which worked quickly
out of the box. However, the results for both 4 and 8 bit quantized models where
**exactly the same** for RTE.

Asking Gemini for potential fixes, it recommended doing dynamic quantization, its
justification being that it is the universally accepted quantization method for encoder-only models like BERT and smaller models like TinyBERT.

The generated code snippet didn't work out-of-the-box since the evaluation didn't
recognize (or load?) the `bert.encoder` layers. For this, I modified the `task_distill.py`
script (the modified copy is `modified_task_distill.py`), specifically the logic for
loading the student model.

After doing so, the `bert.encoder` layers were recognized, the next error thrown was
regarding the `fit_dense.scale` field that couldn't be read when loading the model's
`state_dict`. After more tweaks, I got the modified evaluation to run - still resulting
in the **exact same** RTE accuracy.

### RTE

This is the distilled TinyBERT 6L model from Henry's Google Drive.

Distilled TinyBERT 6L (No Quantization):

Size: 257.7 MB

```
acc = 0.6895306859205776
eval_loss = 0.9556818736924065
```

#### GPTQ

- Quantization done via `quantization.py` script
- Evaluation done via `eval.sh` script

Distilled TinyBERT 6L (4 bits):

Size: 68 MB

```
acc = 0.4729241877256318
eval_loss = 0.7687404751777649
```

Distilled TinyBERT 6L (8 bits):

Size: 88.4 MB

```
acc = 0.4729241877256318
eval_loss = 0.7687404751777649
```

#### Dynamic

- Quantization done via `dynamic_quantization.py` script
- Evaluation done via `modified_eval.sh` script

For the evaluation of the dynamically quantized model to work, I had to
modify the `task_distill.py` section where the student model is loaded.
For this, I modified a copy of the script (`modified_task_distill.py`).

That way, the evaluation finally didn't throw any errors, the result is still
the same as for the GPTQ quantization.

Distilled TinyBERT 6l (torch.qint8)

Size: 132.3 MB

```
acc = 0.4729241877256318
eval_loss = 0.785525812043084
```
