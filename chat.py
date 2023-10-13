import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default="microsoft/phi-1_5", type=str)
args = parser.parse_args()

if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

print("Load model phi-1.5 ...")

start = time.time()
model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
print(f"Time elapsed: {(time.time() - start):.3f} sec.")


with torch.no_grad():
    print("Start inference mode.")
    print('=' * 85)

    while True:
        raw_input_text = input("Input:")
        raw_input_text = str(raw_input_text)
        if len(raw_input_text.strip()) == 0:
            break

        inputs = tokenizer(raw_input_text, return_tensors="pt", return_attention_mask=False)
        outputs = model.generate(**inputs, max_length=200)
        response = tokenizer.batch_decode(outputs)[0]
        print("Response: ", response)
        print("\n")
