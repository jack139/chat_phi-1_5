import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=".", type=str)
parser.add_argument('--max_length', default=200, type=int)
parser.add_argument('--no_compile', action='store_true', help='do not compile model (PyTorch < 2.0)')
args = parser.parse_args()

if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")


start = time.time()
print("Load model phi-1.5 ...")
model = AutoModelForCausalLM.from_pretrained(f"{args.model_path}/phi-1_5", torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(f"{args.model_path}/phi-1_5", trust_remote_code=True)
if not args.no_compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model) # requires PyTorch 2.0 (optional)
print(f"Time elapsed: {(time.time() - start):.3f} sec.")


with torch.no_grad():
    print("Start inference mode.")
    print('=' * 85)

    while True:
        raw_input_text = input("Prompt:")
        raw_input_text = str(raw_input_text)
        if len(raw_input_text.strip()) == 0:
            break

        start = time.time()
        inputs = tokenizer(raw_input_text, return_tensors="pt", return_attention_mask=False)
        outputs = model.generate(**inputs, max_length=args.max_length)
        response = tokenizer.batch_decode(outputs)[0]
        print("Response: ", response)
        print(f">>>>>>> Time elapsed: {(time.time() - start):.3f} sec.\n\n")
