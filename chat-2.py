import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=".", type=str)
parser.add_argument('--max_length', default=200, type=int)
parser.add_argument('--no_compile', action='store_true', help='do not compile model (PyTorch < 2.0)')
args = parser.parse_args()

start = time.time()
print("Load model phi-2 ...")

if torch.cuda.is_available():
    print("using CUDA")
    torch.set_default_device("cuda")
    #model = AutoModelForCausalLM.from_pretrained(f"{args.model_path}/phi-2", torch_dtype="auto", 
    #                flash_attn=True, flash_rotary=True, fused_dense=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(f"{args.model_path}/phi-2", torch_dtype=torch.bfloat16, trust_remote_code=True)
else:
    print("using CPU")
    torch.set_default_device("cpu")
    model = AutoModelForCausalLM.from_pretrained(f"{args.model_path}/phi-2", torch_dtype=torch.float32, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(f"{args.model_path}/phi-2", trust_remote_code=True)
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
