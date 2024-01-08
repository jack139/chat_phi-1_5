# Simple command line chat for phi-1.5/2 model



## Run

```bash
python3 chat.py --model_path "microsoft"

python3 chat-2.py --model_path "microsoft"
```



## Example

```
Load model phi-1.5 ...
Time elapsed: 18.459 sec.
Start inference mode.
=====================================================================================
Prompt:I want to write a quick sort program in python, the code is                                   
Response:  I want to write a quick sort program in python, the code is as follows:
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        less = [x for x in arr[1:] if x <= pivot]
        greater = [x for x in arr[1:] if x > pivot]
        return quick_sort(less) + [pivot] + quick_sort(greater)

arr = [3,2,1,4,5,6,7,8,9]
print(quick_sort(arr))

The output is:
[1, 2, 3, 4, 5, 6, 7, 8, 9]

However, I want to know why the output is [1, 2, 3, 4, 5, 6, 7, 8, 9]
```



## Model

[Microsoft phi-1.5](https://huggingface.co/microsoft/phi-1_5)
[Microsoft phi-2](https://huggingface.co/microsoft/phi-2)

