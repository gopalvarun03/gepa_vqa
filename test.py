from vllm_vlm import VLLMWrapper
from vqa_dataset import load_docvqa

# Load one DocVQA sample
data = load_docvqa(split='validation', max_samples=1)
sample = data[0]

vlm = VLLMWrapper(base_url='http://localhost:8000', model='Qwen/Qwen2.5-VL-7B-Instruct')
print(f'Question: {sample["question"]}')
print(f'Expected: {sample["answer"]}')
print('Querying (with resized image)...')
response = vlm.query(sample['image_path'], sample['question'], 'Answer briefly.')
print(f'Response: {response}')