
import os

import subprocess
from openai import OpenAI
import google.generativeai as genai
from vllm import LLM, SamplingParams
import time


def get_free_gpu(use_gpu_num):
    # Use nvidia-smi command to get GPU usage information
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], 
                            stdout=subprocess.PIPE, text=True)
    # Parse the output
    gpu_info = result.stdout.strip().split('\n')
    free_gpus = [(int(info.split(',')[0]), int(info.split(',')[1])) for info in gpu_info]
    
    # Sort by free memory in descending order
    free_gpus.sort(key=lambda x: x[1], reverse=True)
    
    # Return GPU IDs with the most free memory
    return ",".join(str(i[0]) for i in free_gpus[:use_gpu_num])



class LLMInfer(object):
    def __init__(self, model_path, temperature=0.001, top_p=1, max_tokens=1000, language="zh", max_model_len=8192, tensor_parallel_size=1) -> None:
        gpu_ids = get_free_gpu(use_gpu_num=tensor_parallel_size)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=4096,
            top_p=1,
            top_k=-1
        )
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm = LLM(model=model_path, dtype="auto", trust_remote_code=True, max_model_len=max_model_len, tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=0.9)

    def inference(self, system_prompt, user_prompt):
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        question = [prompt, ]
        outputs = self.llm.generate(question, self.sampling_params, use_tqdm=False)
        if len(question) == 1:
            result = outputs[0].outputs[0].text
        else:
            result = []
            for idx, output in enumerate(outputs):
                out_generated_text = output.outputs[0].text
                result.append(out_generated_text)
        return result


class Llama(LLMInfer):
    def __init__(self, model_path, temperature=0.001, top_p=1, max_tokens=1000, language="zh", max_model_len=8192, tensor_parallel_size=2) -> None:
        super().__init__(model_path, temperature, top_p, max_tokens, language, max_model_len, tensor_parallel_size)


class Deepseek(object):
    def __init__(self, model_name, model_path=None, temperature=0.001, top_p=1, max_tokens=1000, language="zh") -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = "https://api.deepseek.com"
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, timeout=1000, max_retries=1, base_url=base_url)

    def creat_message(self, system_prompt=None, user_prompt=None, few_shot_examples=None):
        messages = []
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}]
        if few_shot_examples:
            for item in few_shot_examples:
                user, assistant = item["user"], item["assistant"]
                messages.append({"role": "system", "name": "example_user", "content": user})
                messages.append({"role": "system", "name": "example_assistant", "content": assistant})
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
        return messages
    
import time

class YourClass:
    def __init__(self, model_name):
        self.model_name = model_name
        self.last_request_time = 0  # To track the time of the last request


    def inference(self, system_prompt, user_prompt):
        messages = self.creat_message(system_prompt=system_prompt, user_prompt=user_prompt)
        result = self.request_openai(messages=messages, model=self.model_name)
        return result


class Gemini(object):
    def __init__(self, model_name, model_path=None, temperature=0.001, top_p=1, max_tokens=1000, language="zh") -> None:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"), transport='rest')
        self.model_name = model_name
        self.last_request_time = 0  # To track the time of the last request

    def creat_message(self, system_prompt=None, user_prompt=None, few_shot_examples=None):
        messages = []
        if few_shot_examples:
            for item in few_shot_examples:
                user, assistant = item["user"], item["assistant"]
                messages.append({"role": "model", "parts": user})
                messages.append({"role": "model", "parts": assistant})
        if user_prompt:
            messages.append({"role": "user", "parts": user_prompt})
        return messages

    def request_gemini(self, system_prompt, messages):
        try:
            # Check time difference from the last request
            current_time = time.time()
            if current_time - self.last_request_time < 7:
                # Sleep if the time difference is less than 6 seconds
                time.sleep(7 - (current_time - self.last_request_time))

            model = genai.GenerativeModel(self.model_name, system_instruction=system_prompt)
            response = model.generate_content(messages)
            result = response.text
            self.last_request_time = time.time()
            return result
        
        except Exception as e:
            raise e

    def inference(self, system_prompt, user_prompt):
        messages = self.creat_message(system_prompt=system_prompt, user_prompt=user_prompt)
        result = self.request_gemini(system_prompt=system_prompt, messages=messages)
        return result

class Kimi(object):
    def __init__(self, model_name, model_path=None, temperature=0.001, top_p=1, max_tokens=1000, language="zh") -> None:
        api_key = os.getenv("KIMI_API_KEY")
        base_url = os.getenv("KIMI_BASE_URL")
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, timeout=1000, max_retries=1, base_url=base_url)

    def creat_message(self, system_prompt=None, user_prompt=None):
        messages = []
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}]
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
        return messages

    def inference(self, system_prompt, user_prompt):
        messages = self.creat_message(system_prompt=system_prompt, user_prompt=user_prompt)
        response = self.client.chat.completions.create(  
            model = self.model_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.0
        )
        return response.choices[0].message.content


model_dict = {}
def get_model(model_name, model_path):
    global model_dict
    if model_name in model_dict:
        model = model_dict[model_name]
    else:
        model_name_lower = model_name.lower()
        if "qwen" in model_name_lower:
            model = LLMInfer(model_path)
        elif "llama" in model_name_lower:
            model = Llama(model_path)
        elif "deepseek" in model_name_lower:
            model = Deepseek(model_name)
        elif "gemini" in model_name_lower:
            model = Gemini(model_name)
        elif "kimi" in model_name_lower:
            model = Kimi(model_name)
        elif model_path:
            model = Llama(model_path)
        else:
            raise("Unsupported model")
        model_dict[model_name] = model
    return model

