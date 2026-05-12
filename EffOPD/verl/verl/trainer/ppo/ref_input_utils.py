# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utility functions for preparing reference model inputs when ref model uses a different tokenizer or prompt template.
Also includes critique distillation utilities for computing ref log probs with external vLLM critique.
"""

import logging
import os
import json
import requests
import concurrent.futures
from typing import Optional, List, Dict, Any

import torch
import numpy as np

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CritiqueClient:
    """Client for calling external vLLM server to generate critiques."""
    
    def __init__(self, server_url: str, model: str = None):
        """Initialize the critique client.
        
        Args:
            server_url: The URL of the vLLM server (e.g., "http://127.0.0.1:8000")
            model: The model name to use for critique generation
        """
        self.server_url = server_url.rstrip('/')
        self.model = model
    
    def generate_critique(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> Optional[str]:
        """Generate a single critique."""
        url = f"{self.server_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=2400)
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.warning(f"Critique request failed: HTTP {response.status_code}")
                return None
        except Exception as e:
            logger.warning(f"Critique request error: {e}")
            return None
    
    def batch_generate_critiques(
        self,
        batch_messages: List[List[Dict[str, str]]],
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_workers: int = 1024,
    ) -> List[Optional[str]]:
        """Generate critiques in batch using concurrent requests.
        
        Args:
            batch_messages: List of message lists, each for one critique request
            max_tokens: Maximum tokens for generation
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_workers: Maximum number of concurrent workers
            
        Returns:
            List of critique strings (None for failed requests)
        """
        def single_request(idx_messages):
            idx, messages = idx_messages
            critique = self.generate_critique(messages, max_tokens, temperature, top_p)
            return idx, critique
        
        results = [None] * len(batch_messages)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(batch_messages))) as executor:
            future_to_idx = {
                executor.submit(single_request, (i, msgs)): i 
                for i, msgs in enumerate(batch_messages)
            }
            
            for future in concurrent.futures.as_completed(future_to_idx):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    logger.warning(f"Batch critique request error: {e}")
                    idx = future_to_idx[future]
                    results[idx] = None
        
        return results


def build_critique_prompt(
    problem: str,
    solution: str,
    ground_truth: str,
    critique_prompt_template: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Build the prompt for generating a critique.
    
    Args:
        problem: The original problem/question
        solution: The actor's generated solution
        ground_truth: The ground truth answer
        critique_prompt_template: Optional custom template for critique generation
        
    Returns:
        List of messages for the critique request
    """
    if critique_prompt_template is None:
        critique_prompt_template = """You are a mathematical reasoning expert. A student attempted to solve the following problem.

Problem:
{problem}

Correct Answer: {ground_truth}

Student's Solution:
{solution}

Please critique each reasoning step in the student's solution, evaluating whether there are any issues. If there are problems, point them out and provide suggestions for correction. Finally, judge the correctness of the student's final answer. Note: You must evaluate each reasoning step sequentially from beginning to end.
"""  
    critique_content = critique_prompt_template.format(
        problem=problem,
        solution=solution,
        ground_truth=ground_truth,
    )
    
    return [{"role": "user", "content": critique_content}]


def prepare_critique_distillation_inputs(
    batch: DataProto,
    tokenizer,
    critique_vllm_url: str,
    critique_model: Optional[str] = None,
    critique_prompt_template: Optional[str] = None,
    ref_apply_chat_template_kwargs: Optional[dict] = None,
    max_critique_tokens: int = 2048,
    critique_temperature: float = 0.0,
    critique_top_p: float = 1.0,
) -> DataProto:
    """Prepare inputs for critique distillation.
    
    This function:
    1. Calls an external vLLM server to generate critiques based on (problem, solution, ground_truth)
    2. Re-tokenizes the input as: [user: problem, assistant: solution, user: critique]
    3. Concatenates with the original response tokens
    
    The resulting ref_input_ids will be used to compute ref log probs, representing
    "what the actor model thinks about its solution after seeing the critique".
    
    Args:
        batch: The data batch containing:
            - raw_prompt (non_tensor_batch): Original messages list
            - responses: Generated response token ids from rollout  
            - response_mask: Mask for valid response tokens
            - reward_model.ground_truth (non_tensor_batch): Ground truth answers
        tokenizer: The tokenizer to use for re-tokenization
        critique_vllm_url: URL of the external vLLM server for critique generation
        critique_model: Model name for the critique server
        critique_prompt_template: Optional custom template for critique generation
        apply_chat_template_kwargs: Additional kwargs for apply_chat_template
        max_critique_tokens: Maximum tokens for critique generation
        critique_temperature: Temperature for critique generation
        
    Returns:
        DataProto: Updated batch with ref_input_ids, ref_attention_mask, ref_position_ids
    """
    if ref_apply_chat_template_kwargs is None:
        ref_apply_chat_template_kwargs = {}
    
    if "raw_prompt" not in batch.non_tensor_batch:
        raise ValueError(
            "raw_prompt not found in batch.non_tensor_batch. "
            "Please set data.return_raw_chat=True in config to enable critique distillation."
        )
    
    batch_size = len(batch)
    raw_prompts = batch.non_tensor_batch["raw_prompt"]  # List of messages
    responses = batch.batch["responses"]  # (batch_size, response_length)
    response_length = responses.shape[1]
    
    # Decode responses to text
    response_texts = []
    for i in range(batch_size):
        # Get valid response tokens using response_mask if available
        if "response_mask" in batch.batch:
            valid_len = int(batch.batch["response_mask"][i].sum().item())
            valid_response_ids = responses[i, :valid_len]
        else:
            valid_response_ids = responses[i]
        response_text = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        response_texts.append(response_text)
    
    # Extract problems from raw_prompts
    problems = []
    for i in range(batch_size):
        messages = raw_prompts[i]
        if isinstance(messages, np.ndarray):
            messages = list(messages)
        # Extract the user's question from messages
        problem = ""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                problem = msg.get("content", "")
                break
        problems.append(problem)
    
    # Extract ground truths
    ground_truths = []
    for i in range(batch_size):
        data_item = batch[i]
        gt = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", "")
        if gt is None:
            raise ValueError(f"Ground truth not found for sample {i}")
        ground_truths.append(str(gt))
    
    # Step 1: Generate critiques using external vLLM
    print(f"Generating critiques for {batch_size} samples using {critique_vllm_url}")
    critique_client = CritiqueClient(server_url=critique_vllm_url, model=critique_model)
    
    critique_prompts = []
    for i in range(batch_size):
        critique_msgs = build_critique_prompt(
            problem=problems[i],
            solution=response_texts[i],
            ground_truth=ground_truths[i],
            critique_prompt_template=critique_prompt_template,
        )
        critique_prompts.append(critique_msgs)
    
    # Batch generate critiques
    critiques = critique_client.batch_generate_critiques(
        batch_messages=critique_prompts,
        max_tokens=max_critique_tokens,
        temperature=critique_temperature,
        top_p=critique_top_p,
    )
    
    # Log critique generation stats
    valid_critiques = sum(1 for c in critiques if c is not None)
    print(f"Generated {valid_critiques}/{batch_size} critiques successfully")
    
    # Step 2: Build multi-turn conversation and tokenize
    # Format: [user: problem, assistant: solution, user: critique]
    ref_prompt_ids_list = []
    
    for i in range(batch_size):
        messages = raw_prompts[i]
        if isinstance(messages, np.ndarray):
            messages = list(messages)

        # Build multi-turn conversation
        # Original: [{"role": "user", "content": problem}]
        # New: [{"role": "user", "content": problem}, 
        #       {"role": "assistant", "content": solution},
        #       {"role": "user", "content": critique}]
        critique = critiques[i] if critiques[i] is not None else "Please review your solution."
        
        multi_turn_messages = list(messages) + [
            {"role": "assistant", "content": response_texts[i]},
            {"role": "user", "content": f"Here is some external feedback for your previous solution:\n<Feedback Begin>\n{critique}\n</Feedback End>" + "\n\nPlease re-generate the solution based on the feedback.\nYou should reuse the previous reasoning steps as long as they are correct, but revise any incorrect steps based on the feedback.\nPlease directly start your new solution by following the same structure and tone as the previous solution, without mentioning any other text."},
        ]

        # Apply chat template
        ref_prompt_str = tokenizer.apply_chat_template(
            multi_turn_messages,
            add_generation_prompt=True,
            tokenize=False,
            **ref_apply_chat_template_kwargs
        )
        
        # Tokenize
        ref_prompt_output = tokenizer(
            ref_prompt_str,
            return_tensors="pt",
            add_special_tokens=False
        )
        ref_prompt_ids = ref_prompt_output["input_ids"][0]
        ref_prompt_ids_list.append(ref_prompt_ids)
    
    # Step 3: Pad prompts (left padding)
    max_prompt_len = max(len(ids) for ids in ref_prompt_ids_list)
    
    ref_prompt_ids_padded = []
    ref_prompt_attention_mask_padded = []
    
    for ref_prompt_ids in ref_prompt_ids_list:
        prompt_len = len(ref_prompt_ids)
        pad_len = max_prompt_len - prompt_len
        
        if pad_len > 0:
            padding = torch.full((pad_len,), tokenizer.pad_token_id, dtype=ref_prompt_ids.dtype)
            ref_prompt_ids_padded_item = torch.cat([padding, ref_prompt_ids], dim=0)
            attention_mask = torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                torch.ones(prompt_len, dtype=torch.long)
            ], dim=0)
        else:
            ref_prompt_ids_padded_item = ref_prompt_ids
            attention_mask = torch.ones(prompt_len, dtype=torch.long)
        
        ref_prompt_ids_padded.append(ref_prompt_ids_padded_item)
        ref_prompt_attention_mask_padded.append(attention_mask)
    
    # Stack into tensors
    ref_prompt_ids_tensor = torch.stack(ref_prompt_ids_padded, dim=0)
    ref_prompt_attention_mask_tensor = torch.stack(ref_prompt_attention_mask_padded, dim=0)
    
    # Step 4: Concat prompt with original responses
    ref_input_ids_tensor = torch.cat([ref_prompt_ids_tensor, responses], dim=1)
    # Response attention mask
    if "response_mask" in batch.batch:
        response_attention_mask = batch.batch["response_mask"]
    else:
        response_attention_mask = torch.ones_like(responses, dtype=torch.long)
    
    ref_attention_mask_tensor = torch.cat([ref_prompt_attention_mask_tensor, response_attention_mask], dim=1)
    
    # Step 5: Compute position_ids
    ref_position_ids_tensor = compute_position_id_with_mask(ref_attention_mask_tensor)
    
    # Add to batch
    batch.batch["ref_input_ids"] = ref_input_ids_tensor
    batch.batch["ref_attention_mask"] = ref_attention_mask_tensor
    batch.batch["ref_position_ids"] = ref_position_ids_tensor

    # # Store critiques for debugging/logging
    # batch.non_tensor_batch["critiques"] = np.array(critiques, dtype=object)
    
    print(
        f"Context distillation: Generated {valid_critiques}/{batch_size} critiques. "
        f"ref_input_ids shape={ref_input_ids_tensor.shape}, "
        f"ref_attention_mask shape={ref_attention_mask_tensor.shape}"
    )
    
    return batch


def prepare_ref_model_inputs(
    batch: DataProto,
    ref_tokenizer,
    apply_chat_template_kwargs: Optional[dict] = None,
) -> DataProto:
    """Prepare input_ids, attention_mask, position_ids for reference model.
    
    When the reference model uses a different prompt template than the actor model,
    we need to re-tokenize the prompts with the ref model's tokenizer and chat template,
    then concatenate with the original response_ids.
    
    Args:
        batch (DataProto): The data batch containing:
            - raw_prompt (in non_tensor_batch): The original messages list (if return_raw_chat=True)
            - responses: The generated response token ids from rollout
            - input_ids, attention_mask, position_ids: Actor model inputs
        ref_tokenizer: The tokenizer used by the reference model (for encoding new prompts)
        apply_chat_template_kwargs (dict, optional): Additional kwargs for apply_chat_template
        
    Returns:
        DataProto: Updated batch with ref_input_ids, ref_attention_mask, ref_position_ids added
    """
    if apply_chat_template_kwargs is None:
        apply_chat_template_kwargs = {}
    
    # Check if raw_prompt is available
    if "raw_prompt" not in batch.non_tensor_batch:
        raise ValueError(
            "raw_prompt not found in batch.non_tensor_batch. "
            "Please set data.return_raw_chat=True in config to enable re-tokenization for ref model."
        )
    
    batch_size = len(batch)
    raw_prompts = batch.non_tensor_batch["raw_prompt"]  # List of messages
    responses = batch.batch["responses"]  # (batch_size, response_length)
    response_length = responses.shape[1]
    
    # Step 1: Tokenize all prompts with ref tokenizer's chat template
    ref_prompt_ids_list = []
    
    for i in range(batch_size):
        # Get the raw messages for this sample
        messages = raw_prompts[i]
        if not isinstance(messages, (list, np.ndarray)):
            raise TypeError(f"raw_prompt must be a list or numpy array, got {type(messages)}")
        messages = list(messages)
        
        # Apply chat template to get the prompt string using ref tokenizer
        ref_prompt_str = ref_tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        # Tokenize prompt
        ref_prompt_output = ref_tokenizer(
            ref_prompt_str, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        ref_prompt_ids = ref_prompt_output["input_ids"][0]
        ref_prompt_ids_list.append(ref_prompt_ids)
    
    # Step 2: Find max prompt length and left pad all prompts
    max_prompt_len = max(len(ids) for ids in ref_prompt_ids_list)
    
    ref_prompt_ids_padded = []
    ref_prompt_attention_mask_padded = []
    
    for ref_prompt_ids in ref_prompt_ids_list:
        prompt_len = len(ref_prompt_ids)
        pad_len = max_prompt_len - prompt_len
        
        if pad_len > 0:
            # Left pad prompt ids
            padding = torch.full((pad_len,), ref_tokenizer.pad_token_id, dtype=ref_prompt_ids.dtype)
            ref_prompt_ids_padded_item = torch.cat([padding, ref_prompt_ids], dim=0)
            # Create attention mask: 0 for padding, 1 for real tokens
            attention_mask = torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                torch.ones(prompt_len, dtype=torch.long)
            ], dim=0)
        else:
            ref_prompt_ids_padded_item = ref_prompt_ids
            attention_mask = torch.ones(prompt_len, dtype=torch.long)
        
        ref_prompt_ids_padded.append(ref_prompt_ids_padded_item)
        ref_prompt_attention_mask_padded.append(attention_mask)
    
    # Stack into tensors
    ref_prompt_ids_tensor = torch.stack(ref_prompt_ids_padded, dim=0)  # (batch_size, max_prompt_len)
    ref_prompt_attention_mask_tensor = torch.stack(ref_prompt_attention_mask_padded, dim=0)  # (batch_size, max_prompt_len)
    
    # Step 3: Concat prompt with original responses
    # responses are already right-padded from rollout
    ref_input_ids_tensor = torch.cat([ref_prompt_ids_tensor, responses], dim=1)  # (batch_size, max_prompt_len + response_length)
    
    # Create attention mask for responses (1 for non-padding tokens)
    # Response attention mask should be derived from the original attention mask or response_mask
    if "response_mask" in batch.batch:
        response_attention_mask = batch.batch["response_mask"]
    else:
        # Fallback: assume all response tokens are valid (no padding)
        response_attention_mask = torch.ones_like(responses, dtype=torch.long)
    
    ref_attention_mask_tensor = torch.cat([ref_prompt_attention_mask_tensor, response_attention_mask], dim=1)
    
    # Step 4: Compute position_ids
    ref_position_ids_tensor = compute_position_id_with_mask(ref_attention_mask_tensor)
    
    # Add to batch
    batch.batch["ref_input_ids"] = ref_input_ids_tensor
    batch.batch["ref_attention_mask"] = ref_attention_mask_tensor
    batch.batch["ref_position_ids"] = ref_position_ids_tensor
    
    print(
        f"Original model input_ids shape={batch.batch['input_ids'].shape}, "
        f"Original model attention_mask shape={batch.batch['attention_mask'].shape}, "
        f"Original model position_ids shape={batch.batch['position_ids'].shape}, "
        f"Prepared ref model inputs: ref_input_ids shape={ref_input_ids_tensor.shape}, "
        f"ref_attention_mask shape={ref_attention_mask_tensor.shape}, "
        f"ref_position_ids shape={ref_position_ids_tensor.shape}"
    )
    
    return batch


def prepare_ref_model_inputs_based_on_correct_solution(
    batch: DataProto,
    tokenizer,
    apply_chat_template_kwargs: Optional[dict] = None,
) -> DataProto:
    """Prepare ref model inputs based on correct (reference) solution.
    
    When the dataset contains ref_solution in extra_info, the teacher distribution
    is the ref model seeing the ref solution as context before evaluating student rollouts.
    
    Format: [user: problem, assistant: ref_solution, user: problem] + student response
    """
    if apply_chat_template_kwargs is None:
        apply_chat_template_kwargs = {}
    
    # Check if raw_prompt is available
    if "raw_prompt" not in batch.non_tensor_batch:
        raise ValueError(
            "raw_prompt not found in batch.non_tensor_batch. "
            "Please set data.return_raw_chat=True in config to enable re-tokenization for ref model."
        )
    
    batch_size = len(batch)
    raw_prompts = batch.non_tensor_batch["raw_prompt"]
    
    # batch.non_tensor_batch["extra_info"] is an array type, each element is a dict
    extra_info_array = batch.non_tensor_batch["extra_info"]
    ref_solutions = []
    for i in range(batch_size):
        extra_info = extra_info_array[i]
        if "ref_solution" not in extra_info:
            raise ValueError(
                f"ref_solution not found in batch.non_tensor_batch['extra_info'][{i}]. "
            )
        ref_solutions.append(extra_info["ref_solution"])
    responses = batch.batch["responses"]  # (batch_size, response_length)
    response_length = responses.shape[1]

    # Extract problems from raw_prompts
    problems = []
    for i in range(batch_size):
        messages = raw_prompts[i]
        if isinstance(messages, np.ndarray):
            messages = list(messages)
        # Extract the user's question from messages
        problem = ""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                problem = msg.get("content", "")
                break
        problems.append(problem)

    ref_prompt_ids_list = []
    
    for i in range(batch_size):
        messages = raw_prompts[i]
        if isinstance(messages, np.ndarray):
            messages = list(messages)
        
        # Build multi-turn conversation
        # Original: [{"role": "user", "content": problem}]
        # New: [{"role": "user", "content": problem}, 
        #       {"role": "assistant", "content": solution},
        #       {"role": "user", "content": critique}]
        # multi_turn_messages = [
        #     {"role": "user", "content": problems[i]},
        #     {"role": "assistant", "content": ref_solutions[i]},
        #     {"role": "user", "content": problems[i]},
        # ]

        # still single-turn conversation
        current_user_prompt = problems[i] + "\n\nHere is a reference solution:\n" + ref_solutions[i] + "\n\nAfter understanding the reference solution, please try to solve this problem using your own approach below."
        multi_turn_messages = [
            {"role": "user", "content": current_user_prompt},
        ]

        # Apply chat template
        ref_prompt_str = tokenizer.apply_chat_template(
            multi_turn_messages,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )

        # # fix tokenizer issue of Qwen3-30B-A3B-Thinking-2507
        # ref_prompt_str = ref_prompt_str.strip("<think>\n") # optional

        # Tokenize
        ref_prompt_output = tokenizer(
            ref_prompt_str,
            return_tensors="pt",
            add_special_tokens=False
        )
        ref_prompt_ids = ref_prompt_output["input_ids"][0]
        ref_prompt_ids_list.append(ref_prompt_ids)
    
    # Step 3: Pad prompts (left padding)
    max_prompt_len = max(len(ids) for ids in ref_prompt_ids_list)
    
    ref_prompt_ids_padded = []
    ref_prompt_attention_mask_padded = []
    
    for ref_prompt_ids in ref_prompt_ids_list:
        prompt_len = len(ref_prompt_ids)
        pad_len = max_prompt_len - prompt_len
        
        if pad_len > 0:
            padding = torch.full((pad_len,), tokenizer.pad_token_id, dtype=ref_prompt_ids.dtype)
            ref_prompt_ids_padded_item = torch.cat([padding, ref_prompt_ids], dim=0)
            attention_mask = torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                torch.ones(prompt_len, dtype=torch.long)
            ], dim=0)
        else:
            ref_prompt_ids_padded_item = ref_prompt_ids
            attention_mask = torch.ones(prompt_len, dtype=torch.long)
        
        ref_prompt_ids_padded.append(ref_prompt_ids_padded_item)
        ref_prompt_attention_mask_padded.append(attention_mask)
    
    # Stack into tensors
    ref_prompt_ids_tensor = torch.stack(ref_prompt_ids_padded, dim=0)
    ref_prompt_attention_mask_tensor = torch.stack(ref_prompt_attention_mask_padded, dim=0)
    
    # Step 4: Concat prompt with original responses
    ref_input_ids_tensor = torch.cat([ref_prompt_ids_tensor, responses], dim=1)
    # Response attention mask
    if "response_mask" in batch.batch:
        response_attention_mask = batch.batch["response_mask"]
    else:
        response_attention_mask = torch.ones_like(responses, dtype=torch.long)
    
    ref_attention_mask_tensor = torch.cat([ref_prompt_attention_mask_tensor, response_attention_mask], dim=1)
    
    # Step 5: Compute position_ids
    ref_position_ids_tensor = compute_position_id_with_mask(ref_attention_mask_tensor)
    
    # Add to batch
    batch.batch["ref_input_ids"] = ref_input_ids_tensor
    batch.batch["ref_attention_mask"] = ref_attention_mask_tensor
    batch.batch["ref_position_ids"] = ref_position_ids_tensor
    
    print(
        f"Original model input_ids shape={batch.batch['input_ids'].shape}, "
        f"Original model attention_mask shape={batch.batch['attention_mask'].shape}, "
        f"Original model position_ids shape={batch.batch['position_ids'].shape}, "
        f"Prepared ref model inputs: ref_input_ids shape={ref_input_ids_tensor.shape}, "
        f"ref_attention_mask shape={ref_attention_mask_tensor.shape}, "
        f"ref_position_ids shape={ref_position_ids_tensor.shape}"
    )
    
    return batch
