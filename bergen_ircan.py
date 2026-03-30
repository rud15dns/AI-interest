'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''
import json
from pathlib import Path
import torch
import gc
from abc import ABC, abstractmethod
from modules.dataset import Tokenized_Sorted_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from jinja2.exceptions import TemplateError
from functools import partial
import random
import numpy as np
import torch.nn.functional as F
from collections import Counter
import os
import hashlib
import pickle
from .neuron_strategies import NeuronStrategies

class Generator(ABC):
    def __init__(self,
                 model_name: str = None,
                 batch_size: int = 1,
                 max_new_tokens: int = 1,
                 max_doc_len: int = 10**10,
                 max_length: int = None,
                 use_middle_truncation: bool = False):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.max_doc_len = max_doc_len
        self.max_length = max_length
        self.use_middle_truncation = use_middle_truncation

    @abstractmethod
    def generate(self, inp):
        pass
    
    @abstractmethod
    def collate_fn(self, inp):
        pass

    def pos_list2str(self, pos_list):
        return '@'.join([str(pos) for pos in pos_list])

    def pos_str2list(self, pos_str):
        return [int(pos) for pos in pos_str.split('@')]
    
    def scaled_input(self, minimum_activations, maximum_activations, batch_size, num_batch):
        """IRCAN을 위한 scaled input 생성"""
        num_points = batch_size * num_batch
        step = (maximum_activations - minimum_activations) / num_points
        res = torch.cat([torch.add(minimum_activations, step * i) for i in range(num_points)], dim=0)
        return res, step[0]
    
    def make_wo_instruction(self, instruction):
        """
        instruction에서 Background부터 Question 이전까지의 내용을 삭제하는 함수
        """
        if "Background" in instruction and "Question" in instruction:
            start = instruction.find("Background")
            end = instruction.find("Question")

            before_bg = instruction[:start]
            from_question = instruction[end:]
            wo_instruction = before_bg + from_question
        
        else:
            wo_instruction = instruction

        wo_instruction = wo_instruction.replace(
            "Your task is to extract relevant information from provided documents and to answer to questions as briefly as possible",
            "Answer the questions as briefly as possible"  
        )

        # print(wo_instruction)
        return wo_instruction.strip()
    
    def get_cache_key(self, dataset, model_name, batch_size):
        dataset_str = str(len(dataset)) + str(type(dataset))
        cache_str = f"{dataset_str}_{model_name}_{batch_size}"
        return hashlib.md5(cache_str.encode()).hexdigest()

    def save_ircan_cache(self, cache_data, cache_key, cache_dir="./ircan_cache"):
        """IRCAN 결과를 캐시 파일로 저장"""
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"ircan_{cache_key}.pkl")
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"IRCAN 결과가 캐시되었습니다: {cache_path}")
        return cache_path

    def load_ircan_cache(self, cache_key, cache_dir="./ircan_cache"):
        """캐시된 IRCAN 결과 로드"""
        cache_path = os.path.join(cache_dir, f"ircan_{cache_key}.pkl")
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            print(f"캐시된 IRCAN 결과를 로드했습니다: {cache_path}")
            return cache_data
        return None
        
        
    def get_context_attr(self, idx, prompt_without_context, prompt_with_context, answer_obj):
        """bergen에 맞춘 IRCAN 적용하기"""
        """
        Args:
        idx: query의 인덱스
        """
        device = self.model.device
        class Args:
            batch_size = 20
            num_batch = 1
            batch_size_per_inference = 2

        args = Args()

        tokens = self.tokenizer.tokenize(answer_obj)
        gold_label_id = self.tokenizer.convert_tokens_to_ids(tokens[0])
        
        # 문맥에 민감한 뉴런을 찾는다. 
        wo_tokenized_inputs = self.tokenizer(prompt_without_context, return_tensors="pt")
        wo_input_ids = wo_tokenized_inputs["input_ids"].to(device)
        wo_attention_mask = wo_tokenized_inputs["attention_mask"].to(device)
        
        w_tokenized_inputs = self.tokenizer(prompt_with_context, return_tensors="pt")
        w_input_ids = w_tokenized_inputs["input_ids"].to(device)
        w_attention_mask = w_tokenized_inputs["attention_mask"].to(device)

        # print(f"wo_input_ids shape: {wo_input_ids.shape}")
        # print(f"w_input_ids shape: {w_input_ids.shape}")
        # print(f"wo length: {len(prompt_without_context)}")
        # print(f"w length: {len(prompt_with_context)}")
        # breakpoint()

        # record results
        res_dict = {
            'idx': idx,
            'wo_all_ffn_activations': [],
            'w_all_ffn_activations': [],
            'all_attr_gold': [],
        }

        for tgt_layer in range(self.model.model.config.num_hidden_layers):
            wo_ffn_activations_dict = dict()
            def wo_forward_hook_fn(module, inp, outp): # Forward 시에 input값을 저장.
                wo_ffn_activations_dict['input'] = inp[0]  # inp type is Tuple

            w_ffn_activations_dict = dict()
            def w_forward_hook_fn(module, inp, outp):
                w_ffn_activations_dict['input'] = inp[0]
            # ========================== get activations when there is no context in the prompt =========================
            # x -> up_proj -> activation -> down_proj -> output(logit)
            # 문맥에 따라서 뉴런이 어떤 출력을 내는지 보려고
            wo_hook = self.model.model.layers[tgt_layer].mlp.down_proj.register_forward_hook(wo_forward_hook_fn)

            # 어떤 뉴런이 활성화되었는지만 확인하면 되기 때문에, gradient를 계산할 필요가 없다. 
            with torch.no_grad():
                wo_outputs = self.model(input_ids=wo_input_ids, attention_mask=wo_attention_mask)

            wo_ffn_activations = wo_ffn_activations_dict['input']
            wo_ffn_activations = wo_ffn_activations[:, -1, :] # 
            wo_logits = wo_outputs.logits[:, -1, :] # 
            wo_hook.remove()
            
            # =========================== get activations when there is context in the prompt ============================
            w_hook = self.model.model.layers[tgt_layer].mlp.down_proj.register_forward_hook(w_forward_hook_fn)
            with torch.no_grad():
                w_outputs = self.model(input_ids=w_input_ids, attention_mask=w_attention_mask)
            w_ffn_activations = w_ffn_activations_dict['input']
            w_ffn_activations = w_ffn_activations[:, -1, :]
            w_logits = w_outputs.logits[:, -1, :]
            w_hook.remove()
            
            wo_ffn_activations.requires_grad_(True)
            w_ffn_activations.requires_grad_(True)

            scaled_activations, activations_step = self.scaled_input(wo_ffn_activations, w_ffn_activations, args.batch_size, args.num_batch)


            scaled_activations.requires_grad_(True)

            # integrated grad at the gold label for each layer
            ig_gold = None

            # 시그마 k=1 부터 m까지
            for batch_idx in range(args.num_batch):
                grad = None
                all_grads = None
                for i in range(0, args.batch_size, args.batch_size_per_inference):
                    # P [] <- []에 들어갈 값. start -> end 그 사이의 값. scaled_actviations : v + a (v`- v)이 모아져있음.
                    batch_activations = scaled_activations[i: i + args.batch_size_per_inference] # (batch_size_per_inference, ffn_size) # 그것들을 배치 사이즈로 처리하려고 묶기

                    # print(f"=== Step {i} Debug ===")
                    # print(f"batch_activations shape: {batch_activations.shape}")
                    # print(f"w_ffn_activations shape: {w_ffn_activations.shape}")
                    # print(f"args.batch_size_per_inference: {args.batch_size_per_inference}")

                    # batch_size_per_inference 만큼 복제.
                    batch_w_activations = w_ffn_activations.repeat(args.batch_size_per_inference, 1) # (batch_size_per_inference, ffn_size)
                    # print(f"batch_w_activations.shape={batch_w_activations.shape}")

                    batched_w_input_ids = w_input_ids.repeat(args.batch_size_per_inference, 1) # (batch_size_per_inference, seq_len)
                    batched_w_attention_mask = w_attention_mask.repeat(args.batch_size_per_inference, 1) # (batch_size_per_inference, seq_len)

                    # print(f"batched_w_input_ids shape: {batched_w_input_ids.shape}")
                    # breakpoint()

                    def i_forward_hook_change_fn(module, inp):
                        inp0 = inp[0]
                        # print(f"[HOOK] Before change inp0[:, -1, :5]: {inp0[:, -1, :5]}")
                        inp0[:, -1, :] = batch_activations # 해당 레이어에서 마지막 위치의 뉴런의 activation을 바꿔줘서 모델의 최종값이 바뀌도록 한다. 
                        # print(f"[HOOK] After change inp0[:, -1, :5]: {inp0[:, -1, :5]}")
                        inp = tuple([inp0])
                        
                        return inp
                    change_hook = self.model.model.layers[tgt_layer].mlp.down_proj.register_forward_pre_hook(i_forward_hook_change_fn) # context와 question에 대해서 해당 layer에서 activation을 바꿔준다.
                    
                    # 똑같은 문장을 batch_size_per_inference 만큼 복제해서 넣어서 activation을 한번에 바꿔서 계산하는 효과 -> 시간 효율적. 
                    outputs = self.model(input_ids=batched_w_input_ids, attention_mask=batched_w_attention_mask)  # (batch, n_vocab), (batch, ffn_size)

                    # compute final grad at a layer at the last position
                    tgt_logits = outputs.logits[:, -1, :] # (batch, n_vocab)
                    tgt_probs = F.softmax(tgt_logits, dim=1) # (batch, n_vocab)

                    # print(f"tgt_probs shape: {tgt_probs.shape}")
                    # print(f"gold_label_id: {gold_label_id}")
                    # print(f"tgt_probs[:, gold_label_id] shape: {tgt_probs[:, gold_label_id].shape}")
                    # print(f"w_ffn_activations for gradient shape: {w_ffn_activations.shape}")

                    # grads_i = torch.autograd.grad(torch.unbind(tgt_probs[:, gold_label_id]), batch_activations) 정답 확률만 꺼내서, 뉴런의 출력값에 대해 gradient를 구한다. 편미분 과정
                    grads_i = torch.autograd.grad(torch.unbind(tgt_probs[:, gold_label_id]), w_ffn_activations, retain_graph=True) # grads_i[0].shape: (1, ffn_size)
                    # breakpoint()
                    del tgt_probs

                    change_hook.remove()  # check_ffn_activations_dict['output'][:,-1,:]
                    
                    all_grads = grads_i[0] if all_grads is None else torch.cat((all_grads, grads_i[0]), dim=0) # 각 스텝별 gradient를 모은다.
                    # print(f"grads_i[0].shape: {grads_i[0].shape}")
                    # print(f"all_grads.shape: {all_grads.shape}") 

                grad = all_grads.sum(dim=0)  # (ffn_size) # 현재 배치에서 gradient를 모두 더한다.
                # print(f"grad.shape: {grad.shape}")  # (ffn_size)
                ig_gold = grad if ig_gold is None else torch.add(ig_gold, grad)  # (ffn_size) <- 해당 레이어에서의 IG 값 
                
            ig_gold = ig_gold * activations_step # 곱해주는 부분.
            res_dict['wo_all_ffn_activations'].append(wo_ffn_activations.squeeze().tolist())
            res_dict['w_all_ffn_activations'].append(w_ffn_activations.squeeze().tolist())
            res_dict['all_attr_gold'].append(ig_gold.tolist())
            
        return res_dict
    
    def convert_to_triplet_ig(self, ig_list):
        ig_triplet = []
        ig = np.array(ig_list) # (layer_num, ffn_size)
        max_ig = ig.max()  # maximum attribution score
        for i in range(ig.shape[0]):
            for j in range(ig.shape[1]):
                if ig[i][j] >= max_ig * 0.1: #3.2에서 최댓값의 10% 이상인 것들만 걸러내는 부분.
                    ig_triplet.append([i, j, ig[i][j]])
        return ig_triplet
    
    def convert_all_neurons_attribution_score_to_triplet_ig(self, ig_list):
        ig_triplet = []
        ig = np.array(ig_list) # (layer_num, ffn_size)
        for i in range(ig.shape[0]):
            for j in range(ig.shape[1]):
                ig_triplet.append([i, j, ig[i][j]])
        return ig_triplet    

    def save_model(self, experiment_mode, train_dataset_name, enhance_strength, top_k):
        """
        save enhanced model
        """
        save_dir = Path("enhanced_models")
        save_dir.mkdir(exist_ok=True)

        model_name = f"{train_dataset_name}_{experiment_mode}_strength{enhance_strength}_top_k{top_k}"
        model_path = save_dir / model_name
        # 모델 가중치 저장
        self.model.save_pretrained(model_path)
        # breakpoint()

    def load_custom_model(self, model_name):
        model_path = Path("/home/data/chaewon/enhanced_models") / model_name
        print(model_path)

        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.model.eval()
        print(f"   Device: {self.model.device}")
        # breakpoint()

    def final_evaluate(self, eval_dataset, experiment_mode, train_dataset_name, use_ircan, original_use_cad, save_result):
        # 그리고 이 모델을 가지고 다시 응답을 생성한다. 
        w_responses, w_instructions = list(), list()
        
        print("데이터셋 길이 확인: ", len(eval_dataset))
        
        query_ids, queries, labels, ranking_labels = list(), list(), list(), list()
        results = list()
        self.use_cad = original_use_cad

        with torch.no_grad():
            if self.tokenizer:
                enhanced_dataset = Tokenized_Sorted_Dataset(eval_dataset, self, training=False)
                enhanced_dataloader = DataLoader(enhanced_dataset, batch_size=self.batch_size, 
                                            collate_fn=partial(self.collate_fn, eval=True), num_workers=4)
            else:
                enhanced_dataloader = DataLoader(eval_dataset, batch_size=self.batch_size, 
                                            collate_fn=partial(self.collate_fn, eval=True), num_workers=4)
                
            for data_dict in tqdm(enhanced_dataloader, desc='Final Evaluating'):
                id_ = data_dict['q_id']
                w_instruction = data_dict['instruction']

                query_ids += id_
                label = data_dict['label']
                labels += label
                queries += data_dict['query']
                ranking_labels += data_dict['ranking_label']

                w_instructions += w_instruction

                
                if self.use_cad:
                    wo_instruction = []
                    for query in data_dict['query']:
                        direction = f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant. Answer the questions as briefly as possible.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nQuestion: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
                        wo_instruction.append(direction)
                    
                    wo_tokenized = self.tokenizer(
                        wo_instruction,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_length if hasattr(self, 'max_length') and self.max_length else None
                    )
                    w_generated_response = self.generate(
                        data_dict['model_input'],
                        wo_tokenized
                    )
                else:
                    w_generated_response = self.generate(
                        data_dict['model_input']
                    )

                w_responses += w_generated_response
                # batch의 각 항목을 개별로 저장
                for i in range(len(id_)):
                    results.append({
                        'q_id': id_[i],
                        'model_input': w_instruction[i],
                        'response': w_generated_response[i],
                        'label': label[i]
                    })  

                torch.cuda.empty_cache()
                gc.collect()
        
        if save_result:
            output_dir = Path("response_logs")
            output_dir.mkdir(exist_ok=True)

            # results 저장
            output_file = output_dir / f"{train_dataset_name}_use_ircan_{use_ircan}_use_cad_{self.use_cad}_no_context_{self.no_context}_exp_{experiment_mode}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
        return query_ids, queries, w_instructions, w_responses, labels, ranking_labels

        
    ## Attr 모두 구하기
    ## ㅁ ㅁ ㅁ ㅁ ㅁ -> 4개 변화량 구해서 평균 구하기
    def eval(self, eval_dataset, 
            train_dataset=None, 
            experiment_mode='original', 
            train_dataset_name=None, 
            enhance_strength=5.0, 
            top_k=50,
            save_model=False,
            load_model_name=None,
            save_result=False):
        """
        eval_dataset: 평가할 데이터셋 (dev/test)
        train_dataset: CN 발견용 데이터셋 (train, optional)
        save_model: True면은 강화된 모델을 저장
        load_model_name: 불러오고 싶은 모델이 주어지면 해당 모델을 로드하고 Final Evaluating만.
        """
        use_ircan = train_dataset is not None
        original_use_cad = self.use_cad
        self.use_cad = False

        if load_model_name is not None:
            # breakpoint()
            self.load_custom_model(load_model_name)
            return self.final_evaluate(
                eval_dataset,
                experiment_mode,
                train_dataset_name, 
                use_ircan, 
                original_use_cad,
                save_result
            )
        
        print("모델 로드 안함")
        if use_ircan: 
            print("IRCAN 시작함")
            train_dataset_len = len(train_dataset)
            with torch.no_grad():
                if self.tokenizer: # kilt_nq train_data인지, 아닌지. 
                    tokenized_and_sorted_dataset = Tokenized_Sorted_Dataset(train_dataset, self, training=False)
                    dataloader = DataLoader(tokenized_and_sorted_dataset, batch_size=self.batch_size, 
                                            collate_fn=partial(self.collate_fn, eval=True), num_workers=4)
                else:
                    dataloader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                            collate_fn=partial(self.collate_fn, eval=True), num_workers=4)
                
                wo_responses, wo_instructions, w_responses, w_instructions = list(), list(), list(), list() 
                query_ids, queries, labels, ranking_labels =  list(), list(), list(), list()
                
                for data_dict in tqdm(dataloader, desc='Generating wo/w context(TRAIN)'):
                    id_ = data_dict['q_id']
                    w_instruction = data_dict['instruction']

                    query_ids += id_
                    label = data_dict['label']
                    labels += label
                    queries += data_dict['query']
                    ranking_labels += data_dict['ranking_label']
                    w_instructions += w_instruction 

                    # wo_instruction 생성
                    wo_instruction = list()
                    for query in data_dict['query']:
                        # llama 버전
                        direction = f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant. Answer the questions as briefly as possible.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nQuestion: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
                        wo_instruction.append(direction)
                    wo_instructions += wo_instruction

                    w_generated_response = self.generate(data_dict['model_input'])
                    w_responses += w_generated_response

                    wo_tokenized = self.tokenizer(
                        wo_instruction,  # 문자열 리스트
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_length if hasattr(self, 'max_length') and self.max_length else None
                    )

                    wo_generated_response = self.generate(wo_tokenized)
                    wo_responses += wo_generated_response

                    torch.cuda.empty_cache()
                    gc.collect()

            # IRCAN 계산
            ircan_results = list()
            all_neuron_stats = {}

            """
            all_neuron_stats = {
                (0, 100): [0.00001, 0.00002, 0.000015, ...],  # 30개 데이터셋의 Attribution 점수
                (0, 101): [0.00003, 0.00002, 0.000018, ...],
                (1, 50): [0.00005, 0.00004, 0.000052, ...],
                ...
            }
            """

            for i in tqdm(range(100), desc="Processing IRCAN"):
                try:
                    res_dict = self.get_context_attr(
                        idx=query_ids[i], 
                        prompt_without_context=wo_instructions[i],
                        prompt_with_context=w_instructions[i],
                        answer_obj=labels[i][0]
                    )
                    # 여기에 확인 코드 추가
                    for layer_idx in range(len(res_dict['all_attr_gold'])):
                        for neuron_idx, score in enumerate(res_dict['all_attr_gold'][layer_idx]):
                            key = (layer_idx, neuron_idx)
                            if key not in all_neuron_stats:
                                all_neuron_stats[key] = []
                            all_neuron_stats[key].append(score)
                            
                    res_dict["all_attr_gold"] = self.convert_to_triplet_ig(res_dict["all_attr_gold"])
                    
                    ircan_results.append(res_dict)
                    
                except Exception as e:
                    print(f"Error processing sample {query_ids[i] if i < len(query_ids) else i}: {e}")

            print(f"\nSelecting neurons for experiment: {experiment_mode}, {enhance_strength}")

            ## Experiment Modes 선택하기
            strategy_map = {
                'exp1': lambda: NeuronStrategies.counter_attribution_desc(ircan_results, top_k, train_dataset_name),
                'exp2': lambda: NeuronStrategies.low_unc_high_attr_enhance(ircan_results, all_neuron_stats, top_k=top_k, train_dataset_name=train_dataset_name),
                'exp3': lambda: NeuronStrategies.low_attr_suppress(ircan_results),
                'exp4': lambda: NeuronStrategies.low_attr_high_unc_counter_enhance(ircan_results, all_neuron_stats, top_k=top_k, train_dataset_name=train_dataset_name),
                'exp5': lambda: NeuronStrategies.high_attr_low_unc_important_layer_enhance(ircan_results, all_neuron_stats, top_k=top_k, train_dataset_name=train_dataset_name),
                'exp7': lambda: NeuronStrategies.weighted_sum_attribution_score_with_unc_enhance(all_neuron_stats, alpha=0.3, top_k=top_k, train_dataset_name=train_dataset_name),
                'exp8': lambda: NeuronStrategies.multiply_attribution_score_with_unc_enhance(all_neuron_stats, top_k=top_k, train_dataset_name=train_dataset_name),
                'exp9': lambda: NeuronStrategies.z_score_subspace_enhance(all_neuron_stats, top_k=top_k, train_dataset_name=train_dataset_name),
                'exp10': lambda: NeuronStrategies.high_attr_high_unc_counter_enhance(ircan_results, all_neuron_stats, top_k=top_k, train_dataset_name=train_dataset_name),
                'exp11': lambda: NeuronStrategies.high_unc_low_attr_counter_suppress(ircan_results, all_neuron_stats, top_k=top_k, train_dataset_name=train_dataset_name),
                'exp12': lambda: NeuronStrategies.dual_uncertainty_combined(ircan_results, all_neuron_stats, enhance_top_k=top_k, suppress_top_k=top_k, train_dataset_name=train_dataset_name)
            }
            
            
            if experiment_mode in strategy_map:
                cns = strategy_map[experiment_mode]()
            
            # enhance 루프 직전에 원래 값 저장
            test_layer, test_neuron = cns[0]
            original_weight = self.model.model.layers[test_layer].mlp.down_proj.weight[:, test_neuron].clone()
            
            for layer, pos in cns: # 아까 저장한 레이어, 뉴런 위치에 대해서 
                with torch.no_grad():
                    self.model.model.layers[layer].mlp.down_proj.weight[:, pos] *= enhance_strength # model의 그 위치의 weight를 강화한다. 
            
            enhanced_weight = self.model.model.layers[test_layer].mlp.down_proj.weight[:, test_neuron]
            if save_model:
                self.save_model(
                    experiment_mode,
                    train_dataset_name,
                    enhance_strength,
                    top_k
                )

        return self.final_evaluate(
            eval_dataset,
            experiment_mode,
            train_dataset_name, 
            use_ircan, 
            original_use_cad,
            save_result
        )

    def get_response(self):
        """
        This replaces the 'generation_prompt' in case the generator does not have a chat_template.
        It's used to prompt and also to identify the label positions to mask prompt in training.
        """
        return '\nResponse:\n'

    def get_response_template_ids(self):
        response_template =  self.get_response()
        return self.tokenizer.encode(response_template, add_special_tokens=False)
    
    def compile_prompt(self, system_prompt: str, user_prompt: str, question: str, docs: str = None, label: str = None):
            """
            Applying the chat template if it exists:
            NB: seemingly unused args are used in the 'eval' call.
            NB: if the label is not None, we assume training=True and then the answer of the model is added to the full prompt
            This method returns a tuple consisting of:
            - the final prompt
            - if a label is provided, the position of the first label index within the tokenized sequence (for masking in training)
            """
            # the prompt should finish with a generation prompt if we are in 'eval' mode i.e. when there is no label
            # NB: the generation prompt is empty (automatically included in the template rather) for llama/mistral/solar at least
            add_generation_prompt = (label is None) 
            
            label_start_index = None
            if self.tokenizer.chat_template is None:
                user_prompt_with_values = eval(user_prompt).replace(':\ ', ': ')
                # we add the 'reponse incitation' to non chat template
                prompt = f"{system_prompt}\n{user_prompt_with_values}" + self.get_response()
                if label is not None:
                    # Compute prompt size in tokens without labels.
                    label_start_index = len(self.tokenizer(prompt, add_special_tokens=False)['input_ids'])
                    prompt += label + self.tokenizer.eos_token

            else:        
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": eval(user_prompt).replace(':\ ', ': ')}
                ]
                try:
                    # Handle the label
                    if label is not None:
                        # Compute the prompt without label, to know its length and hence where to mask in training
                        label_start_index = len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, add_special_tokens=False))
                        messages.append({"role": "assistant", "content": label})
                    
                    prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False)

                except TemplateError as e:
                    if "System role not supported" in str(e):
                        messages = [{"role": "user", "content": messages[0]['content'] + '\n' + messages[1]['content']}]

                        if label is not None:
                            # Compute the prompt without label, to know its length and hence where to mask in training  
                            label_start_index = len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, add_special_tokens=False))
                            messages.append({"role": "assistant", "content": label})

                        prompt = self.tokenizer.apply_chat_template(messages,  add_generation_prompt=add_generation_prompt, tokenize=False)
                    else:
                        raise e
            
            
            if label is not None:
                assert label_start_index is not None # check we did find the prompt length
                if not prompt.endswith(self.tokenizer.eos_token):
                    prompt += self.tokenizer.eos_token # most models have this already, but not gemma-2b !
                
            return prompt, label_start_index

    def middle_truncation(self, docs):
        """
        Truncate documents by removing the middle section while preserving both the beginning and end.
        Args:
            docs (str): The document text to truncate
                
        Returns:
            str: The truncated document text
        """
        if docs is None or self.max_length is None or not hasattr(self, 'tokenizer'):
            return docs
        
        tokenized_docs = self.tokenizer(docs, truncation=False, return_tensors="pt")['input_ids'][0]
        docs_length = len(tokenized_docs)
        
        truncation_threshold = self.max_length - 128
        assert truncation_threshold >= 0, "Truncation threshold must be non-negative. Check max_length value."
        
        if docs_length > truncation_threshold:
            half = int(truncation_threshold / 2)
            
            first_half = tokenized_docs[:half]
            second_half = tokenized_docs[-half:]
            
            first_half_text = self.tokenizer.decode(first_half, skip_special_tokens=True)
            second_half_text = self.tokenizer.decode(second_half, skip_special_tokens=True)
            docs = first_half_text + second_half_text

        return docs


    def format_instruction(self, sample: dict, eval: bool = True) -> (str, int):
        """
        Makes the actual prompt from the prompt template and the model chat template
        Also return start index of the label in that prompt, if eval=True and a label is provided, None otherwise.
        If eval=True, then no label is added to the prompt.
        If eval=False, then the label is added to the prompt, for training (teacher forcing)
        """
        question = sample['query']
        label = None
        if not eval:
            label = (sample['label'] if isinstance(sample['label'], str) else random.choice(sample['label']))
            assert label is not None
        if 'doc' in sample:
            # We have retrieved documents:
            docs = ''
            input_docs = sample['doc']
            input_docs = [doc for doc in input_docs if len(doc.strip()) > 0]
            for i, doc in enumerate(input_docs):
                doc = ' '.join(doc.split()[:self.max_doc_len])
                docs += f"Document {i+1}: {doc}\n"
            if self.use_middle_truncation:
                docs = self.middle_truncation(docs)
            return self.compile_prompt(self.prompt.system, self.prompt.user, question, docs, label=label)
        else:
            # We have no retrieved documents: switch to no doc prompt
            return self.compile_prompt(self.prompt.system_without_docs, self.prompt.user_without_docs, question, label=label)
