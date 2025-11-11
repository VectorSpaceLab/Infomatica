import random, json, re
import string
import numpy as np


def get_unique_entities_iterative(data):
    if not data:
        return []

    entities_set = set()
    nodes_to_visit = [data]
    while nodes_to_visit:
        current_node = nodes_to_visit.pop()
        if 'entity' in current_node:
            entities_set.add(current_node['entity'])
        if 'children' in current_node and current_node['children']:
            nodes_to_visit.extend(current_node['children'])
            
    return list(entities_set)


def make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj

def convert_array_to_list(data):
    if isinstance(data, np.ndarray):
        return [convert_array_to_list(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_array_to_list(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [convert_array_to_list(item) for item in data]
    else:
        return data

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score

def is_valid_sequence(text):
    # NOTE: Qwen2.5 Serise
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)
    
    if not assistant_match:
        return False, "Missing assistant marker"
    
    start_pos = assistant_match.end()
    content = text[start_pos:]
    
    tags_to_check = ["think", "search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"
            
    if not content.lstrip().startswith("<think>"):
        return False, "Sequence must start with <think> tag"

    split_pattern = r"(</?(?:think|search|information|answer)>)"
    parts = re.split(split_pattern, content)
    
    state = "start"  # start -> think -> search -> information -> think -> ... -> answer -> end
    
    for i, part in enumerate(parts):
        if not part.strip():
            continue
            
        if re.match(r"</?(?:think|search|information|answer)>", part):
            if part == "<think>" and state in ["start", "information"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state == "after_think": 
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            if state in ["in_think", "in_search", "in_information", "in_answer"]:
                pass
            elif part.strip():
                return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
    
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"
        
    return True, "Valid sequence format"

def extract_information_blocks(text: str) -> list[str]:
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]

def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def extract_final_answer(text: str) -> str | None:

    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def compute_score_em(solution_str, ground_truth, structure_format_score=0.2, retrieval_score=0.1, process_reward=False, score=1.0, data_item=None, **kwargs):

    do_print = random.random() < 0.003
    dense_reward, dense_reward_pos = 0.0, 0.0
    
    log_data = {
        'solution_str': solution_str,
        'ground_truth': ground_truth['target'],
        'structure_format_score': structure_format_score,
        'retrieval_score': retrieval_score,
        'full_score': score,
        'meta_info': make_serializable(kwargs.get('data', {}))
    }

    is_valid_format, reason = is_valid_sequence(solution_str)
    if not is_valid_format:
        final_score = 0.0
        log_data.update({
            'is_valid_format': False,
            'format_reason': reason,
            'extracted_answer': None,
            'retrieval_correct': False,
            'final_score': final_score
        })

        if do_print:
            print(f"--------------------------------")
            print(f"Golden answers: {ground_truth['target']}")
            print(f"Solution string: {solution_str}")
            print(f"Format INVALID. Reason: {reason}. Reward: 0.0")

        return final_score, dense_reward

    log_data['is_valid_format'] = True
    log_data['format_reason'] = 'Valid'

    answer = extract_final_answer(solution_str)
    log_data['extracted_answer'] = answer
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
        print(f"Format VALID. Extracted answer: {answer}. Expected answer: {ground_truth['target']}")

    if answer is not None and em_check(answer, ground_truth['target']):
        final_score = score

        retrieval_correct = True
        log_data.update({
            'retrieval_correct': retrieval_correct,
            'final_score': final_score
        })

        if do_print:
            print(f"Answer is CORRECT. Reward: {final_score} \n")

        return final_score, dense_reward_pos
    else:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth['target'])
        final_score = structure_format_score  # + (retrieval_score if retrieval_correct else 0)
        
        log_data.update({
            'retrieval_correct': retrieval_correct,
            'final_score': final_score
        })
        if do_print:
            print(f"Answer is WRONG. Retrieval correct: {retrieval_correct}. Process Reward: {dense_reward}. Reward: {final_score} \n")

        return final_score, dense_reward