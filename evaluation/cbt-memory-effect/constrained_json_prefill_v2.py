"""Prospective whitespace-prefill grammar; v1 files/results remain unchanged.

Same complete generic value domain for every question/arm. The fixed assistant
prefix includes a colon-following space. No claim that v1's error was BPE-caused.
"""
import hashlib
import json
import constrained_json_grammar as previous

VERSION='common-whitespace-prefill-json-answer-grammar-v2'
PREFIX='{"answer": '
STATUSES=previous.STATUSES
canonical=previous.canonical
digest=previous.digest


def policy():
    return {'version':VERSION,'schema':previous.policy()['schema'],
        'fixed_assistant_prefix':PREFIX,'prefix_has_colon_following_space':True,
        'prefix_placement':'append separately encoded prefix to chat-template add_generation_prompt input',
        'suffix_language':'generic JSON value followed by closing brace; no leading space',
        'selection_inputs':'generated suffix token prefix only; no arm, question, gold or memory state',
        'eos_policy':'only after complete accepted suffix',
        'domain_is_not_question_specific':True,'is_semantic_validation':False,
        'diagnosis':'v1 null/11 tokenization showed no colon-null merge; no causal BPE claim'}


def suffix_texts():
    return [canonical(v)+'}' for v in [None,*range(10000),*STATUSES]]


def append_prefill_ids(base_ids,prefix_ids,maximum=4096):
    if not base_ids or not prefix_ids or len(base_ids)+len(prefix_ids)>maximum:
        raise ValueError('nonempty_bounded_base_plus_fixed_prefix_required')
    return list(base_ids)+list(prefix_ids)


def input_preflight(tokenizer,messages,trie):
    """No model/label access: hash the actual base-plus-fixed-prefix tensor."""
    inputs=tokenizer.apply_chat_template(messages,add_generation_prompt=True,
        tokenize=True,return_dict=True,return_tensors='pt')
    if set(inputs)-{'input_ids','attention_mask'}:raise ValueError('unsupported_chat_tensor_field')
    if len(inputs['input_ids'].shape)!=2 or inputs['input_ids'].shape[0]!=1:
        raise ValueError('single_example_chat_input_required')
    base=inputs['input_ids'][0].tolist();actual=append_prefill_ids(base,trie.prefix_ids)
    return {'input_tokens':len(actual),'actual_input_token_count':len(actual),
        'base_input_token_count':len(base),'prefilled_token_count':len(trie.prefix_ids),
        'prefilled_token_ids':list(trie.prefix_ids),'input_tensor_sha256':digest([actual]),
        'base_input_tensor_sha256':digest([base])}


class AnswerTrie:
    def __init__(self,tokenizer,eos_token_ids):
        self.eos=sorted(set([eos_token_ids] if type(eos_token_ids) is int else eos_token_ids))
        if not self.eos or any(type(i) is not int or i<0 for i in self.eos):
            raise ValueError('explicit_nonempty_integer_eos_ids_required')
        self.prefix_ids=tokenizer.encode(PREFIX,add_special_tokens=False)
        if (not self.prefix_ids or any(type(i) is not int or i<0 or i in self.eos for i in self.prefix_ids)
            or tokenizer.decode(self.prefix_ids,skip_special_tokens=False)!=PREFIX):
            raise ValueError('fixed_whitespace_prefix_must_roundtrip_without_eos')
        self.root={};self.terminals={};encoded=[]
        for text in suffix_texts():
            ids=tokenizer.encode(text,add_special_tokens=False)
            if (not ids or any(type(i) is not int or i<0 or i in self.eos for i in ids)
                or tokenizer.decode(ids,skip_special_tokens=False)!=text
                or tokenizer.decode(self.prefix_ids+ids,skip_special_tokens=False)!=PREFIX+text):
                raise ValueError('every_suffix_and_prefilled_object_must_roundtrip')
            key=tuple(ids)
            if key in self.terminals: raise ValueError('ambiguous_tokenized_suffixes')
            self.terminals[key]=text;node=self.root
            for token in ids:node=node.setdefault(token,{})
            encoded.append([text,ids])
        self.maximum_candidate_tokens=max(len(k) for k in self.terminals)
        self.identity={'policy':policy(),'policy_sha256':digest(policy()),
            'tokenizer_chat_template_sha256':digest(tokenizer.chat_template),
            'tokenizer_class':type(tokenizer).__name__,'encoded_language_sha256':digest(encoded),
            'fixed_prefix_token_ids':self.prefix_ids,'fixed_prefix_token_count':len(self.prefix_ids),
            'candidate_count':len(encoded),'eos_token_ids':self.eos,
            'maximum_candidate_tokens':self.maximum_candidate_tokens}

    def allowed(self,prefix):
        key=tuple(int(i) for i in prefix);node=self.root
        for token in key:
            if token not in node:raise ValueError('generated_suffix_outside_frozen_grammar')
            node=node[token]
        result=sorted(node)
        if key in self.terminals:result+=self.eos
        if not result:raise ValueError('nonterminal_dead_end')
        return result

    def callback(self,actual_input_tokens):
        def allowed_tokens(batch_id,input_ids):
            if batch_id!=0:raise ValueError('single_example_generation_required')
            return self.allowed(input_ids.tolist()[actual_input_tokens:])
        return allowed_tokens

    def finish(self,token_ids):
        if not token_ids or token_ids[-1] not in self.eos:
            raise ValueError('suffix_did_not_finish_with_eos')
        key=tuple(token_ids[:-1])
        if any(i in self.eos for i in key) or key not in self.terminals:
            raise ValueError('not_one_complete_accepted_suffix')
        text=PREFIX+self.terminals[key]
        return {'response_text':text,'chosen_answer':json.loads(text)['answer'],
            'raw_generated_token_ids':list(token_ids),'fixed_assistant_prefix':PREFIX,
            'prefilled_token_ids':list(self.prefix_ids),'prefilled_token_count':len(self.prefix_ids),
            'decoding_identity_sha256':digest(self.identity)}


def generate(model,tokenizer,messages,trie,*,max_new_tokens=64):
    """Keep source messages unchanged; append the common fixed prefix tokens."""
    import torch
    if max_new_tokens<=trie.maximum_candidate_tokens:
        raise ValueError('output_budget_must_fit_every_suffix_plus_eos')
    inputs=tokenizer.apply_chat_template(messages,add_generation_prompt=True,
        tokenize=True,return_dict=True,return_tensors='pt')
    if set(inputs)-{'input_ids','attention_mask'}:raise ValueError('unsupported_chat_tensor_field')
    if len(inputs['input_ids'].shape)!=2 or inputs['input_ids'].shape[0]!=1:
        raise ValueError('single_example_chat_input_required')
    base_ids=inputs['input_ids'][0].tolist();actual_ids=append_prefill_ids(base_ids,trie.prefix_ids)
    base_count=len(base_ids);actual_count=len(actual_ids)
    inputs['input_ids']=torch.tensor([actual_ids],dtype=inputs['input_ids'].dtype,device=model.device)
    mask=inputs.get('attention_mask')
    if mask is not None:
        if tuple(mask.shape)!=(1,base_count):raise ValueError('attention_mask_shape_changed')
        inputs['attention_mask']=torch.cat((mask.to(model.device),
            torch.ones((1,len(trie.prefix_ids)),dtype=mask.dtype,device=model.device)),dim=-1)
    with torch.inference_mode():
        output=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,use_cache=True,
            pad_token_id=tokenizer.eos_token_id,prefix_allowed_tokens_fn=trie.callback(actual_count))
    ids=output[0][actual_count:].tolist();result=trie.finish(ids)
    if tokenizer.decode(trie.prefix_ids+ids,skip_special_tokens=True)!=result['response_text']:
        raise ValueError('actual_prefilled_decoded_object_differs_from_accepted_object')
    return {**result,'raw_generated_text':tokenizer.decode(ids,skip_special_tokens=False),
        'response_sha256':hashlib.sha256(result['response_text'].encode()).hexdigest(),
        'base_input_token_count':base_count,'actual_input_token_count':actual_count,
        'base_input_tensor_sha256':digest([base_ids]),'input_tensor_sha256':digest([actual_ids]),
        'finish_reason':'eos','usage':{'prompt_tokens':actual_count,
            'completion_tokens':len(ids),'total_tokens':actual_count+len(ids)}}
