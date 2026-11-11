"""Arm/gold-blind finite answer grammar; no model, files or corpus access."""
import hashlib
import json

VERSION = 'common-finite-json-answer-grammar-v1'
STATUSES = ('active','agreed','approved','cancelled','canceled','completed','confirmed','continued',
    'declined','deferred','expired','failed','helpful','improved','in_progress','mixed',
    'no_change','not_agreed','not_reported','not_started','ongoing','open','partially_completed',
    'paused','pending','planned','proposed','rejected','resumed','stopped','succeeded',
    'unchanged','unhelpful','unknown','withdrawn','worse','worsened')


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()


def policy():
    return {'version':VERSION,'schema':{'type':'object','required':['answer'],
        'additionalProperties':False,'answer':{'integer_minimum':0,'integer_maximum':9999,
            'string_enum':list(STATUSES),'null_allowed':True}},
        'selection_inputs':'generated token prefix only; no arm, question, gold or memory state',
        'candidate_text_form':'compact JSON with answer as the sole key',
        'eos_policy':'only after a complete accepted JSON object',
        'is_semantic_validation':False}


def candidate_texts():
    return [canonical({'answer':v}) for v in [None,*range(10000),*STATUSES]]


class AnswerTrie:
    """Only the tokenizer and generic, fixed output language enter this trie."""
    def __init__(self,tokenizer,eos_token_ids):
        self.eos = sorted(set([eos_token_ids] if type(eos_token_ids) is int else eos_token_ids))
        if not self.eos or any(type(i) is not int or i<0 for i in self.eos):
            raise ValueError('explicit_nonempty_integer_eos_ids_required')
        self.root={};self.terminals={};encoded=[]
        for text in candidate_texts():
            ids=tokenizer.encode(text,add_special_tokens=False)
            if (not ids or any(type(i) is not int or i<0 or i in self.eos for i in ids)
                or tokenizer.decode(ids,skip_special_tokens=False)!=text):
                raise ValueError('tokenizer_must_roundtrip_every_candidate_without_eos')
            key=tuple(ids)
            if key in self.terminals: raise ValueError('ambiguous_tokenized_candidates')
            self.terminals[key]=text;node=self.root
            for token in ids: node=node.setdefault(token,{})
            encoded.append([text,ids])
        self.maximum_candidate_tokens=max(len(k) for k in self.terminals)
        self.identity={'policy':policy(),'policy_sha256':digest(policy()),
            'tokenizer_chat_template_sha256':digest(tokenizer.chat_template),
            'tokenizer_class':type(tokenizer).__name__,'encoded_language_sha256':digest(encoded),
            'candidate_count':len(encoded),'eos_token_ids':self.eos,
            'maximum_candidate_tokens':self.maximum_candidate_tokens}

    def allowed(self,prefix):
        key=tuple(int(i) for i in prefix);node=self.root
        for token in key:
            if token not in node: raise ValueError('generated_prefix_outside_frozen_grammar')
            node=node[token]
        result=sorted(node)
        if key in self.terminals: result+=self.eos
        if not result: raise ValueError('nonterminal_dead_end')
        return result

    def callback(self,input_tokens):
        def allowed_tokens(batch_id,input_ids):
            if batch_id!=0: raise ValueError('single_example_generation_required')
            return self.allowed(input_ids.tolist()[input_tokens:])
        return allowed_tokens

    def finish(self,token_ids):
        if not token_ids or token_ids[-1] not in self.eos:
            raise ValueError('grammar_output_did_not_finish_with_eos')
        key=tuple(token_ids[:-1])
        if any(i in self.eos for i in key) or key not in self.terminals:
            raise ValueError('generated_tokens_not_one_complete_accepted_object')
        text=self.terminals[key]
        return {'response_text':text,'chosen_answer':json.loads(text)['answer'],
            'raw_generated_token_ids':list(token_ids),'decoding_identity_sha256':digest(self.identity)}


def generate(model,tokenizer,messages,trie,*,max_new_tokens=64):
    """Same chat template and greedy model; add only a token-allowlist callback."""
    import torch
    if max_new_tokens<=trie.maximum_candidate_tokens:
        raise ValueError('output_budget_cannot_fit_every_candidate_plus_eos')
    inputs=tokenizer.apply_chat_template(messages,add_generation_prompt=True,
        tokenize=True,return_dict=True,return_tensors='pt')
    count=int(inputs['input_ids'].shape[-1])
    if count>4096: raise ValueError('input_token_limit_no_truncation')
    inputs={k:v.to(model.device) for k,v in inputs.items()}
    with torch.inference_mode():
        output=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,use_cache=True,
            pad_token_id=tokenizer.eos_token_id,prefix_allowed_tokens_fn=trie.callback(count))
    ids=output[0][count:].tolist();result=trie.finish(ids)
    if tokenizer.decode(ids,skip_special_tokens=True)!=result['response_text']:
        raise ValueError('decoded_wire_differs_from_accepted_grammar')
    return {**result,'raw_generated_text':tokenizer.decode(ids,skip_special_tokens=False),
        'response_sha256':hashlib.sha256(result['response_text'].encode()).hexdigest(),
        'input_tensor_sha256':digest(inputs['input_ids'].tolist()),'finish_reason':'eos',
        'usage':{'prompt_tokens':count,'completion_tokens':len(ids),'total_tokens':count+len(ids)}}
