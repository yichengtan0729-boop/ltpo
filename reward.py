import re

try:
    from termcolor import colored
except ImportError:
    def colored(text, *args, **kwargs):
        return text


VERA_ANSWER_SYMBOL = "THE SCORE IS:"


class RewardModel(object):
    def __init__(
        self,
        model,
        tokenizer,
        num_thought_tokens,
        device: str = "cuda",
        rule_format_string: str = None,
        model_name: str = "",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_thought_tokens = num_thought_tokens
        self.device = device
        self.rule_format_string = rule_format_string
        self.model_name = (model_name or getattr(tokenizer, "name_or_path", "")).lower()

    def _latent_token_string(self):
        if "llama" in self.model_name:
            return "".join(f"<|reserved_special_token_{i}|>" for i in range(self.num_thought_tokens))
        elif "qwen" in self.model_name:
            return "<|endoftext|>" * self.num_thought_tokens
        elif "mistral" in self.model_name:
            return "<unk>" * self.num_thought_tokens
        else:
            # 默认走 llama 风格，至少不直接崩
            return "".join(f"<|reserved_special_token_{i}|>" for i in range(self.num_thought_tokens))

    def _find_start_idx(self, input_ids, token_ids):
        # 在 input_ids 中找 token_ids 第一次出现的位置
        seq = input_ids[0].tolist()
        n = len(token_ids)
        for i in range(len(seq) - n + 1):
            if seq[i:i + n] == token_ids:
                return i
        raise ValueError("Could not find latent thought token span in reward prompt.")

    def get_reward(self, question, specil_tokens_embeds=None, special_tokens_embeds=None):
        # 同时兼容旧拼写 specil_tokens_embeds 和正确拼写 special_tokens_embeds
        token_embeds = special_tokens_embeds if special_tokens_embeds is not None else specil_tokens_embeds
        if token_embeds is None:
            raise ValueError("Either specil_tokens_embeds or special_tokens_embeds must be provided.")

        system_instruct = (
            "You are a critical scorer tasked with scoring some special tokens for reasoning problems. "
            f"You will be provided a question and {self.num_thought_tokens} special tokens. "
            "These special tokens will be used by another language model as compressed latent reasoning hints.\n"
            "Your job is to carefully read the question and score how useful these special tokens are for helping the model reason correctly.\n"
            "The score MUST be between 0 and 1, inclusive, where 0 means useless and 1 means extremely useful.\n"
        )

        latent_thought_tokens = self._latent_token_string()

        vera_prompt = (
            f"{system_instruct}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"SPECIAL TOKENS:\n{latent_thought_tokens}\n\n"
            f"INSTRUCTIONS:\n"
            f'You MUST output the score in the following format: "{VERA_ANSWER_SYMBOL}[your_score]".\n'
            f'For example: "{VERA_ANSWER_SYMBOL}0.75"\n'
        )

        message = [{"role": "user", "content": vera_prompt}]
        inputs = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        inputs_embeds = self.model.get_input_embeddings()(inputs["input_ids"])

        token_ids = self.tokenizer.encode(
            latent_thought_tokens,
            add_special_tokens=False,
        )
        start_idx = self._find_start_idx(inputs["input_ids"], token_ids)

        # 注意：如果 tokenizer 把多个特殊 token 编成多个 id，这里仍然按长度覆盖
        end_idx = start_idx + token_embeds.shape[0]

        inputs_embeds[0, start_idx:end_idx] = token_embeds
        inputs["inputs_embeds"] = inputs_embeds
        inputs.pop("input_ids")

        answer = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
        )[0]
        response = self.tokenizer.decode(answer, skip_special_tokens=True)
        print(colored(f"Response from Reward Model:\n{response}", "light_blue"))

        return self.extract_score(response)

    def extract_score(self, response: str) -> float:
        pattern = r"(?:THE SCORE IS:?|THE SCORES ARE:?|\*\*Score:?\*\*|\*Score:?\*|Score:?)\s*(\d+(?:\.\d+)?)"
        matches = re.findall(pattern, response, re.IGNORECASE)

        if not matches:
            print(colored(
                f"WARNING in extract_score: no score found.\n{'-' * 30}\n{response}\n{'-' * 30}",
                "yellow",
            ))
            return 0.0

        score = float(matches[-1])
        score = max(0.0, min(1.0, score))
        print(colored(f"Verifier score for {self.num_thought_tokens} tokens: {score}.", "green"))
        return score
