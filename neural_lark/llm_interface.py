import abc
import time
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List
import json

import openai
import google.generativeai as palm
import google.api_core.exceptions as palm_exceptions

from http import HTTPStatus
import dashscope
from dashscope import Generation
from zhipuai import ZhipuAI



import neural_lark.utils as utils
from neural_lark.flags import FLAGS
from neural_lark.train_utils import logger
from neural_lark.structs import LLMResponse



class LargeLanguageModel(abc.ABC):
    """A pretrained large language model."""

    @abc.abstractmethod
    def get_id(self) -> str:
        """Get a string identifier for this LLM.

        This identifier should include sufficient information so that
        querying the same model with the same prompt and same identifier
        should yield the same result (assuming temperature 0).
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _sample_completions(self,
                            prompt: str,
                            temperature: float,
                            stop_token: str,
                            num_completions: int = 1) -> List[LLMResponse]:
        """This is the main method that subclasses must implement.

        This helper method is called by sample_completions(), which
        caches the prompts and responses to disk.
        """
        raise NotImplementedError("Override me!")
    
    def sample_completions(self,
                           prompt: str,
                           temperature: float,
                           stop_token: str,
                           num_completions: int = 1,
                           disable_cache: bool = False) -> List[LLMResponse]:
        """Sample one or more completions from a prompt.

        Higher temperatures will increase the variance in the responses.
        The seed may not be used and the results may therefore not be
        reproducible for LLMs where we only have access through an API
        that does not expose the ability to set a random seed. Responses
        are saved to disk.
        """

        # Set up the cache file.
        os.makedirs(FLAGS.llm_cache_dir, exist_ok=True)
        llm_id = self.get_id()
        prompt_id = utils.str_to_identifier(prompt)
        # If the temperature is 0, the seed does not matter.
        escaped_stop_token = stop_token.replace("\n", "\\n")
        if temperature == 0.0:
            config_id = f"most_likely_{num_completions}_{escaped_stop_token}_{FLAGS.freq_penalty}"
        else:
            config_id = f"{temperature}_{FLAGS.seed}_{num_completions}_{escaped_stop_token}_{FLAGS.freq_penalty}"
        cache_filename = f"{llm_id}_{config_id}_{prompt_id}.pkl"
        cache_filepath = Path(FLAGS.llm_cache_dir) / cache_filename
        if not os.path.exists(cache_filepath):
            os.makedirs(os.path.dirname(cache_filepath), exist_ok=True)
        need_fresh = disable_cache or not os.path.exists(cache_filepath)
        if need_fresh:
            logger.debug(f"Querying LLM {llm_id} with new prompt.")
            completions = self._sample_completions(prompt,
                                                   temperature,
                                                   stop_token, num_completions)
            # Cache the completions.
            with open(cache_filepath, 'wb') as f:
                pickle.dump(completions, f)
            logger.debug(f"Saved LLM response to {cache_filepath}.")
            return completions

        # Load the saved completion. If cache is corrupted, re-query once.
        try:
            with open(cache_filepath, 'rb') as f:
                completions = pickle.load(f)
            logger.debug(f"Loaded LLM response from {cache_filepath}.")
            return completions
        except (EOFError, pickle.UnpicklingError):
            logger.warning(f"Cache file {cache_filepath} corrupted, regenerating.")
            completions = self._sample_completions(prompt,
                                                   temperature,
                                                   stop_token, num_completions)
            with open(cache_filepath, 'wb') as f:
                pickle.dump(completions, f)
            logger.debug(f"Overwrote corrupted cache at {cache_filepath}.")
            return completions
    
    def greedy_completion(self,
                          prompt: str,
                          stop_token: str) -> LLMResponse:
        """Sample a greedy completion from a prompt."""
        responses = self.sample_completions(prompt, 0.0, stop_token)
        assert len(responses) == 1
        return responses[0]

    #@abc.abstractmethod
    def _sample_next_token_with_logit_bias(self, prompt, logit_bias, temperature):
        """Sample the next token from the model with a logit bias."""
        raise NotImplementedError("Override me!")

    
class GPT(LargeLanguageModel):
    AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    def __init__(self, model_name: str, use_azure=True) -> None:
        self._model_name = model_name
        self.use_azure = use_azure
        if self.use_azure:
            openai.api_key = self.AZURE_API_KEY
            openai.api_base =  "https://symdistill.openai.azure.com/"
            openai.api_type = 'azure'
            openai.api_version = '2023-03-15-preview'
        else:
            openai.api_key = self.OPENAI_API_KEY

    def get_id(self) -> str:
        return f"gpt_{self._model_name}"

    def _sample_completions(
            self,
            prompt: str,
            temperature: float,
            stop_token: str,
            num_completions: int = 1) -> List[LLMResponse]:  
        response = None
        for _ in range(6):
            try:
                response = openai.Completion.create(
                    engine=self._model_name,
                    prompt=prompt,
                    temperature=temperature,
                    stop=stop_token,
                    max_tokens=FLAGS.max_tokens,
                    frequency_penalty=FLAGS.freq_penalty,
                    n=num_completions)
                # Successfully queried, so break.
                break
            except (openai.error.RateLimitError,
                    openai.error.APIConnectionError, openai.error.APIError):
                # Wait for 60 seconds if this limit is reached. Hopefully rare.
                time.sleep(6)

        if response is None:
            raise RuntimeError("Failed to query OpenAI API.")
        
        assert len(response["choices"]) == num_completions
        return [
            self._raw_to_llm_response(r, prompt, temperature,stop_token, num_completions)
            for r in response["choices"]
        ]
    
    def _sample_next_token_with_logit_bias(self, prompt, logit_bias, temperature=0.0):
        response = None
        for _ in range(6):
            try:
                response = openai.Completion.create(
                    engine=self._model_name,
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=2,
                    logit_bias=logit_bias)
                break
            except (openai.error.RateLimitError,
                    openai.error.APIConnectionError, openai.error.APIError):
                time.sleep(6)
        if response is None:
            raise RuntimeError("Failed to query OpenAI API.") 
        return response["choices"][0]["text"]

    @staticmethod
    def _raw_to_llm_response(raw_response: Dict[str, Any], 
                             prompt: str,
                             temperature: float, 
                             stop_token: str,
                             num_completions: int) -> LLMResponse:
        text = raw_response["text"]

        text = text.strip()
        text = text.replace("<|im_end|>", "")
        text = text.replace("<|im_sep|>", "")

        prompt_info = {
            "temperature": temperature,
            "num_completions": num_completions,
            "stop_token": stop_token,
        }
        return LLMResponse(prompt,
                           text,
                           prompt_info=prompt_info,
                           other_info=raw_response.copy())

    def evaluate_completion(self, prefix: str, suffix:str, average=True) -> float:
        while True:
            try:
                return self._evaluate_gpt_completion(prefix, suffix, average)
            except Exception as runtime_error:
                if "This model's maximum context length is 8001 tokens" in str(runtime_error):
                    raise runtime_error
                else:
                    time.sleep(3)
                    logger.warning(str(runtime_error))
                    logger.info("retrying...") 

    def _evaluate_completion(self, 
                                prefix: str, 
                                suffix: str, 
                                average:bool) -> float:
        _prompt = f"{prefix}{suffix}"
        response = openai.Completion.create(
            engine=self._model_name,
            prompt=_prompt,
            echo=True,
            logprobs=1,
            temperature=0,
            max_tokens=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        logprobs = response["choices"][0]["logprobs"]["token_logprobs"]
        offsets = response["choices"][0]["logprobs"]["text_offset"]
        try:
            suffix_start_token_id = offsets.index(len(prefix))
        except ValueError:
            # find the closest token
            suffix_start_token_id = min(range(len(offsets)), key=lambda i: abs(offsets[i] - len(prefix)))
            logger.warning("suffix_start_token_id not found, using closest token")

        # corner case: prefix is empty
        if suffix_start_token_id == 0: 
            assert logprobs[0] is None
            logprobs[0] = 0

        if average:
            suffix_logit = sum(logprobs[suffix_start_token_id:]) / len(logprobs[suffix_start_token_id:])
        else:
            suffix_logit = sum(logprobs[suffix_start_token_id:])

        return suffix_logit


class PaLM(LargeLanguageModel):
    PALM_API_KEY = os.environ.get("PALM_API_KEY")

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        palm.configure(api_key=self.PALM_API_KEY)
    
    def get_id(self) -> str:
        return f"palm_{self._model_name}"
    
    def _sample_completions(self, 
                            prompt: str, 
                            temperature: float, 
                            stop_token: str, 
                            num_completions: int = 1) -> List[LLMResponse]:
        
        response = None
        for _ in range(12):
            try:
                response = palm.generate_text(prompt=prompt,
                                            model=self._model_name,
                                            max_output_tokens=FLAGS.max_tokens,
                                            temperature=temperature,
                                            stop_sequences=[stop_token],
                                            candidate_count=num_completions)
            except palm_exceptions.ResourceExhausted as e:
                logger.debug("ResourceExhausted, waiting..")
                # logger.warn(f"Error {str(e)}")
                time.sleep(120)
        
        if response is None:
            raise RuntimeError("Failed to query Palm API.")
        
        if len(response.candidates) == 0:
            logger.warning("Palm returned empty response.")
            return [LLMResponse(prompt, "", prompt_info={"temperature": temperature}, other_info={})]
        
        assert len(response.candidates) == num_completions
        return [
            self._raw_to_llm_response(r, prompt, temperature,stop_token, num_completions)
            for r in response.candidates
        ]

    @staticmethod
    def _raw_to_llm_response(raw_response: Dict[str, Any], 
                             prompt: str,
                             temperature: float, 
                             stop_token: str,
                             num_completions: int) -> LLMResponse:
        text = raw_response["output"]
        text = text.strip()
        prompt_info = {
            "temperature": temperature,
            "num_completions": num_completions,
            "stop_token": stop_token,
        }
        return LLMResponse(prompt,
                           text,
                           prompt_info=prompt_info,
                           other_info=raw_response.copy())


class ChatGPT(LargeLanguageModel):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        openai.api_key = self.OPENAI_API_KEY

    def get_id(self) -> str:
        return f"chatgpt_{self._model_name}"

    def _sample_completions(
            self,
            prompt: str,
            temperature: float,
            stop_token: str,
            num_completions: int = 1) -> List[LLMResponse]:  
        """
        Note that sys and user prompt are assumed to be separated by a newline.
        """
        
        chunks = prompt.split("\n")
        sys_prompt = chunks[0]
        user_prompt = "\n".join(chunks[1:])

        response = None
        for _ in range(6):
            try:
                response = openai.ChatCompletion.create(
                    model=self._model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    stop=stop_token,
                    max_tokens=FLAGS.max_tokens,
                    frequency_penalty=FLAGS.freq_penalty,
                    n=num_completions)
                # Successfully queried, so break.
                break
            except (openai.error.RateLimitError,
                    openai.error.APIConnectionError, openai.error.APIError):
                # Wait for 60 seconds if this limit is reached. Hopefully rare.
                time.sleep(6)

        if response is None:
            raise RuntimeError("Failed to query OpenAI API.")
        
        assert len(response["choices"]) == num_completions
        return [
            self._raw_to_llm_response(r, prompt, temperature,stop_token, num_completions)
            for r in response["choices"]
        ]

    @staticmethod
    def _raw_to_llm_response(raw_response: Dict[str, Any], 
                             prompt: str,
                             temperature: float, 
                             stop_token: str,
                             num_completions: int) -> LLMResponse:
        text = raw_response["message"]["content"]
        prompt_info = {
            "temperature": temperature,
            "num_completions": num_completions,
            "stop_token": stop_token,
        }
        return LLMResponse(prompt,
                           text,
                           prompt_info=prompt_info,
                           other_info=raw_response.copy())


class Qwen(LargeLanguageModel):
    DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        dashscope.api_key = self.DASHSCOPE_API_KEY

    def get_id(self) -> str:
        return f"qwen_{self._model_name}"

    def _sample_completions(
            self,
            prompt: str,
            temperature: float,
            stop_token: str,
            num_completions: int = 1) -> List[LLMResponse]:

        # 目前先只支持生成一个 completion
        if num_completions != 1:
            logger.warning("Qwen currently only supports num_completions=1; got "
                           f"{num_completions}, will use 1 instead.")
            num_completions = 1

        response = None
        for _ in range(6):
            try:
                rsp = Generation.call(
                    model=self._model_name,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=FLAGS.max_tokens,
                    stop=stop_token if stop_token else None,
                    result_format='text',  # 返回纯文本
                )
                if rsp.status_code != HTTPStatus.OK:
                    raise RuntimeError(f"Qwen API error: {rsp.code}, {rsp.message}")
                response = rsp
                break
            except Exception as e:
                logger.warning(f"Qwen API error {e}, retrying...")
                time.sleep(6)

        if response is None:
            raise RuntimeError("Failed to query Qwen API.")

        # 按照 dashscope 文档，从 output 里取文本
        # 如果你用的是别的 result_format，请对应调整这里的取值方式
        try:
            out = response["output_text"]
        except Exception:
            try:
                out = response["output"]
            except Exception:
                out = ""
        text = str(out).strip()

        # 如果 dashscope 返回的是 JSON 字符串（包含 "text" 字段），取其中的 text 作为真正输出
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
                text = parsed["text"].strip()
        except Exception:
            pass

        prompt_info = {
            "temperature": temperature,
            "num_completions": num_completions,
            "stop_token": stop_token,
        }
        llm_resp = LLMResponse(
            prompt,
            text,
            prompt_info=prompt_info,
            other_info={"raw_output": text},
        )
        return [llm_resp]


class ZhipuChat(LargeLanguageModel):
    ZHIPU_API_KEY = os.environ.get("ZHIPUAI_API_KEY")

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._client = ZhipuAI(api_key=self.ZHIPU_API_KEY)

    def get_id(self) -> str:
        # 用在缓存文件名前缀里
        return f"zhipu_{self._model_name}"

    def _sample_completions(
            self,
            prompt: str,
            temperature: float,
            stop_token: str,
            num_completions: int = 1) -> List[LLMResponse]:

        # 智谱的 chat 接口目前一次只返回 1 条，这里直接限制一下
        if num_completions != 1:
            logger.warning("ZhipuChat only supports num_completions=1 for now; "
                           f"got {num_completions}, will use 1 instead.")
            num_completions = 1

        response = None
        for _ in range(6):
            try:
                chunks = prompt.split("\n")
                sys_prompt = chunks[0]
                user_prompt = "\n".join(chunks[1:])

                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=FLAGS.max_tokens,
                )

                break
            except Exception as e:
                logger.warning(f"Zhipu API error {e}, retrying...")
                time.sleep(6)

        if response is None:
            raise RuntimeError("Failed to query Zhipu API.")

        # 按官方 SDK 返回结构取文本，典型写法如下
        choice = response.choices[0]
        # 有的 SDK 是 choice.message.content，有的直接是 content，看你版本文档
        text = getattr(choice.message, "content", None) or choice.message["content"]
        text = text or ""
        text = text.strip()

        # 手动处理 stop_token（如果智谱接口本身不支持 stop 参数）
        if stop_token:
            idx = text.find(stop_token)
            if idx != -1:
                text = text[:idx]

        prompt_info = {
            "temperature": temperature,
            "num_completions": num_completions,
            "stop_token": stop_token,
        }
        llm_resp = LLMResponse(
            prompt,
            text,
            prompt_info=prompt_info,
            other_info=response.model_dump() if hasattr(response, "model_dump") else response.__dict__,
        )
        return [llm_resp]



def setup_llm(engine):
    split_point = engine.index("/")
    platform, engine_short = engine[:split_point], engine[split_point+1:]
    if platform == "azure":
        llm = GPT(engine_short)
    elif platform == "google":
        llm = PaLM(engine_short)
    elif platform == "openai":
        if engine_short == "code-davinci-002":
            llm = GPT(engine_short, use_azure=False)
        else:
            llm = ChatGPT(engine_short)
    elif platform == "qwen":
        llm = Qwen(engine_short)
    elif platform == "zhipu":
        llm = ZhipuChat(engine_short)
    else:
        raise NotImplementedError(f"platform {platform} not supported")
    return llm
