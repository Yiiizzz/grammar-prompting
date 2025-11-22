import os
import json
import random
import tqdm

import wandb
import functools
import collections

import numpy as np

from minEarley.parser import EarleyParser

from neural_lark.flags import FLAGS, parse_args
from neural_lark.dataset import load_sempar_data, load_sem_parser, evaluate_programs, evaluate_grammars, counter2pred, evaluate_dfa, evaluate_fol
from neural_lark.llm_interface import setup_llm
from neural_lark.retriever import retrieve_fn_dict, setup_bm25
from neural_lark.earley import predict_program_with_earley_correction, predict_rules_with_earley_correction
from neural_lark.train_utils import logger, setup_logger_file
from neural_lark.lark_utils import * 
from neural_lark.overnight_utils import remove_lf_space as remove_lf_space_overnight


def construct_rule_instruction(rules, dataset):
    if dataset == "geoquery" or dataset == "overnight":
        instruction = "First, you should write grammar rules by choosing from the following BNF rules. Then, you should write programs that conform to your predicted rules.\n"
        add_rules_flag = True
    elif dataset == "smc" or dataset == "regex":
        instruction= "First, you should write a grammar that contains all the necessary BNF rules. Then, you should write programs that conform to your predicted rules.\n"
        add_rules_flag = False
    elif dataset == "folio":
        instruction = "First, you should write a BNF grammar that covers all the necessary predicates, constants and logical rules. Then, you should write first-order logic formulas that conform to your predicted rules. Note that constants should start with lowercase; predicates should start with uppercase.\n"
        add_rules_flag = True
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if add_rules_flag:
        lark_str = rulelist2larkstr(rules)
        bnf_str = lark2bnf(lark_str)
        instruction = f"{instruction}\n[BEGIN RULES]\n{bnf_str}\n[END RULES]\n\n" 
    return instruction

DELIMITER = "\nprogram based on the BNF grammar rules:\n"

prompt_templates = {
    "std": {
        "instruction": ("You are an expert programmer, and you need to write a program" 
                        " for the given natural language query.\n"),
        "rule_instruction": None,
        "exemplar": lambda ex: f"query: {ex.source}\nprogram:\n{ex.target}\n\n",
        "prediction": lambda ex: f"query: {ex.source}\nprogram:\n",
    },
    "wrule": {
        "instruction": ("You are an expert programmer, and you need to write a program" 
                        " for the given natural language query.\n"),
        "rule_instruction": "",
        "exemplar": lambda ex: f"query: {ex.source}\nBNF grammar rules:\n{ex.grammar}{DELIMITER}{ex.target}\n\n",
        "rule_exemplar": lambda ex: f"query: {ex.source}\nBNF grammar rules:\n{ex.grammar}\n\n",
        "prediction": lambda ex: f"query: {ex.source}\nBNF grammar rules:\n",
        "prediction_given_rule": lambda ex: f"query: {ex.source}\nBNF grammar rules:\n{ex.grammar}{DELIMITER}",
    },
    # folio任务的prompt模板
    "fol_std": {
        "instruction": ("You are an expert programmer, and you need to write first-order logic formulas"
                        " for the given natural language sentences. The goal is to determine whether the final sentence can be inferred from the previous sentences.\n"),
        "rule_instruction": None,
        "exemplar": lambda ex: f"sentences:\n{ex.source}\nprogram:\n{ex.target}\n\n",
        "prediction": lambda ex: f"sentences:\n{ex.source}\nprogram:\n",
    },
    "fol_wrule": {
        "instruction": ("You are an expert programmer, and you need to write first-order logic formulas"
                        " for the given natural language sentences. The goal is to determine whether the final sentence can be inferred from the previous sentences.\n"),
        "rule_instruction": "",
        "exemplar": lambda ex: f"sentences:\n{ex.source}\nBNF grammar rules:\n{ex.grammar}{DELIMITER}{ex.target}\n\n",
        "rule_exemplar": lambda ex: f"sentences:\n{ex.source}\nBNF grammar rules:\n{ex.grammar}\n\n",
        "prediction": lambda ex: f"sentences:\n{ex.source}\nBNF grammar rules:\n",
        "prediction_given_rule": lambda ex: f"sentences:\n{ex.source}\nBNF grammar rules:\n{ex.grammar}{DELIMITER}",
    }
}

def batch_prompt_predict(
    llm, 
    input_examples, 
    train_examples, 
    prompt_template, 
    retrieve_fn,
    use_linearized_tree_flag,
    constrain_prog_gen_flag,
    overnight_flag,
    predefined_fewshot_prompt=None,
):
    """
    Args:
        overnight_flag: overnight lf needs special handling when using linearized tree
        fewshot_prompt: if None, construct the prompt from the template
    """
    if use_linearized_tree_flag:
        assert not constrain_prog_gen_flag, "linearized tree is not compatible with earley correction"

    prompts, predictions = [], []
    template_ex = prompt_template["exemplar"]
    template_p = prompt_template["prediction"]

    #对每一个要预测的样本 input_example，构造一个 few-shot prompt 字符串，后面会把这个字符串喂给 LLM 让它生成程序。
    #构造时包含三大块：

    #1.顶部的 instruction（你是专家程序员…）；
    #2.可选的 rule_instruction（如果启用了 grammar 提示，就会插入那段“先写规则再写程序 + BNF 规则表”）；
    #3.若干个 few-shot 示例（由 template_ex(exemplar) 拼出来）。
    for input_example in tqdm.tqdm(input_examples, total=len(input_examples)):

        #判断有没有“预定义好的完整 prompt 文本”（即提前写好了一个完整 prompt 保存在文件里）
        if predefined_fewshot_prompt is None:
            fewshot_prompt = prompt_template["instruction"]
            if prompt_template["rule_instruction"]:
                fewshot_prompt += prompt_template["rule_instruction"]
            exemplars = retrieve_fn(input_example, train_examples)
            for exemplar in exemplars:
                if use_linearized_tree_flag:
                    if not hasattr(exemplar, "linearized"):
                        exemplar_tree = global_parser.parse(exemplar.target)
                        exemplar.target = linearize_tree(exemplar_tree)
                        exemplar.linearized = True
                fewshot_prompt += template_ex(exemplar)
        #如果有，就直接用预定义的 prompt
        else:
            fewshot_prompt = predefined_fewshot_prompt

        #再把当前要预测的样本 input_example（即当前 query） 拼到 prompt 后面，形成最终的 prompt
        _prompt = fewshot_prompt + template_p(input_example)
        prompts.append([_prompt])

        #调用 LLM 生成程序
        ret_predictions = []
        if constrain_prog_gen_flag:
            prediction = predict_program_with_earley_correction(llm, _prompt, global_parser)
            ret_predictions.append(prediction)
        else:
            responses = llm.sample_completions(_prompt, FLAGS.temperature, stop_token="\n\n")
            assert len(responses) == 1
            prediction = responses[0].response_text

            if use_linearized_tree_flag:
                # recover the original program
                logger.debug("prediction before linearization: " + prediction)
                if overnight_flag:
                    prediction = linearized_tree_to_program(prediction, delimiter=" ")
                    prediction = remove_lf_space_overnight(prediction)
                else:
                    prediction = linearized_tree_to_program(prediction)

            ret_predictions.append(prediction)
                    
        _counter = collections.Counter(ret_predictions)
        predictions.append(_counter)
        logger.info("Summary:" + "-" * 80)
        logger.info(f"number of unique predictions from std prompt: {len(_counter)}")
        logger.info(f"frequency distribution of new predictions: {list(_counter.values())}")

        logger.info(f"    source:\n{input_example.source}")
        logger.info(f"prediction:\n{counter2pred(_counter)}")
        logger.info(f"    target:\n{input_example.target}")
        logger.info("-" * 80)
    return prompts, predictions

def batch_prompt_wrule_predict(
        llm,
        input_examples, #要预测的样本列表
        train_examples, #训练样本列表，用来从中挑 few‑shot 示例塞到 prompt 里。
        prompt_template,
        retrieve_fn,
        use_oracle_rule_flag,
        constrain_rule_gen_flag, #控制规则生成阶段是否用 Earley/grammar 约束
        constrain_prog_gen_flag, #控制程序生成阶段是否用 Earley 约束
        separate_rule_gen_flag, #是否把“写规则”和“写程序”拆成两阶段
        lazy_constrain_flag, 
        predefined_fewshot_prompt=None,
    ):
    """
    Args:
        use_oracle_rule_flag: if True, use oracle rule to generate the prompt
        constrain_rule_gen_flag: if True, constrain rule generation
        constrain_prog_gen_flag: if True, constrain program generation
        seperate_rule_gen_flag: if True, generate rule first, then program using different prompts
        lazy_constrain_flag: sample k candidates first; if no candidate is valid, then use early two-stage generation
    """
    prompts, predictions, grammars = [], [], []
    template_rule_prog_ex = prompt_template["exemplar"] #把一个训练示例转成“query + BNF 规则 + program”的文本，用在 few-shot 里。
    template_rule_ex = prompt_template["rule_exemplar"] #把一个训练示例转成“query + BNF 规则（不带 program）”的文本，用在“只学写规则”的 few-shot。
    template_starts_wrule_pred = prompt_template["prediction"] #当前样本的“开始写规则”那部分起始文本，一般是query: ...\nBNF grammar rules:\n
    template_prog_given_rule_pred = prompt_template["prediction_given_rule"] #当前样本在“规则已知的情况下，要生成程序”的起始文本
    
    #遍历每个测试样本 input_example。
    for input_example in tqdm.tqdm(input_examples, total=len(input_examples)):
        #第一步：构造“规则+程序”的 few-shot （fewshot_rule_prog_prompt）
        # 如果没有预定义的 prompt，就自己拼
        if predefined_fewshot_prompt is None:
            #用 BM25 / rand / all 从训练集里选出若干条 few-shot 示例。
            exemplars = retrieve_fn(input_example, train_examples)
            #先放一段prompt_template["instruction"]
            fewshot_rule_prog_prompt = prompt_template["instruction"]
            #如果模板里有 rule_instruction，（可能包含那全局 BNF 规则表），就拼上去。
            if prompt_template["rule_instruction"]:
                fewshot_rule_prog_prompt += prompt_template["rule_instruction"]
            #对前面选出的 exemplars 中的每个 exemplar
            for exemplar in exemplars:
                #gen_min_lark(exemplar.target, global_parser)：用全局语法解析这条示例程序，抽取“刚好需要的最小规则集合”（专用子语法）。
                #lark2bnf(...)：把这个子语法从 Lark 形式转成 BNF 文本。
                exemplar.grammar = lark2bnf(gen_min_lark(exemplar.target, global_parser))
                #把 exemplar 拼成“query + BNF 规则 + program”的形式，接到 fewshot_rule_prog_prompt 后面。
                fewshot_rule_prog_prompt += template_rule_prog_ex(exemplar)
        else:
            fewshot_rule_prog_prompt = predefined_fewshot_prompt
        
        #先设一个标志位，控制后面要不要走“Earley 两阶段生成”（先规则、再程序）。
        do_earley_two_stage_gen_flag = False
        
        #分支一：use_oracle_rule_flag（用真规则，不让 LLM 发明）即从上帝视角直接用正确program反推最正确的规则集合
        if use_oracle_rule_flag:
            bnf_grammar = lark2bnf(gen_min_lark(input_example.target, global_parser))
            input_example.grammar = bnf_grammar

            #template_prog_given_rule_pred(input_example) 会生成：
            #       query: 当前 query
            #       BNF grammar rules:
            #       <oracle BNF 规则>
            #       program based on the BNF grammar rules:

            #把它接在 few-shot 后面，得到 prompt_for_prog，即给 LLM 写程序的 prompt。
            
            prompt_for_prog = fewshot_rule_prog_prompt + template_prog_given_rule_pred(input_example)

            lark_grammar = bnf2lark(bnf_grammar)
            assert check_grammar_validity(global_rules, lark_grammar)
            #将这套 oracle grammar 作为该样本的“预测 grammar”。
            ret_grammars = [lark_grammar]

            #如果开了约束程序生成（--constrain_prog_gen_flag），就用 Earley correction 生成程序；
            if constrain_prog_gen_flag:
                prediction = predict_program_with_earley_correction(llm, prompt_for_prog, global_parser)
                ret_predictions = [prediction]
            #否则就让 LLM 自己随便生成程序字符串。
            else:
                response = llm.sample_completions(prompt_for_prog, FLAGS.temperature, stop_token="\n\n")[0]
                ret_predictions = [response.response_text] 
        
        #分支二：lazy_constrain_flag
        # （先让 LLM 自由发挥直接一起生成若干候选“规则+程序”，然后筛选；如果都不合格再走 Earley 两阶段生成）
        elif lazy_constrain_flag:
            assert not separate_rule_gen_flag
            #template_starts_wrule_pred(input_example) 如：
            #       query: 当前 query
            #       BNF grammar rules:
            prompt_for_rule_prog = fewshot_rule_prog_prompt + template_starts_wrule_pred(input_example)
            #让 LLM 直接生成若干“规则+程序”候选
            responses = llm.sample_completions(prompt_for_rule_prog, FLAGS.temperature, stop_token="\n\n")
            raw_predictions = [r.response_text for r in responses] 
            
            ret_predictions, ret_grammars = [], []
            #筛选这些候选，看看有没有符合 grammar 约束的
            for raw_pred in raw_predictions:
                #把“规则”和“程序”拆开，中间用 DELIMITER（那行 program based on ...）分隔。
                try:
                    pred_bnf_grammar, pred_program = raw_pred.split(DELIMITER)
                    #把 grammar 文本转回 Lark 形式
                    pred_lark_grammar = bnf2lark(pred_bnf_grammar)
                    #如果开了 constrain_rule_gen_flag，就检测这 grammar 是否与全局规则兼容，失败就丢掉；
                    if constrain_rule_gen_flag and not check_grammar_validity(global_rules, pred_lark_grammar):
                        continue
                    #如果开了 constrain_prog_gen_flag，就用 Earley parser 检测程序是否合法，失败就丢掉；
                    if constrain_prog_gen_flag:
                        local_parser = EarleyParser(decorate_grammar(pred_lark_grammar), start=global_parser.option.start)
                        local_parser.parse(pred_program)
                    #能通过检查的，就加入 ret_grammars、ret_predictions
                    ret_grammars.append(pred_lark_grammar)
                    ret_predictions.append(pred_program)
                except Exception as e:
                    logger.warning(f"failed to find prediction from {raw_pred} due to {e}")
            #如果一个合法的候选都没有，就把标志位置为 True，后面走 Earley 两阶段生成。
            if len(ret_predictions) == 0:
                #打日志说“要调用 Earley 纠错了”
                logger.info("invoking earley correction")
                do_earley_two_stage_gen_flag = True
        #分支三：默认走两阶段 Earley（非 oracle 且非 lazy）
        else:
            #如果要分开生成规则和程序，就不能走这个分支
            assert not separate_rule_gen_flag
            #直接设定走 Earley 两阶段生成。
            do_earley_two_stage_gen_flag = True
        
        #所有走两阶段 Earley 的情况都在这里处理:先生成语法，再在这个语法下生成程序。
        if do_earley_two_stage_gen_flag:
            #构造“只生成规则”的 prompt
            if separate_rule_gen_flag:
                #拼“few-shot + 当前 query 的开头部分”
                #顶部放一段任务说明。
                fewshot_rule_prompt = prompt_template["instruction"]
                # 如果启用了全局规则说明，就拼上去。
                if prompt_template["rule_instruction"]:
                    fewshot_rule_prompt += prompt_template["rule_instruction"]
                #对每个 few‑shot 示例
                for exemplar in exemplars:
                    #从全局 grammar 里抽这一条示例所需的最小规则集合；转成 BNF 文本，放进 exemplar.grammar；
                    exemplar.grammar = lark2bnf(gen_min_lark(exemplar.target, global_parser))
                    #template_rule_ex(exemplar) 会生成的示例结构为：
                    #      示例1：
                    #       query: 示例的自然语言
                    #       BNF grammar rules:
                    #       示例专用 BNF 规则
                    #      示例2：
                    #       ......
                    #即这批 few‑shot 只在教模型“怎么根据 query 写语法规则”。
                    fewshot_rule_prompt += template_rule_ex(exemplar)
                #再加上“当前样本”的“开始写规则”部分，比如：
                #       query: 当前 query
                #       BNF grammar rules:
                #最后的prompt_for_rule 就是“用来生成规则的 prompt”。
                prompt_for_rule = fewshot_rule_prompt + template_starts_wrule_pred(input_example)
            else:
                #否则就用之前构造好的“规则+程序” few-shot prompt，直接接上当前样本的“开始写规则”部分。
                prompt_for_rule = fewshot_rule_prog_prompt + template_starts_wrule_pred(input_example)

            #生成规则（pred_bnf_grammar）并（可选）Earley 修正
            try:
                if constrain_rule_gen_flag:
                    pred_bnf_grammar = predict_rules_with_earley_correction(llm, prompt_for_rule, global_rules, DELIMITER)
                else:
                    response = llm.sample_completions(prompt_for_rule, FLAGS.temperature, stop_token=DELIMITER)[0]
                    pred_bnf_grammar = response.response_text 
                pred_lark_grammar = bnf2lark(pred_bnf_grammar)
                #input_example.grammar = pred_bnf_grammar：把这段预测规则挂到当前样本对象上。
                #template_prog_given_rule_pred(input_example) 会生成类似：
                #           query: 当前 query
                #           BNF grammar rules:
                #           <pred_bnf_grammar>
                #           program based on the BNF grammar rules:
                #
                #把它接在原来的 few-shot 规则+程序前缀后面，得到 prompt_for_prog：这是第二阶段生成程序要用的 prompt。
                input_example.grammar = pred_bnf_grammar
                prompt_for_prog = fewshot_rule_prog_prompt + template_prog_given_rule_pred(input_example)

                #在预测语法下生成程序（pred_program），并（可选）Earley 修正
                if constrain_prog_gen_flag:
                    try:
                        logger.info(f"earley correction with grammar\n{pred_lark_grammar}")
                        local_parser = EarleyParser(decorate_grammar(pred_lark_grammar), start=global_parser.option.start)
                    except Exception as e:
                        logger.warning(f"failed to create parser due to {e}, reverting to global parser")
                        local_parser = global_parser
                    pred_program = predict_program_with_earley_correction(llm, prompt_for_prog, local_parser)
                else:
                    resposne = llm.sample_completions(prompt_for_prog, FLAGS.temperature, stop_token="\n\n")[0]
                    pred_program = resposne.response_text 
            #如果上面整个“生成规则 + 生成程序”流程在任意一步抛异常（比如规则不合法，parser 构造失败等），就进入这个 except。
            except Exception as e:
                logger.warning(f"failed to find prediction due to {e}")
                prompt_for_rule_prog = fewshot_rule_prog_prompt + template_starts_wrule_pred(input_example) 
                response = llm.sample_completions(prompt_for_rule_prog, FLAGS.temperature, stop_token="\n\n")[0]
                try:
                    pred_bnf_grammar, pred_program = response.split(DELIMITER)
                    pred_lark_grammar = bnf2lark(pred_bnf_grammar)
                except:
                    logger.warning(f"failed to find prediction from {response.response_text} due to {e}")
                    pred_lark_grammar, pred_program = None, None
            
            #把结果打包返回上层
            ret_grammars = [pred_lark_grammar]
            ret_predictions = [pred_program]

        
        
        #**“对当前这个样本收尾：记录用过的 prompt、整理预测结果、打日志，然后循环下一个样本”**

        #记录当前样本用过的所有 prompt
        used_prompts = []
        if "prompt_for_prog" in locals():
            used_prompts.append(prompt_for_prog)
        if "prompt_for_rule_prog" in locals():
            used_prompts.append(prompt_for_rule_prog)
        if "prompt_for_rule" in locals():
            used_prompts.append(prompt_for_rule)
        prompts.append(used_prompts)
        #把预测结果整理成 Counter
        _pred_counter = collections.Counter(ret_predictions)
        predictions.append(_pred_counter)
        _grammar_counter = collections.Counter(ret_grammars)
        grammars.append(_grammar_counter)
        #打日志
        logger.info("Summary:" + "-" * 80)#开头的行，在日志里分隔开。
        logger.info(f"number of unique predictions: {len(_pred_counter)}")#这条样本有多少种不同的程序预测（一般情况下 1 种）。
        logger.info(f"frequency distribution of predictions: {list(_pred_counter.values())}")#每种预测出现了多少次（如果做多采样就有意义）。

        logger.info(f"    source:\n{input_example.source}")#这条样本的自然语言输入。
        logger.info(f"prediction:\n{counter2pred(_pred_counter)}")#这条样本的最终预测程序。
        logger.info(f"    target:\n{input_example.target}")#gold program（真实标注程序）。
        logger.info(f"   grammar:\n{counter2pred(_grammar_counter)}")#取“出现次数最多的 grammar”作为这条样本的最终 grammar 预测。
        logger.info("-" * 80)#分界线
    return prompts, predictions, grammars


#主程序控制流程

if __name__ == "__main__":
    # 项目名，组名传给 wandb，用来把这次 run 归类
    project_name = "Rule-ICL"
    group_name = "sempar-rule-icl"

    # 1. setup 

    #1.1 old wandb and logger
    # parse_args()
    # random.seed(FLAGS.seed)
    # config = vars(FLAGS)
    #
    # exp_name = "-".join([f"{k[:3]}_{v}" for k, v in sorted(config.items()) if k not in ["eval_only"]])
    # wandb.init(project=project_name, group=group_name, name=exp_name, config=config)
    # log_dir = f"log/{group_name}/{exp_name}"
    # setup_logger_file(logger, log_dir)
    # wandb.run.log_code("./neural_lark")

    ##1.1 wandb and logger
    parse_args()
    random.seed(FLAGS.seed)
    config = vars(FLAGS)
        # ===== New: structured log dir =====
    def _sanitize(name: str) -> str:
        """Make a string safe for Windows paths."""
        import re
        name = re.sub(r'[<>:"/\\|?*]+', '-', str(name))
        return name.replace(' ', '_').strip(' .')

    cfg = config  # 原始参数字典

    # 顶层文件夹按模型
    engine_folder = _sanitize(cfg.get("engine", "unknown_engine"))
    # 第二层按数据集
    dataset_folder = _sanitize(cfg.get("dataset", "unknown_dataset"))

    # 短标签：shots + 模式 + 模板 + 约束
    parts = []
    shots = cfg.get("num_shot", None)
    if shots is not None:
        parts.append(f"{shots}shot")
    parts.append(_sanitize(cfg.get("prompt_mode", "std")))
    parts.append(_sanitize(cfg.get("prompt_template", "std")))
    if cfg.get("add_rule_instruction_flag", False):
        parts.append("instru")
    if cfg.get("constrain_rule_gen_flag", False):
        parts.append("gcon")
    if cfg.get("constrain_prog_gen_flag", False):
        parts.append("pcon")
    if cfg.get("lazy_constrain_flag", False):
        parts.append("lazy")
    if cfg.get("use_oracle_rule_flag", False):
        parts.append("oracle")
    parts.append(f"s{cfg.get('seed', 0)}")

    short_tag = "-".join(parts)

    # 最终日志目录
    log_dir = os.path.join("log", engine_folder, dataset_folder, short_tag)

    # 用短标签作为实验名
    exp_name = short_tag
    wandb.init(project=project_name, group=group_name, name=exp_name, config=config)

    os.makedirs(log_dir, exist_ok=True)
    setup_logger_file(logger, log_dir)
    wandb.run.log_code("./neural_lark")


    ##1.2 setup grammar and parser
    global_parser, global_rules = load_sem_parser(config)

    ## 1.3 初始化 llm
    llm = setup_llm(FLAGS.engine)

    # 2. 加载数据
    train_examples, dev_examples, test_examples = load_sempar_data(config)
    logger.info(f"loaded {len(train_examples)} indist examples, {len(dev_examples)} dev examples, {len(test_examples)} test examples")

    #3. 构造 few‑shot 检索函数和 prompt 模板
    retrieve_fn = retrieve_fn_dict[FLAGS.retrieve_fn]
    if config["retrieve_fn"] == "bm25":
        bm25 = setup_bm25(train_examples)
        retrieve_fn = functools.partial(retrieve_fn, batch_size=FLAGS.batch_size, bm25=bm25)
    else:
        retrieve_fn = functools.partial(retrieve_fn, batch_size=FLAGS.batch_size)

    prompt_template = prompt_templates[FLAGS.prompt_template]
    if FLAGS.add_rule_instruction_flag:
        new_instruction = construct_rule_instruction(global_rules, FLAGS.dataset)
        prompt_template["rule_instruction"] = new_instruction
    
    # 4. few-shot prompting
    assert FLAGS.prompt_mode in ["std", "rot"]
    logger.info("few-shot prompting on the test set")
    if not config["eval_only"]:
        if FLAGS.prompt_from_file:
            with open(FLAGS.prompt_from_file) as f:
                predefined_fewshot_prompt = f.read()
        else:
            predefined_fewshot_prompt = None

        if  config["prompt_mode"] == "std":
            test_prompts, test_prediction_counters = batch_prompt_predict(llm, test_examples, train_examples, prompt_template, retrieve_fn, use_linearized_tree_flag=FLAGS.use_linearized_tree, constrain_prog_gen_flag=FLAGS.constrain_prog_gen_flag, overnight_flag=FLAGS.dataset=="overnight", predefined_fewshot_prompt=predefined_fewshot_prompt)
            test_grammar_counters = None
        else:
            test_prompts, test_prediction_counters, test_grammar_counters = batch_prompt_wrule_predict(llm, test_examples, train_examples, prompt_template, retrieve_fn, use_oracle_rule_flag=FLAGS.use_oracle_rule_flag, separate_rule_gen_flag=FLAGS.separate_rule_gen_flag, constrain_rule_gen_flag=FLAGS.constrain_rule_gen_flag, constrain_prog_gen_flag=FLAGS.constrain_prog_gen_flag, lazy_constrain_flag=FLAGS.lazy_constrain_flag, predefined_fewshot_prompt=predefined_fewshot_prompt)

        ##  dump to json and wandb
        json_results = {
            "test_prompts": test_prompts,
            "test_predictions": test_prediction_counters,
            "test_grammars": test_grammar_counters,
        }

        with open(f"{log_dir}/results.json", "w") as f:
            logger.info(f"dumping results to {log_dir}/results.json")
            json.dump(json_results, f, indent=2)

    else:
        # load from json
        with open(f"{log_dir}/results.json", "r") as f:
            json_results = json.load(f)
        test_prediction_counters = [collections.Counter(d) for d in json_results["test_predictions"]]
        test_grammar_counters = [collections.Counter(d) for d in json_results["test_grammars"]]

    ## 4.1 evaluation
    if config["dataset"] == "regex":
        test_accuracy = evaluate_dfa(test_prediction_counters, test_examples)
        test_grammar_accuracy = 0.0
    elif config["dataset"] == "folio":
        test_accuracy = evaluate_fol(test_prediction_counters, test_examples, global_parser)
        test_grammar_accuracy = 0.0
    else:
        test_accuracy = evaluate_programs(test_prediction_counters, test_examples)
        # test_grammar_accuracy = evaluate_grammars(test_grammar_counters, test_examples, global_parser)
        test_grammar_accuracy = 0.0
        logger.info(f"test accuracy {test_accuracy}")

    ## log to wandb
    wandb.log({
        "test_accuracy": test_accuracy, 
        "test_grammar_accuracy": test_grammar_accuracy
    }, step=1)