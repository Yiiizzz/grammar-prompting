import itertools
import collections

import re
import lark
from lark.load_grammar import _TERMINAL_NAMES, load_grammar
from minEarley.tree import Tree

from dataclasses import dataclass
from neural_lark.train_utils import logger

from stanza.models.constituency.parse_tree import Tree as StanzaTree

"""
For convenince, we use SimpleRule instead of lark.grammar.Rule for 1) putting rules 
in the instruction, 2) check if model-generated rules are valid. 
In the future, we may want to directly use lark.grammar.Rule, e.g., let the model
generate rules in EBNF or BNF format.
"""

# these nonterminals will be inlined when constructing rules
inline_terminal_names = {
        # for SMC dataset
        "WORD", "NUMBER", "ESCAPED_STRING", "L", 
        # for regex dataset
        "STRING", "INT", "CHARACTER_CLASS", "CONST",
        # for overnight
        # "PROPERTY", "SINGLETON_VALUE", "ENTITY_VALUE", "NUMBER_VALUE",
        # for molecule
        "N", "C", "O", "F", "c",
        # for fol
        "PNAME", "CNAME", "LCASE_LETTER"
}
for k, v in _TERMINAL_NAMES.items():
    inline_terminal_names.add(v)

## these are the nonterminals that are not needed to be predicted from model, will be used to to check the validity of the generated rules
skipped_nonterminal_names = (
    # for smc and regex
    "string", "number", "literal", "delimiter",
    # "VALUE"  # for mtop
    # "property", "value",  # for overnight
)

"""
Some concepts:
    - larkstr: a string in Lark format 
    - bnfstr: a string in BNF format (use ::= instead of :)
"""


# poor man's rule
@dataclass
class SimpleRule:
    origin: str
    expansion: tuple

    def __hash__(self):
        return hash(str(self))
    
    def __str__(self):
        return self.to_lark()
    
    def to_lark(self):
        return f"{self.origin} : {' '.join(self.expansion)}"
    
    def to_bnf(self):
        return f"{self.origin} ::= {' '.join(self.expansion)}"
    
    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, SimpleRule):
            return False
        return str(self) == str(__o)

def _wrap_string(s):
    if s.startswith("\"") and s.endswith("\""):
        # a bit complex to preserve the quotation marks
        s = f"\"\\{s[:-1]}\\\"{s[-1]}"
    else:
        s = f"\"{s}\""    

    # escape unicode characters 
    if "\\u" in s:
        s = s.replace("\\u", "\\\\u")
    
    return s

def split_rule(rule):
    split_idx = rule.index(":")
    lhs, rhs = rule[:split_idx].strip(), rule[split_idx+1:].strip()
    return lhs, rhs

def treenode2rule(treenode):
    if treenode is None:
        return None

    if isinstance(treenode, Tree):
        origin = f"{treenode.data.value}"
        expansion = []

        for child in treenode.children:
            if child is None:
                continue

            if isinstance(child, Tree):
                expansion.append(child.data.value)
            else:
                if child.type.startswith("__") or child.type in inline_terminal_names:
                    expansion.append(_wrap_string(child.value))
                else:
                    expansion.append(child.type)
    else: # terminal
        if treenode.type.startswith("__") or treenode.type in inline_terminal_names:
            return None
        else:
            origin = treenode.type
            expansion = [_wrap_string(treenode.value)]
    return SimpleRule(origin, tuple(expansion))
    
def extract_rule_stat(tree, rule_stat):
    """
    Count the occurrence of each rule
    """
    cur_rule = treenode2rule(tree)
    if cur_rule is None:
        return
    if cur_rule not in rule_stat:
        rule_stat[cur_rule] = 1
    else:
        rule_stat[cur_rule] += 1

    if getattr(tree, "children", None):
        for child in tree.children:
            extract_rule_stat(child, rule_stat)

def tree2rulelist(tree):
    rule_list = []
    def recur_add(node, rule_list):
        cur_rule = treenode2rule(node)
        if cur_rule is None:
            return
        rule_list.append(cur_rule)

        if getattr(node, "children", None):
            for child in node.children:
                recur_add(child, rule_list)
    recur_add(tree, rule_list)
    return rule_list

def extract_nonterminal_paths(tree):
    """
    从 parse tree 中抽出所有 root->leaf 的非终结符路径。
    每条路径是一个非终结符名字列表，例如 ['query', 'answer_type', 'state', ...]
    """
    paths = []

    def dfs(node, path):
        if isinstance(node, Tree):
            new_path = path + [node.data.value]
            if not getattr(node, "children", None):
                paths.append(new_path)
            else:
                for child in node.children:
                    dfs(child, new_path)

    dfs(tree, [])
    return paths

def linearize_tree(tree):
    def recur_add(node):
        if getattr(node, "children", None) is None:
            return "{" + f"{node.value}" + "}"
        else:
            ret_str = f"[{node.data.value} "
            for child in node.children:
                ret_str += recur_add(child)
                ret_str += " "
            ret_str += "]"
            return ret_str
    return recur_add(tree)

def linearized_tree_to_program(linearized_tree, delimiter=""):
    tokens = re.findall(r'{(.*?)}', linearized_tree)
    return delimiter.join(tokens)

def normalize_program(program, parser):
    tree = parser.parse(program)
    linearized_tree = linearize_tree(tree)
    return linearized_tree_to_program(linearized_tree)

def rulelist2larkstr(rule_stat):
    lhs2rhs = collections.OrderedDict()
    for rule in rule_stat:
        lhs, rhs = rule.origin, rule.expansion
        if lhs not in lhs2rhs:
            lhs2rhs[lhs] = []
        lhs2rhs[lhs].append(rhs)
    
    grammar = ""
    for lhs in lhs2rhs:
        grammar += f"{lhs} :"
        for rhs in lhs2rhs[lhs]:
            rhs_str = " ".join(rhs)
            grammar += f" {rhs_str} |"
        grammar = grammar[:-2]
        grammar += "\n"
    
    return grammar.strip()

def rulelist2bnfstr(rule_list):
    """
    Convert list of rules to lark grammar string
    """
    larkstr = rulelist2larkstr(rule_list)
    bnf_str = lark2bnf(larkstr)
    return bnf_str

def extract_min_grammar_from_trees(trees, return_rules=False):
    """
    Extract minimal grammar to reconstruct the tree
    """
    rule_stat = collections.OrderedDict()
    for tree in trees:
        extract_rule_stat(tree, rule_stat)
    grammar = rulelist2larkstr(rule_stat)

    if return_rules:
        return grammar, list(rule_stat.keys())
    else:
        return grammar

def lark2bnf(grammar):
    """
    Make it easier for GPT to generate
    """
    #grammar = grammar.replace(" : ", " -> ")
    grammar = grammar.replace(" : ", " ::= ")
    return grammar

def bnf2lark(grammar):
    """
    Opposite of lark2bnf 
    """
    # grammar = grammar.replace(" -> ", " : ")
    grammar = grammar.replace(" ::= ", " : ")
    return grammar

def decorate_grammar(grammar):
    """
    Add auxiliary rules to the grammar
    """
    grammar += "\n%import common.DIGIT"
    grammar += "\n%import common.LCASE_LETTER"
    grammar += "\n%import common.UCASE_LETTER"
    grammar += "\n%import common.WS"
    grammar += "\n%ignore WS"
    return grammar

def collect_rules_from_examples(programs, parser):
    """
    Parse programs to extract rules and collect them. Mostly for debugging
    """
    rule_stat = collections.OrderedDict()
    for program in programs:
        tree = parser.parse(program)
        extract_rule_stat(tree, rule_stat)
    
    rulestr_set = set()
    for rule in rule_stat:
        rulestr = str(rule).strip()
        rulestr_set.add(rulestr)
    return rulestr_set

def collect_rules_from_larkfile(lark_file):
    """
    Parse bnf file (.lark) to extract rules
    """
    rule_stat = collections.OrderedDict() # used as ordered set
    aux_rules = []

    with open(lark_file, "r") as f:
        cur_nonterminal = None
        for line in f:
            line = line.strip()
            if line.startswith("%"):
                aux_rules.append(line)
            elif line == "" or line.startswith("//"):
                continue
            elif line.startswith("|"):
                rhs = line[1:].strip()
                for rhs_part in rhs.split("|"):
                    rhs_part = rhs_part.strip()
                    if rhs_part == "":
                        continue
                    assert cur_nonterminal is not None
                    rule = SimpleRule(cur_nonterminal, tuple(rhs_part.split()))
                    rule_stat[rule] = 1
            elif ":" in line and "\":" not in line: # for rules like :duration
                lhs, rhs = split_rule(line)
                cur_nonterminal = lhs
                for rhs_part in rhs.split("|"):
                    rhs_part = rhs_part.strip()
                    if rhs_part == "":
                        continue
                    rule = SimpleRule(cur_nonterminal, tuple(rhs_part.split()))
                    rule_stat[rule] = 1
            else:
                raise ValueError(f"Unknown line: {line}")
    rule_set = list(rule_stat.keys())
    return rule_set, aux_rules

def build_grammar_index(global_rules):
    """
    从全局 SimpleRule 列表构建索引：
      - symbol_to_rules: 终结符 token -> 包含该 token 的规则集合
      - lhs_to_rules: 非终结符 -> 以该非终结符为左侧的规则集合
      - known_symbols: 所有出现过的终结符 token 列表
    """
    symbol_to_rules = collections.defaultdict(set)
    lhs_to_rules = collections.defaultdict(set)
    child_to_parent_rules = collections.defaultdict(set)
    known_symbols = set()

    for rule in global_rules:
        lhs = rule.origin
        lhs_to_rules[lhs].add(rule)
        for sym in rule.expansion:
            if isinstance(sym, str) and sym.startswith("\"") and sym.endswith("\""):
                raw = sym.strip("\"")
                # 简单版：直接对整个 raw 用 regex 抓 token，不区分括号前后
                for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", raw):
                    symbol_to_rules[token].add(rule)
                    known_symbols.add(token)
            else:
                child_to_parent_rules[sym].add(rule)


    return {
        "symbol_to_rules": symbol_to_rules,
        "lhs_to_rules": lhs_to_rules,
        "child_to_parent_rules": child_to_parent_rules,
        "known_symbols": sorted(known_symbols),
    }

def extract_symbols_from_program(program: str):
    """
    从草稿代码字符串里提取候选 symbol 名字（函数名、常量等）。
    简单做法：用正则抓所有字母开头的 token。
    """
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", program)
    # 去重并排序，防止无序集合带来不稳定性
    return list(sorted(set(tokens)))

def extract_symbols_from_parsed_program(program: str, parser, grammar_index=None):
    """
    更强版 symbol 抽取：
      - 先用 parser 解析草稿程序
      - 再从 parse tree 中抽出终结符里的函数名（括号前的部分）
      - 如果给了 grammar_index，只保留在 known_symbols 里的 token
    """
    try:
        tree = parser.parse(program)
    except Exception as e:
        logger.warning(f"failed to parse draft program for symbol extraction: {program} due to {e}")
        # 解析失败时退回到简单版
        return extract_symbols_from_program(program)

    # 简单做法：重用 build_grammar_index 的逻辑，遍历 tree2rulelist 得到的 SimpleRule
    used_rules = set(tree2rulelist(tree))
    candidates = set()

    for rule in used_rules:
        for sym in rule.expansion:
            if isinstance(sym, str) and sym.startswith("\"") and sym.endswith("\""):
                raw = sym.strip("\"")
                func_name = None
                if "(" in raw:
                    prefix = raw.split("(", 1)[0]
                    if re.match(r"[A-Za-z_][A-Za-z0-9_]*$", prefix):
                        func_name = prefix
                else:
                    # 不含括号的终结符，退回到 regex 抓 token
                    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", raw)
                    if toks:
                        # 通常只有一个，比如 answer
                        func_name = toks[0]

                if func_name is not None:
                    candidates.add(func_name)

    # 如果有 grammar_index，就只保留合法 DSL 函数名
    if grammar_index is not None:
        known = set(grammar_index.get("known_symbols", []))
        candidates = candidates.intersection(known)

    return list(sorted(candidates))


def induce_grammar_from_symbols(symbols, global_rules, grammar_index, start_lhs: str = None):
    """
    根据 symbol 集合，在全局 SimpleRule 里找相关规则，并做闭包补全，返回 lark 子语法字符串。
    简化实现：
      - 根据 symbol_to_rules 选出所有包含这些 symbol 的规则
      - 再根据 lhs_to_rules 对这些规则的 lhs 做闭包
    """
    symbol_to_rules = grammar_index["symbol_to_rules"]
    lhs_to_rules = grammar_index["lhs_to_rules"]

    rule_set = set()
    # 1. 由 symbol 触发的初始规则集合
    for s in symbols:
        if s in symbol_to_rules:
            rule_set.update(symbol_to_rules[s])

    if not rule_set:
        return None  # 完全没匹配上

    # 2. 闭包补全：对当前已有规则的 lhs，把所有定义该 lhs 的规则都拉进来
    changed = True
    while changed:
        changed = False
        current_rules = list(rule_set)
        for rule in current_rules:
            lhs = rule.origin
            for r2 in lhs_to_rules.get(lhs, []):
                if r2 not in rule_set:
                    rule_set.add(r2)
                    changed = True

    # 可以根据 start_lhs 再做一遍从 start 可达性的裁剪（先不做也可以）
    lark_str = rulelist2larkstr(rule_set)
    return lark_str

def specialize_nonterminals_for_draft(draft_program,
                                      global_parser,
                                      to_specialize_nts=None):
    """
    完整版专门化：
      - 用 global_parser 解析 draft_program
      - 从 parse tree 中取出草稿实际用到的 SimpleRule（tree2rulelist）
      - 找出草稿中实际出现过的、且在 to_specialize_nts 中的非终结符
      - 为这些非终结符生成专门化名字 nt__draft
      - 对 used_rules 中所有规则：
          * 如果 lhs 在这些 nt 中，则把 lhs 换成专门化名
          * 如果 RHS 中出现这些 nt，也换成专门化名
    返回：一组新的 SimpleRule（专门化后的规则集合），或 None（解析失败）
    """
    if to_specialize_nts is None:
        # 针对 geoquery，先专门化这些非终结符
        to_specialize_nts = {"state", "city", "river", "place", "num"}

    try:
        tree = global_parser.parse(draft_program)
    except Exception as e:
        logger.warning(f"failed to parse draft program for specialization: {draft_program} due to {e}")
        return None, None

    used_rules = set(tree2rulelist(tree))
    nt_paths = extract_nonterminal_paths(tree)
    used_nts = set()
    for p in nt_paths:
        used_nts.update(p)

    nts_to_specialize_here = used_nts.intersection(to_specialize_nts)
    if not nts_to_specialize_here:
        # 没有需要专门化的非终结符，就返回原规则和空的改名表
        return used_rules, {}

    nt_rename = {}
    for nt in nts_to_specialize_here:
        nt_rename[nt] = f"{nt}__draft"

    specialized_rules = set()
    for rule in used_rules:
        lhs = rule.origin
        rhs = list(rule.expansion)

        if lhs in nt_rename:
            lhs = nt_rename[lhs]

        new_rhs = []
        for sym in rhs:
            if sym in nt_rename:
                new_rhs.append(nt_rename[sym])
            else:
                new_rhs.append(sym)

        specialized_rules.add(SimpleRule(lhs, tuple(new_rhs)))

    return specialized_rules, nt_rename

def induce_minimal_intent_grammar_from_draft(draft_program,
                                             global_rules,
                                             grammar_index,
                                             parser=None,
                                             symbol_mapper=None,
                                             nt_rename=None,
                                             start_lhs: str = None,
                                             use_closure: bool = False):


    """
    Minimal Intent 版本的静态语法归纳：
      - 从 draft_program 抽取 symbol 集合
      - 幻觉 symbol 用 SymbolMapper 映射到最相近的 DSL symbol
      - 只保留包含这些 symbol 的分支（SimpleRule）
      - 按非终结符依赖向上回溯父规则，只保留能通向这些分支的路径
    返回：lark 子语法字符串；若完全匹配不上，则返回 None
    """
    # 1. 从草稿中提取 symbol 集合
    if parser is not None:
        # 有 parser 时，用 parse tree 抽 symbol（更精确）
        symbols = extract_symbols_from_parsed_program(
            draft_program, parser, grammar_index=grammar_index
        )
    else:
        # 没有 parser（比如在专门化世界），用正则简单抽取，再与 known_symbols 交集
        symbols = extract_symbols_from_program(draft_program)
        known = set(grammar_index.get("known_symbols", []))
        symbols = [s for s in symbols if s in known]


    symbol_to_rules = grammar_index["symbol_to_rules"]
    child_to_parent = grammar_index["child_to_parent_rules"]

    # 1.1 幻觉修正：把不在 DSL 里的 symbol 映射到最近合法 symbol
    if symbol_mapper is not None:
        from neural_lark.code_retriever import refine_symbols_with_mapper
        symbols = refine_symbols_with_mapper(symbols, grammar_index, symbol_mapper)

    # 1.2 如果给了 nt_rename（比如 state -> state__draft），在符号层面也做同样改名
    if nt_rename:
        renamed = []
        for s in symbols:
            renamed.append(nt_rename.get(s, s))
        symbols = renamed


    # 2. 初始规则集合：所有包含这些 symbol 的 SimpleRule（已经是“单分支”级别）
    rule_set = set()
    for s in symbols:
        if s in symbol_to_rules:
            rule_set.update(symbol_to_rules[s])

    if not rule_set:
        return None

    # 3. 向上回溯（闭包，按开关控制）
    if use_closure:
        frontier = set(rule.origin for rule in rule_set)
        visited_rules = set(rule_set)

        changed = True
        while changed:
            changed = False
            new_frontier = set()
            for child_nt in frontier:
                for parent_rule in child_to_parent.get(child_nt, []):
                    if parent_rule not in visited_rules:
                        visited_rules.add(parent_rule)
                        rule_set.add(parent_rule)
                        new_frontier.add(parent_rule.origin)
                        changed = True
            frontier = new_frontier


    # 4. （可选）如果给了 start_lhs，可以再做一遍从 start_lhs 可达性的裁剪
    # 这里先简单返回 rule_set 对应的 lark 语法
    lark_str = rulelist2larkstr(rule_set)
    return lark_str

def collect_rules_from_parser(parser, debug_rules=None):
    """
    Collect rules directly from parser. Note in some cases we 
    need to add " " to the terminal rules

    DEPRECATED unless updated
    TODO: currently I expand all terminals which is not good
    """
    def repattern2list(pattern):
        if pattern.type == "str":
            return [pattern.raw]
        else:
            re_stmt = pattern.value
            # unescape regex
            re_stmt = re_stmt.replace("\\", "")
            assert re_stmt[:3] == "(?:" and re_stmt[-1] == ")"
            elements = re_stmt[3:-1].split("|")
            return [f"\"{e}\"" for e in elements]
    
    rule_defs = parser.rules
    rule_set = set()
    for rule_def in rule_defs:
        origin = rule_def.origin.name.value

        catersian_product = []
        for nt_t in rule_def.expansion:
            if isinstance(nt_t, lark.grammar.Terminal):
                term_def = parser.get_terminal(nt_t.name)
                pattern = term_def.pattern
                candidates = repattern2list(pattern)
                catersian_product.append(candidates)
            elif isinstance(nt_t, lark.grammar.NonTerminal):
                catersian_product.append([nt_t.name])
        
        rhs_l = list(itertools.product(*catersian_product))
        for rhs in rhs_l:
            rule = SimpleRule(origin, list(rhs))
            rule_set.add(rule)
    
    # compress into string
    rulestr_set = set()
    for rule in rule_set:
        rulestr = str(rule).strip()
        rulestr_set.add(rulestr)
    
    if debug_rules:
        for rule in debug_rules:
            if rule not in rulestr_set: 
                import pdb; pdb.set_trace()
    return rulestr_set


def larkstr2rulelist(lark_str, rhs_sep=None):
    """
    Convert lark grammar string to list of rules.
    TODO: use load_grammar function from lark
    """
    for raw_rule in lark_str.split("\n"):
        lhs, rhs = split_rule(raw_rule)
        rhs_l = rhs.split("|")
        for rhs in rhs_l:
            rhs = rhs.strip()
            if rhs_sep is not None:
                rhs = rhs.split(rhs_sep)
                rule = SimpleRule(lhs, rhs)
            else:
                # treat rhs as a single token, which is enough 
                # for checking grammar validity bc. the the resulting string is the same
                rule = SimpleRule(lhs, (rhs,) )
            yield rule

def check_grammar_validity(valid_rules, pred_lark_str):
    """
    Check if the grammar (i.e., bnf_str produced by model) is valid
    """
    for rule in larkstr2rulelist(pred_lark_str):
        if rule.origin not in skipped_nonterminal_names and rule not in valid_rules:
            logger.debug(f"Found invalid rule {rule}")
            return False
    return True

def check_grammar_correctness(tgt_rules, pred_lark_str, debug=False):
    """
    Evaluate the correctness of the grammar
    """
    if pred_lark_str is None:
        return False
    tgt_ruleset = set(tgt_rules)
    pred_ruleset = set(larkstr2rulelist(pred_lark_str))

    if debug:
        logger.debug(f"Rules in pred but not in tgt: {pred_ruleset - tgt_ruleset}")
        logger.debug(f"Rules in tgt but not in pred: {tgt_ruleset - pred_ruleset}")

    return pred_ruleset == tgt_ruleset

def gen_min_lark(program, parser):
    """
    Obtain the minimal grammar from a program
    """
    parse_trees = []
    if "\n" in program:
        program = program.split("\n")
        for line in program:
            parse_tree = parser.parse(line)
            parse_trees.append(parse_tree)
    else:
        parse_tree = parser.parse(program)
        parse_trees.append(parse_tree)
    grammar = extract_min_grammar_from_trees(parse_trees)
    return grammar

def program2rules(program, parser):
    try:
        tree = parser.parse(program)
        rule_list = tree2rulelist(tree)
        return " ## ".join([rule.to_bnf() for rule in rule_list])
    except:
        # there are some bad cases, see run_parse_smc.py
        return program
    
def aggregate_grammar_from_examples(examples, parser):
    """
    给定若干带 target 程序的 Example，用全局 parser 抽出每个例子的最小语法，
    再做并集，返回：
      - agg_lark_grammar: Lark 格式的语法字符串
      - agg_bnf_grammar:  BNF 格式的语法字符串
    如果一个有效规则都抽不到，返回 (None, None)
    """
    rule_set = set()
    for ex in examples:
        try:
            min_lark = gen_min_lark(ex.target, parser)
            for r in larkstr2rulelist(min_lark):
                rule_set.add(r)
        except Exception as e:
            logger.warning(f"failed to extract rules from exemplar {getattr(ex, 'source', '')} due to {e}")

    if not rule_set:
        return None, None

    agg_lark_grammar = rulelist2larkstr(rule_set)
    agg_bnf_grammar = lark2bnf(agg_lark_grammar)
    return agg_lark_grammar, agg_bnf_grammar

def rules_to_lark_grammar(rules):
    """
    将一组 SimpleRule 转成 lark 格式语法字符串。
    """
    return rulelist2larkstr(rules)
