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
      - child_to_parent_rules: 任意符号 -> 以该符号出现在 RHS 的父规则集合
      - nt_forward: 非终结符 -> RHS 中出现过的子符号集合（图的正向边）
      - nt_reverse: 符号 -> 所有以它为子符号的父非终结符集合（图的反向边）
      - nt_components: 非终结符 -> 所在连通块 id（简单版社区/模块）
      - known_symbols: 所有出现过的终结符 token 列表
    """
    symbol_to_rules = collections.defaultdict(set)
    lhs_to_rules = collections.defaultdict(set)
    child_to_parent_rules = collections.defaultdict(set)
    nt_forward = collections.defaultdict(set)
    nt_reverse = collections.defaultdict(set)
    known_symbols = set()

    for rule in global_rules:
        lhs = rule.origin
        lhs_to_rules[lhs].add(rule)
        for sym in rule.expansion:
            # 终结符（带引号）
            if isinstance(sym, str) and sym.startswith("\"") and sym.endswith("\""):
                raw = sym.strip("\"")
                for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", raw):
                    symbol_to_rules[token].add(rule)
                    known_symbols.add(token)
                # 终结符也算一个“子符号”，方便向上扩散
                child_to_parent_rules[sym].add(rule)
                nt_forward[lhs].add(sym)
                nt_reverse[sym].add(lhs)
            else:
                # 非终结符 / 其它符号
                child_to_parent_rules[sym].add(rule)
                nt_forward[lhs].add(sym)
                nt_reverse[sym].add(lhs)

    # 简单版“社区”：对 nt_forward / nt_reverse 建无向图，做连通分量
    nt_graph = collections.defaultdict(set)
    nonterminals = set(lhs_to_rules.keys())

    # 确保孤立 NT 也在图里（否则后面遍历会漏）
    for nt in nonterminals:
        _ = nt_graph[nt]

    for a, children in nt_forward.items():
        if a not in nonterminals:
            continue
        for b in children:
            # b 是非终结符：1) 不带引号 2) 也是某条规则的 LHS
            if isinstance(b, str) and (not b.startswith("\"")) and (b in nonterminals):
                nt_graph[a].add(b)
                nt_graph[b].add(a)


    nt_components = {}
    comp_id = 0
    visited = set()
    for nt in nonterminals:
        if nt in visited:
            continue
        comp_nodes = []
        stack = [nt]
        visited.add(nt)
        while stack:
            u = stack.pop()
            comp_nodes.append(u)
            for v in nt_graph[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        for u in comp_nodes:
            nt_components[u] = comp_id
        comp_id += 1

    return {
        "symbol_to_rules": symbol_to_rules,
        "lhs_to_rules": lhs_to_rules,
        "child_to_parent_rules": child_to_parent_rules,
        "nt_forward": nt_forward,
        "nt_reverse": nt_reverse,
        "nt_components": nt_components,
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


def collect_symbols_for_induction(
    draft_program: str,
    parser,
    grammar_index=None,
    symbol_mapper=None,
):
    """
    从草稿程序中抽取 symbol 集合，并做一次“先分组再修复”的幻觉修复：
      1) raw_tokens: 不做过滤，先尽量多抓 token
      2) in_grammar: 已经在 DSL 里的 token（在 known_symbols 中）
      3) out_of_grammar: 不在 DSL 里的 token，尝试用 SymbolMapper 映射到最近的 DSL 符号
      4) 最终返回 in_grammar + repaired 的去重结果
    """
    # 1. 先尽量多拿 raw tokens，不在这里用 grammar_index 过滤
    if parser is not None:
        # 注意：这里不要传 grammar_index，避免在 extract 阶段就截断掉“非 DSL 的” token
        raw_tokens = extract_symbols_from_parsed_program(
            draft_program, parser, grammar_index=None
        )
    else:
        raw_tokens = extract_symbols_from_program(draft_program)

    # 如果没有 grammar_index，就没法知道哪些在 DSL 里，直接去重返回
    if grammar_index is None:
        return sorted(set(raw_tokens))

    known = set(grammar_index.get("known_symbols", []))

    # 2. 拆成“已在 DSL 里的”和“不在 DSL 里的”
    in_grammar = [t for t in raw_tokens if t in known]
    out_of_grammar = [t for t in raw_tokens if t not in known]

    # 3. 对不在 DSL 里的 token 用 SymbolMapper 尝试修复
    repaired = []
    if symbol_mapper is not None:
        for t in out_of_grammar:
            mapped = symbol_mapper.map_symbol(t)
            # 保险起见，再确认一下 mapped 也在 known 里
            if mapped is not None and mapped in known:
                repaired.append(mapped)

    # 4. 最终 symbol 集合 = 已在 DSL 里的 + 修复成功的
    symbols = sorted(set(in_grammar + repaired))
    return symbols


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
    # 1. 从草稿中抽取 symbol 集合，并做幻觉修复
    symbols = collect_symbols_for_induction(
        draft_program,
        parser,
        grammar_index=grammar_index,
        symbol_mapper=symbol_mapper,
    )

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

def induce_grammar_by_spreading_activation(
    draft_program,
    global_rules,
    grammar_index,
    parser=None,
    symbol_mapper=None,
    start_lhs: str = None,
    k_hop_up: int = 2,
    k_hop_down: int = 2,
    large_nt_threshold: int = 20,
    large_nt_percentile: float = 80.0,
    max_rules: int = 600,
    min_rules: int = 200,
    max_k_up: int = 6,
    max_k_down: int = 6,
):

    """
    图扩散版静态语法归纳（简化版）：
      - 从 draft_program 提取 symbol
      - 经 SymbolMapper 做幻觉修正
      - 用 symbol_to_rules 找到含这些 symbol 的规则 -> 初始 rule_set（种子）
      - 在非终结符图上做向上扩散（父节点、模块），再做受限向下扩散（避免大枚举 NT 全展开）
      - 返回子语法的 lark 字符串；若匹配不上，返回 None
    """
    if grammar_index is None:
        return None
    
    def _percentile_int(values, pct: float) -> int:
        values = sorted(int(v) for v in values)
        if not values:
            return 0
        if pct <= 0:
            return values[0]
        if pct >= 100:
            return values[-1]
        idx = int(round((pct / 100.0) * (len(values) - 1)))
        return values[idx]

    def _auto_large_nt_threshold(lhs_to_rules_dict) -> int:
        counts = [len(v) for v in lhs_to_rules_dict.values()]
        thr = _percentile_int(counts, large_nt_percentile)
        return max(1, int(thr))

    def _auto_k_up(seed_nts, start_symbol, nt_reverse_dict, nonterminals_set) -> int:
        if not seed_nts:
            return 2

        # start_symbol 可能是 str，也可能是 list/tuple/set
        if isinstance(start_symbol, (list, tuple, set)):
            start_set = set(start_symbol)
        elif start_symbol:
            start_set = {start_symbol}
        else:
            start_set = set()

        if not start_set:
            return 2

        # 如果 seed 已经包含任一 start，距离就是 0
        if seed_nts.intersection(start_set):
            return 0

        frontier = set(seed_nts)
        visited = set(frontier)
        dist = 0

        while frontier and dist < max_k_up:
            dist += 1
            new_frontier = set()
            for nt in frontier:
                for parent in nt_reverse_dict.get(nt, []):
                    if parent not in nonterminals_set:
                        continue
                    if parent in start_set:
                        return dist
                    if parent not in visited:
                        visited.add(parent)
                        new_frontier.add(parent)
            frontier = new_frontier

        return 2


    def _select_rules_for_large_nt(rules, keep_n: int):
        rules_sorted = sorted(rules, key=lambda r: str(r))
        return rules_sorted[:keep_n]


    # 1. 从草稿中抽取 symbol 集合，并做幻觉修复
    symbols = collect_symbols_for_induction(
        draft_program,
        parser,
        grammar_index=grammar_index,
        symbol_mapper=symbol_mapper,
    )


    symbol_to_rules = grammar_index["symbol_to_rules"]
    lhs_to_rules = grammar_index["lhs_to_rules"]
    if large_nt_threshold is None or large_nt_threshold <= 0:
        large_nt_threshold = _auto_large_nt_threshold(lhs_to_rules)

    child_to_parent = grammar_index["child_to_parent_rules"]
    nt_forward = grammar_index.get("nt_forward", {})
    nt_reverse = grammar_index.get("nt_reverse", {})
    nt_components = grammar_index.get("nt_components", {})

    def _add_rules_for_nt(nt: str) -> bool:
        all_rules = list(lhs_to_rules.get(nt, []))
        if not all_rules:
            return True

        if len(all_rules) > large_nt_threshold:
            all_rules = _select_rules_for_large_nt(all_rules, large_nt_threshold)

        for r in all_rules:
            if len(rule_set) >= max_rules:
                return False
            rule_set.add(r)
        return True


    # 2. 初始 rule_set：所有包含这些 symbol 的规则
    rule_set = set()
    seed_nts = set()
    for s in symbols:
        for r in symbol_to_rules.get(s, []):
            if len(rule_set) >= max_rules:
                return rulelist2larkstr(rule_set)
            rule_set.add(r)
            seed_nts.add(r.origin)

    if not rule_set:
        return None

    activated_nts = set(seed_nts)
    nonterminals = set(lhs_to_rules.keys())
    if k_hop_up is None or k_hop_up <= 0:
        k_hop_up = _auto_k_up(seed_nts, start_lhs, nt_reverse, nonterminals)
    k_hop_up = min(int(k_hop_up), int(max_k_up))


    # 2.1 模块激活：把和 seed_nts 在同一个连通块里的 NT 一起激活
    comp_to_nts = collections.defaultdict(set)
    for nt, cid in nt_components.items():
        comp_to_nts[cid].add(nt)
    for nt in list(seed_nts):
        cid = nt_components.get(nt, None)
        if cid is None:
            continue
        for other in comp_to_nts[cid]:
            if other not in activated_nts:
                activated_nts.add(other)
                if not _add_rules_for_nt(other):
                    return rulelist2larkstr(rule_set)


    # 3. 向上扩散（沿 child_to_parent / nt_reverse）
    frontier = set(activated_nts)
    for _ in range(k_hop_up):
        new_frontier = set()
        for nt in frontier:
            # 从 NT 往父 NT
            for parent_nt in nt_reverse.get(nt, []):
                if parent_nt not in activated_nts:
                    activated_nts.add(parent_nt)
                    new_frontier.add(parent_nt)
                    if not _add_rules_for_nt(parent_nt):
                        return rulelist2larkstr(rule_set)

        frontier = new_frontier
        if not frontier:
            break

    # 如果指定了 start_lhs，可以保证它/它们在激活集合里
    if start_lhs is not None:
        # 有的语法 start_lhs 是字符串，有的是列表，这里统一成列表来处理
        if isinstance(start_lhs, (list, tuple, set)):
            start_symbols = list(start_lhs)
        else:
            start_symbols = [start_lhs]

        for s in start_symbols:
            if s not in activated_nts:
                activated_nts.add(s)
                if not _add_rules_for_nt(s):
                    return rulelist2larkstr(rule_set)



    # 4. 受限向下扩散（预算/阈值控制）
    frontier = set(activated_nts)
    down_steps = k_hop_down if (k_hop_down is not None and k_hop_down > 0) else max_k_down

    for _ in range(down_steps):
        if len(rule_set) >= max_rules:
            break

        # 只有在“自动 k_down 模式”（k_hop_down<=0）才用 min_rules 当停止目标
        if (k_hop_down is None or k_hop_down <= 0) and len(rule_set) >= min_rules:
            break

        new_frontier = set()
        for nt in frontier:
            all_rules_for_lhs = list(lhs_to_rules.get(nt, []))
            if not all_rules_for_lhs:
                continue

            # 大分支：选一部分，不再直接跳过
            if len(all_rules_for_lhs) > large_nt_threshold:
                selected_rules = _select_rules_for_large_nt(all_rules_for_lhs, large_nt_threshold)
            else:
                selected_rules = all_rules_for_lhs

            # 加规则（受 max_rules 约束）
            for r in selected_rules:
                if len(rule_set) >= max_rules:
                    break
                rule_set.add(r)

            if len(rule_set) >= max_rules:
                break

            # 向下激活子 NT
            for r in selected_rules:
                for sym in r.expansion:
                    if isinstance(sym, str) and not sym.startswith("\""):
                        if sym not in activated_nts:
                            activated_nts.add(sym)
                            new_frontier.add(sym)

        frontier = new_frontier
        if not frontier:
            break


    # 5. 打包成 lark 语法
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
