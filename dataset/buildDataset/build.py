#venv\Scripts\activate
import os
import re
import math
import json
import subprocess
from concurrent.futures import as_completed
import sys
import pandas as pd
import requests
import textstat
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any
from dotenv import load_dotenv
import textwrap
import os
from datetime import datetime, timedelta
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Optional
import csv
import tqdm
import numpy as np
import shutil
os.environ["PYTHONUTF8"] = "1"


REPO_BASE_DIR = Path(os.environ.get(
    "REPO_BASE_DIR", Path(__file__).resolve().parent / "cloned_repos"))
OUTPUT_DIR = Path("./dataset/data")
REPO_BASE_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {
    '.cs',                          # C#
    '.c', '.cpp', '.cc', '.h', '.hpp', '.cxx', '.hxx', # C/C++
    '.go',                          # Golang
    '.java',                        # Java
    '.js', '.jsx',                  # JavaScript
    '.kt', '.kts',                  # Kotlin
    '.php',                         # PHP
    '.py',                          # Python
    '.scala',                       # Scala
    '.swift',                       # Swift
}

dotenv_path = Path(".") / ".env"
load_dotenv(dotenv_path)

TOKENS = [
    os.environ.get("GITHUB_TOKEN_1"),
    os.environ.get("GITHUB_TOKEN_2"),
    os.environ.get("GITHUB_TOKEN_3")
]
TOKENS = [t for t in TOKENS if t]
if not TOKENS:
    raise ValueError("At least one GITHUB_TOKEN is required")
MAX_WORKERS = len(TOKENS)

print("GitHub token loaded successfully.")
# --- Utility Functions ---
def _resolve_semgrep():
    """
    Locate the semgrep executable.

    Falls back to the interpreter's Scripts/bin directory, then to
    `python -m semgrep`, because a bare "semgrep" only works when the active
    venv's Scripts directory is on PATH - which silently zeroed every
    semgrep_findings_count when the venv was not activated.
    """
    def _use(path):
        # semgrep.exe is a launcher that shells out to its sibling `pysemgrep`,
        # so its Scripts directory has to be on PATH or it exits 127.
        folder = str(Path(path).parent)
        if folder not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
        return [str(path)]

    found = shutil.which("semgrep")
    if found:
        return _use(found)

    scripts = Path(sys.executable).parent
    for candidate in (scripts / "semgrep.exe", scripts / "semgrep",
                      scripts / "Scripts" / "semgrep.exe"):
        if candidate.exists():
            return _use(candidate)

    # user-site install (pip install --user) - the usual Windows layout
    try:
        import site
        roots = [site.getuserbase(), str(Path(site.getusersitepackages()).parent)]
        for base in filter(None, roots):
            for candidate in (Path(base) / "Scripts" / "semgrep.exe",
                              Path(base) / "Scripts" / "semgrep",
                              Path(base) / "bin" / "semgrep"):
                if candidate.exists():
                    return _use(candidate)
    except Exception:
        pass

    return [sys.executable, "-m", "semgrep"]


SEMGREP_CMD = _resolve_semgrep()


def remove_repo_tree(path):
    """
    Delete a cloned repo, clearing the read-only bits git leaves on Windows.

    A silent failure here is dangerous: the stale checkout would be reused by the
    next repo that hashes to the same directory.
    """
    import stat

    def _clear_readonly(func, target, _exc):
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except Exception:
            pass

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=lambda f, t, e: _clear_readonly(f, t, e))
    else:
        shutil.rmtree(path, onerror=_clear_readonly)
    return not os.path.exists(path)


def get_git_output(repo: Path, args: List[str]) -> List[str]:
    cmd = ["git", "-C", str(repo)] + args
    try:
        result = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        return result.strip().split('\n') if result.strip() else []
    except subprocess.CalledProcessError:
        return []

def get_time_based_shas(
    local_repo: Path, 
    merged_at_str: str, 
    window_days: int = 14
) -> Dict[str, Optional[str]]:

    merge_date = datetime.fromisoformat(merged_at_str.replace("Z", "+00:00"))
    results = {}

    for months in [1, 3]:
        # The start of our search 
        start_search = merge_date + relativedelta(months=months)
        # The end of our 'soft boundary' 
        end_search = start_search + timedelta(days=window_days)
        
        # We use --since AND --until to trap the search in a specific window
        args = [
            "rev-list", "--all", "--reverse",
            f"--since={start_search.isoformat()}",
            f"--until={end_search.isoformat()}",
            "-n", "1"
        ]
        
        shas = get_git_output(local_repo, args)
        results[f"{months}_month_mark"] = shas[0] if shas else None
            
    return results

def get_target_shas(local_repo: Path, merge_sha: str) -> List[Optional[str]]:
    args = ["rev-list", "--topo-order", "--reverse", "--all", f"^{merge_sha}"]
    commits = get_git_output(local_repo, args)
    indices = [4, 9, 19]
    return [commits[i] if i < len(commits) else None for i in indices]


def calculate_entropy(text):
    if not text or len(text.strip()) == 0:
        return 0.0
    words = tokenize(text)
    if not words:
        return 0.0
    counts = Counter(words)
    probs = [c / len(words) for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs)

def doc_redundancy(doc_text): # get percent of tokens that are repeated within the documentation
    tokens = tokenize(doc_text)
    if not tokens:
        return 0.0
    
    unique = len(set(tokens))
    return 1 - (unique / len(tokens))

def doc_code_overlap(doc_text, code_text): # get percent of tokens that overlap between code and text
    doc_tokens = set(tokenize(doc_text))
    code_tokens = set(tokenize(code_text))

    if not doc_tokens or not code_tokens:
        return 0.0
    overlap = doc_tokens.intersection(code_tokens)
    return len(overlap) / len(doc_tokens)

def strip_comments(text, file_extension=None):

    ext = file_extension.lower() if file_extension else ""
    parser = _get_ts_parser(ext)
    if parser is None:
        return text

    parse_text = "<?php\n" + text if ext == ".php" else text
    parse_bytes = parse_text.encode("utf-8")
    tree = parser.parse(parse_bytes)

    remove_ranges = []

    def visit(node):
        if "comment" in node.type:
            remove_ranges.append((node.start_byte, node.end_byte))
            return
        if (ext == ".py" and node.type == "string"
                and node.parent is not None and node.parent.type in ("module", "block")):
            remove_ranges.append((node.start_byte, node.end_byte))
            return
        for child in node.children:
            visit(child)

    visit(tree.root_node)
    remove_ranges.sort()

    kept = bytearray()
    cursor = 0
    for start, end in remove_ranges:
        kept += parse_bytes[cursor:start]
        cursor = end
    kept += parse_bytes[cursor:]

    result = kept.decode("utf-8", errors="replace")
    if ext == ".php":
        result = result[len("<?php\n"):]

    # Only keep lines that still have content after comment removal, same as
    # the original function's behavior.
    clean_lines = [line.rstrip() for line in result.splitlines() if line.strip()]
    return "\n".join(clean_lines)

def tokenize(text): #this tokenizer will split words based on the aplabetical content, 
    if not text:
        return []
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower())


def find_documentation_header(lines, start_line, file_extension=None):
    ext = file_extension.lower() if file_extension else ""
    parser = _get_ts_parser(ext)
    if parser is None:
        return start_line

    MAX_BLANK_LINES = 5
    php_offset = 1 if ext == ".php" else 0

    full_text = "\n".join(lines)
    parse_text = "<?php\n" + full_text if ext == ".php" else full_text
    tree = parser.parse(parse_text.encode("utf-8"))

    target_row = (start_line - 1) + php_offset

    def find_node_at_row(node, row):
        for child in node.children:
            if child.start_point[0] > row or child.end_point[0] < row:
                continue
            if child.end_point[0] == row and child.end_point[1] == 0:
                continue
            deeper = find_node_at_row(child, row)
            return deeper if deeper is not None else child
        return None

    node = find_node_at_row(tree.root_node, target_row)
    if node is None:
        return start_line

    while node.parent is not None and node.parent.start_point[0] == node.start_point[0]:
        node = node.parent

    def _is_own_line_comment(cnode):
        """
        True only if the comment is the first thing on its line.

        A trailing comment - `define(...); // 1 hour` - is its own sibling node in
        the parse tree, so walking back over it would drag the preceding statement
        into the "documentation" block along with it.
        """
        row = cnode.start_point[0] - php_offset
        if row < 0 or row >= len(lines):
            return False
        return not lines[row][:cnode.start_point[1]].strip()

    adj_row = node.start_point[0]
    prev = node.prev_sibling
    while prev is not None and "comment" in prev.type and _is_own_line_comment(prev):
        gap = adj_row - prev.end_point[0] - 1
        if gap > MAX_BLANK_LINES:
            break
        adj_row = prev.start_point[0]
        prev = prev.prev_sibling

    return adj_row - php_offset + 1


from tree_sitter import Parser as _TSParser
from tree_sitter_language_pack import get_language as _ts_get_language

_TS_LANGUAGE_MAP = {
    '.py': 'python',
    '.js': 'javascript', '.jsx': 'javascript',
    '.java': 'java',

    '.c': 'c', '.h': 'cpp',
    '.cpp': 'cpp', '.cc': 'cpp', '.hpp': 'cpp', '.cxx': 'cpp', '.hxx': 'cpp',
    '.cs': 'csharp',
    '.go': 'go',
    '.kt': 'kotlin', '.kts': 'kotlin',
    '.php': 'php',
    '.scala': 'scala',
    '.swift': 'swift',
}
_ts_parser_cache = {}


def _get_ts_parser(file_extension):
    lang_name = _TS_LANGUAGE_MAP.get(file_extension.lower())
    if lang_name is None:
        return None
    if lang_name not in _ts_parser_cache:
        _ts_parser_cache[lang_name] = _TSParser(_ts_get_language(lang_name))
    return _ts_parser_cache[lang_name]


_TS_FUNC_NODES = {
    'python':     {'function_definition'},
    'javascript': {'function_declaration', 'function_expression', 'arrow_function',
                   'method_definition', 'generator_function_declaration'},
    'go':         {'function_declaration', 'method_declaration', 'func_literal'},
    'java':       {'method_declaration', 'constructor_declaration'},
    'c':          {'function_definition'},
    'cpp':        {'function_definition'},
    'csharp':     {'method_declaration', 'constructor_declaration', 'local_function_statement'},
    'kotlin':     {'function_declaration'},
    'php':        {'function_definition', 'method_declaration', 'anonymous_function'},
    'scala':      {'function_definition'},
    'swift':      {'function_declaration', 'init_declaration'},
}


INCLUDE_ANONYMOUS_FUNCTIONS = True

_ANON_NODE_TYPES = {'arrow_function', 'function_expression', 'func_literal', 'anonymous_function'}

_TS_STRING_NODES = ('interpreted_string_literal', 'raw_string_literal', 'string',
                    'string_literal', 'template_string')

_TS_QUOTES = "'\"`"


def _ts_node_text(node, src_bytes):
    return src_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _ts_declarator_name(node, src_bytes):
    """
    C/C++ put the name under nested `declarator` fields:
    function_definition -> function_declarator -> identifier.
    """
    cur = node.child_by_field_name("declarator")
    for _ in range(6):
        if cur is None:
            return None
        if cur.type in ("identifier", "field_identifier", "qualified_identifier",
                        "operator_name", "destructor_name", "type_identifier"):
            return _ts_node_text(cur, src_bytes)
        cur = cur.child_by_field_name("declarator")
    return None


def _ts_leading_identifier(node, src_bytes):
    """
    Kotlin (and friends) expose the name as an unnamed leading child rather than
    a `name` field. Scan children up to the parameter list / body so a parameter
    is never mistaken for the function name.
    """
    for i in range(node.child_count):
        field = node.field_name_for_child(i)
        child = node.child(i)
        if field in ("parameters", "parameter", "body", "return_type", "type"):
            break
        if "parameter" in child.type or child.type in (
                "function_body", "statement_block", "compound_statement", "block"):
            break
        if child.type in ("simple_identifier", "identifier", "type_identifier", "field_identifier"):
            return _ts_node_text(child, src_bytes)
    return None


def _has_error_ancestor(node):
    """True if the node sits inside a subtree the grammar failed to parse."""
    parent = node.parent
    while parent is not None:
        if parent.type == "ERROR":
            return True
        parent = parent.parent
    return False


def _ts_function_name(node, src_bytes):
    """
    Best-effort name for a function node.

    Falls back through the enclosing syntax so that idioms lizard reports as
    unnamed still get a usable label: const foo = () => {}, { foo: fn },
    foo = function () {}, and table-driven tests such as
    t.Run("InvalidURL", func(t *testing.T) {...}).
    """
    named = node.child_by_field_name("name")
    if named is not None:
        return _ts_node_text(named, src_bytes)

    if node.type not in _ANON_NODE_TYPES:
        declared = _ts_declarator_name(node, src_bytes)
        if declared:
            return declared
        leading = _ts_leading_identifier(node, src_bytes)
        if leading:
            return leading

    parent = node.parent
    if parent is None:
        return "(anonymous)"

    if parent.type in ("variable_declarator", "short_var_declaration", "property_declaration"):
        nm = parent.child_by_field_name("name") or parent.child_by_field_name("left")
        if nm is not None:
            return _ts_node_text(nm, src_bytes)

    if parent.type in ("pair", "keyed_element"):
        key = parent.child_by_field_name("key")
        if key is not None:
            return _ts_node_text(key, src_bytes).strip(_TS_QUOTES)

    if parent.type in ("assignment_expression", "assignment"):
        left = parent.child_by_field_name("left")
        if left is not None:
            return _ts_node_text(left, src_bytes)

    # t.Run("InvalidURL", func(t *testing.T) {...}) / describe("x", () => {...})
    if parent.type in ("argument_list", "arguments"):
        call = parent.parent
        if call is not None:
            callee = call.child_by_field_name("function")
            label = _ts_node_text(callee, src_bytes) if callee is not None else ""
            for child in parent.children:
                if child.type in _TS_STRING_NODES:
                    arg = _ts_node_text(child, src_bytes).strip(_TS_QUOTES)
                    if arg:
                        return "{}[{}]".format(label, arg) if label else arg
            if label:
                return label

    return "(anonymous)"


def find_functions(source_text, file_extension):
    """
    Identify every function in source_text with tree-sitter.

    Returns a list of dicts, sorted by position:
        {name, text, start_byte, end_byte, start_line, end_line, node_type,
         anonymous, has_error}

    text is sliced from the node's *byte* range, not its line range. That
    distinction matters: a callback such as .then(function () { ... }) starts
    mid-line, so slicing by line would capture ".then(function () {" and lose the
    real boundary. Line numbers are 1-based and inclusive, for the callers that
    still need them (changed-line filtering, semgrep attribution).
    """
    ext = file_extension.lower() if file_extension else ""
    lang_name = _TS_LANGUAGE_MAP.get(ext)
    parser = _get_ts_parser(ext)
    if parser is None or lang_name is None:
        return []

    wanted = _TS_FUNC_NODES.get(lang_name)
    if not wanted:
        return []

    php_offset = 1 if ext == ".php" else 0
    parse_text = "<?php\n" + source_text if ext == ".php" else source_text
    src_bytes = parse_text.encode("utf-8")
    tree = parser.parse(src_bytes)

    results = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in wanted and not _has_error_ancestor(node):

            anonymous = (node.type in _ANON_NODE_TYPES
                         and node.child_by_field_name("name") is None)
            if INCLUDE_ANONYMOUS_FUNCTIONS or not anonymous:
                results.append({
                    "name": _ts_function_name(node, src_bytes),
                    "text": _ts_node_text(node, src_bytes),
                    "start_byte": node.start_byte,
                    "end_byte": node.end_byte,
                    "start_line": node.start_point[0] + 1 - php_offset,
                    "end_line": node.end_point[0] + 1 - php_offset,
                    "node_type": node.type,
                    "anonymous": anonymous,
                    "has_error": node.has_error,
                })
        stack.extend(node.children)

    results.sort(key=lambda f: (f["start_line"], f["start_byte"]))
    return results


def _lizard_span(func_info):
    """(start_line, end_line) for one lizard functions_info entry, or None."""
    location = func_info.get("location", "")
    try:
        if '@' in location:
            parts = location.split('@')
            if len(parts) >= 3 and '-' in parts[1]:
                s, e = map(int, parts[1].split('-'))
                return s, e
        if ' ' in location:
            file_line, _ = location.rsplit(' ', 1)
            if ':' in file_line:
                s = int(file_line.rsplit(':', 1)[1])
                return s, s + func_info.get("length", 1) - 1
    except (ValueError, IndexError):
        return None
    return None


def match_lizard_metrics(lizard_functions, start_line, end_line):
    """
    Pair a tree-sitter function with lizard's metrics for the same region.

    Tree-sitter owns the boundaries; lizard is kept only for cyclomatic
    complexity, nloc and parameter counts, so those stay comparable with prior
    work. Matching is by largest line overlap, which tolerates the small boundary
    disagreements that motivated the switch. Returns None when lizard saw nothing
    in that region.
    """
    best, best_overlap = None, 0
    for f in lizard_functions:
        span = _lizard_span(f)
        if span is None:
            continue
        fs, fe = span
        overlap = min(end_line, fe) - max(start_line, fs) + 1
        if overlap > best_overlap:
            best, best_overlap = f, overlap
    return best


def extract_documentation_text(block_text, file_extension):
    """
    Extract comments/docstrings from an already-sliced block of source.

    Line-range variant of this lives in extract_documentation, which now
    delegates here. Taking text directly lets callers pass a byte-accurate
    function body rather than a line window.
    """
    ext = file_extension.lower() if file_extension else ""
    parser = _get_ts_parser(ext)
    if parser is None:
        return []

    parse_text = "<?php\n" + block_text if ext == ".php" else block_text
    parse_bytes = parse_text.encode("utf-8")
    tree = parser.parse(parse_bytes)
    extracted_docs = []

    def visit(node):
        if "comment" in node.type:
            text = parse_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            extracted_docs.append(text.strip())
            return

        if (ext == ".py" and node.type == "string"
                and node.parent is not None and node.parent.type in ("module", "block")):
            text = parse_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            extracted_docs.append(text.strip())
            return

        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return extracted_docs



def extract_documentation(lines, start, end, file_extension):
    """
    Extracts documentation (comments + docstrings) from the given line range
    using a tree-sitter parse tree.

    Thin line-range wrapper around extract_documentation_text.
    """
    raw_block = "\n".join(lines[start - 1:end])
    return extract_documentation_text(raw_block, file_extension), lines

# --- Lizard Parsing ---
_FUNCTION_PATTERN = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(.+)")
_SUMMARY_PATTERN = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(.+)$")

def parse_detailed_lizard(stdout_text: str) -> Dict[str, Any]:
    functions_info = []
    running_total_nloc = 0
    max_values = {
        "max_nloc": 0, "max_ccn": 0, "max_tokens": 0, 
        "max_params": 0, "max_length": 0
    }
    
    lines = stdout_text.splitlines()
    for line in lines:
        f_match = _FUNCTION_PATTERN.match(line)
        if f_match:
            nloc, ccn, tokens, params, length, loc = f_match.groups()
            nloc, ccn, tokens, params, length = int(nloc), int(ccn), int(tokens), int(params), int(length)
           # print(f"Parsed function: nloc={nloc}, ccn={ccn}, tokens={tokens}, params={params}, length={length}, location={loc.strip()}")
            functions_info.append({
                "nloc": nloc, "ccn": ccn, "tokens": tokens,
                "params": params, "length": length, "location": loc.strip()
            })
            
            max_values["max_nloc"] = max(max_values["max_nloc"], nloc)
            max_values["max_ccn"] = max(max_values["max_ccn"], ccn)
            max_values["max_tokens"] = max(max_values["max_tokens"], tokens)
            max_values["max_params"] = max(max_values["max_params"], params)
            max_values["max_length"] = max(max_values["max_length"], length)
            continue

        s_match = _SUMMARY_PATTERN.match(line)
        if s_match:
            file_nloc = int(s_match.group(1))
            file_name = s_match.group(6).strip()
            if file_name.lower() != "total":
                running_total_nloc += file_nloc

    if not functions_info and running_total_nloc == 0:
        return {"_error": "no_metrics_found"}

    func_count = len(functions_info)
    avg_ccn = sum(f["ccn"] for f in functions_info) / func_count if func_count > 0 else 0.0

    result = {
        "total_nloc": running_total_nloc,
        "function_count": func_count,
        "avg_ccn": round(avg_ccn, 2),
        "functions_info": functions_info
    }
    result.update(max_values)
    
    return result

def find_file_at_commit(repo, sha, rel_path):
    cmd = [
        "git", "-C", str(repo),
        "ls-tree", "-r", "--name-only", sha
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)

    files = res.stdout.splitlines()
    # print()
    # print()
    # print(files)
    # print(rel_path)
    # print()
    # print()
    # exact match
    if rel_path in files:
        return rel_path


    filename = Path(rel_path).name

    matches = [f for f in files if filename == Path(f).name]

    if len(matches) == 1:
        return matches[0]
 
    return None

def getTurnover(local_repo: Path, rel_path: str, func_name: str, pr_doc_text: str, merge_sha: str, merged_at: str):
    month_shas = get_time_based_shas(local_repo, merged_at, window_days=14)
    number_shas = get_target_shas(local_repo, merge_sha)
    shas = [number_shas[0], number_shas[1], number_shas[2], month_shas['1_month_mark'], month_shas['3_month_mark']]
    
    file_extension = Path(rel_path).suffix.lower()
    res = []
    pr_tokens = set(tokenize(pr_doc_text))
    
    if not pr_tokens:
        return [None] * 5

    for sha in shas:
        if sha is None:
            res.append(None)   
            continue
        try:
            path_at_commit = find_file_at_commit(local_repo, sha, rel_path)

            if not path_at_commit:
                res.append(None)
                continue


            cmd_show = ["git", "-C", str(local_repo), "show", f"{sha}:{path_at_commit}"]
            file_res = subprocess.run(cmd_show, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            
            if file_res.returncode != 0: 
                print(file_res.stderr.strip())
                res.append(None)
                continue
            
            content = file_res.stdout
            lines = content.splitlines()

            try:
                # Locate the function in this later revision with tree-sitter.
                # Exact name first, then a substring match - but never on an empty
                # name: the previous `"" in e_name` test was always true and
                # silently matched the first function in the file.
                candidates = find_functions(content, file_extension)

                match = None
                for cand in candidates:
                    if func_name and cand["name"] == func_name:
                        match = cand
                        break

                if match is None and func_name:
                    for cand in candidates:
                        if cand["name"] and (func_name in cand["name"] or cand["name"] in func_name):
                            match = cand
                            break

                if match:
                    adj_start = find_documentation_header(lines, match["start_line"], file_extension)
                    doc_prefix = "\n".join(lines[adj_start - 1:match["start_line"] - 1])
                    block = (doc_prefix + "\n" + match["text"]) if doc_prefix.strip() else match["text"]
                    future_doc_text = " ".join(extract_documentation_text(block, file_extension))

                    future_tokens = set(tokenize(future_doc_text)) # tokenize same way we tokenize other text
                    if not future_tokens:
                        res.append((sha, 1.0)) # 100% turnover if docs were deleted
                    else:
                        overlap = len(pr_tokens.intersection(future_tokens)) # compare the text from old to new
                        turnover = 1.0 - (overlap / len(pr_tokens))
                        res.append((sha, round(turnover, 4)))
                else:
                    res.append(None) # file existed but function was removed

            except Exception as e:
                print(f"Turnover parsing error: {e}")
                res.append(None)
                    
        except Exception as e:
            res.append(None)
            print(f"Error occurred while processing {sha}: {e}")

    return res

# --- Main Miner ---
class AiDevMiner:
    def __init__(self):
        self.session = requests.Session()
        # if GITHUB_TOKEN:
        #     self.session.headers.update({"Authorization": f"token {GITHUB_TOKEN}"})
    def fetch_pr_details(self, owner, repo, pr_number):
        """Fetch the main PR metadata to get the head SHA."""
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
        r = self.session.get(url)
        if r.status_code != 200:
            # print(f"Failed to fetch PR {pr_number} details: {r.text}")
            return None
        return r.json()

    def transform_url(self, api_url):
        return api_url.replace("api.github.com/repos", "github.com").replace("api.github.com", "github.com")

    def get_repo(self, repo_url):
        # Qualify by owner: repo names are not unique (11 different owners in the
        # dev list have a repo called "awesome-kubernetes"). Keying on the bare
        # name let one owner's checkout be mined and labelled as another's.
        parts = repo_url.replace(".git", "").rstrip("/").split("/")
        owner, repo_name = (parts[-2], parts[-1]) if len(parts) >= 2 else ("_", parts[-1])
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{owner}__{repo_name}")
        local_path = REPO_BASE_DIR / safe
        if not local_path.exists():
            try:
                repo = subprocess.run(
                    ["git", "clone", repo_url, str(local_path)],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                # print(f"Skipping repo {repo_url}: {e}")
                return None
        return local_path

    def fetch_pr_files(self, owner, repo, pr_number):
        """Fetch files and patches changed in the PR using GitHub API."""
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
        files = []
        page = 1
        while True:
            r = self.session.get(url, params={"page": page, "per_page": 100})
            if r.status_code != 200:
                # print(f"Failed to fetch PR {pr_number} files: {r.text}")
                break
            page_files = r.json()
            if not page_files:
                break
            files.extend(page_files)
            page += 1
        return files

    def parse_patch(self, patch_text):
        """Return set of line numbers added in the patch."""
        lines = set()
        if not patch_text:
            return lines
        curr = 0
        for line in patch_text.splitlines():
            if line.startswith('@@'):
                m = re.search(r'\+(\d+)(?:,(\d+))?', line)
                if m:
                    curr = int(m.group(1))
            elif line.startswith('+') and not line.startswith('+++'):
                lines.add(curr)
                curr += 1
            elif not line.startswith('-'):
                curr += 1
        return lines 

    def run_quality_check(self, file_path):
        os.environ["PYTHONUTF8"] = "1"
        
        extension_map = {
            '.py': 'python',
            '.js': 'javascript', '.jsx': 'javascript',
            '.java': 'java',
            '.cs': 'csharp',
            '.go': 'golang',
            '.c': 'c', '.cpp': 'c', '.cc': 'c', '.h': 'c', '.hpp': 'c',
            '.kt': 'kotlin', '.kts': 'kotlin',
            '.php': 'php',
            '.scala': 'scala',
            '.swift': 'swift'
        }
        
        ext = Path(file_path).suffix.lower()
        lang_config = extension_map.get(ext)
        

        configs = ["p/security-audit", "p/default", "p/owasp-top-ten"]
        if lang_config:
            configs.append(f"p/{lang_config}")
        else:
            print(f"Warning: No language-specific rules found for {ext}")

        all_findings = {}

        for config in configs:
            cmd = [
                *SEMGREP_CMD,
                f"--config={config}",
                "--json",
                "--quiet",
                "--no-git-ignore", 
                str(file_path)
            ]

            try:
                res = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=180
                )

                # Check if Semgrep succeeded (0=no findings, 1=findings found)
                if res.returncode not in (0, 1):
                    print(f"  {config}: FAILED (Code {res.returncode})")
                    continue

                if res.stdout.strip():
                    data = json.loads(res.stdout)
                    all_findings[config] = data.get("results", [])
                else:
                    all_findings[config] = []

            except FileNotFoundError:
                raise RuntimeError(
                    "Semgrep executable not found (tried: {}). Install it or add its "
                    "Scripts directory to PATH - continuing would silently record "
                    "semgrep_findings_count=0 for every row.".format(" ".join(SEMGREP_CMD))
                )
            except subprocess.TimeoutExpired:
                print(f"Semgrep timed out on {file_path} with config {config}")
                continue
            except json.JSONDecodeError:
                print(f"Semgrep output was not valid JSON for {config}")
                continue

        return all_findings

    def process_pr(self, row, token): # currently processing form current stae, not the
        self.session.headers.update({"Authorization": f"token {token}"})
        results = []
        stats = {
            "total_files_seen": 0,
            "files_unsupported": 0,
            "files_processed": 0,
            "skipped_clone_fail": 0,
            "skipped_too_many_files": 0,
            "lizard_timeouts": 0,
            "fail_pr_fetch": 0
        }
                                                                                     
        try:
            repo_url = row['repo_url']
            owner, repo_name = "/".join(repo_url.split("/")[-2:]).replace(".git", "").split("/")
            pr_number = row['number']

            local_repo = self.get_repo(f"https://github.com/{owner}/{repo_name}.git")

            if not local_repo: # exit if failed to clone repo
                # print(f"Failed to clone repository for PR {pr_number}, skipping.")
                # print()
                # print()
                stats["skipped_clone_fail"] += 1
                return {"data": [], "stats": stats}
            fetch_cmd = [
                "git", "-C", str(local_repo), 
                "fetch", "origin", f"pull/{pr_number}/head"
            ]
            fetch_res = subprocess.run(fetch_cmd, capture_output=True, text=True)

            if fetch_res.returncode != 0:
                stats["fail_pr_fetch"] += 1
                return {"data": [], "stats": stats}
            pr_files = self.fetch_pr_files(owner, repo_name, pr_number) # get the files changed in pr
            stats["total_files_seen"] = len(pr_files)
            if len(pr_files) > 100:
                stats["skipped_too_many_files"] += 1
                return {"data": [], "stats": stats} 
            pr_details = self.fetch_pr_details(owner, repo_name, pr_number)
            if not pr_details: 
                stats["fail_pr_fetch"] += 1
                return {"data": [], "stats": stats}
              
            
  
            pr_commit_sha = pr_details.get('head', {}).get('sha')
            merge_sha = pr_details.get("merge_commit_sha")
            # print(f"Processing PR {pr_number} in repo {owner}/{repo_name}, commit SHA: {pr_commit_sha}")
            merged_at = row['merged_at']
            closed_at = row['closed_at']
            created_at = row['created_at']
            # print(merged_at, pr_commit_sha)
            for file_info in pr_files:
                rel_path = file_info['filename']
                file_extension = Path(rel_path).suffix.lower()
                if file_extension not in SUPPORTED_EXTENSIONS:
                    stats["files_unsupported"] += 1
                    continue # Ignore files using unsupported languages

                
     
                cmd_base = ["git", "-C", str(local_repo), "show", f"{pr_commit_sha}:{rel_path}"]
                res_base = subprocess.run(cmd_base, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                    
                if res_base.returncode != 0:                    
                    print(f"error pulling baseline for {rel_path} using file SHA")
                    continue

                baseline_content = res_base.stdout
                baseline_lines = baseline_content.splitlines()


                with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False, mode='w', encoding='utf-8') as tf:
                    tf.write(res_base.stdout)
                    tmp_path = tf.name
                try:
                    patch = file_info.get('patch', '')
                    changed_lines = self.parse_patch(patch)
                    # print(self.parse_patch(patch))
                        
                    if not os.path.exists(tmp_path):
                        continue

                    try:
                        cp = subprocess.run([sys.executable, "-m", "lizard", str(tmp_path)], capture_output=True, text=True, timeout=60)
                        metrics = parse_detailed_lizard(cp.stdout)
                        raw_findings = self.run_quality_check(tmp_path)
                        # if quality_res.returncode not in (0, 1): # semgrep returns 1 if it finds issues, so we allow both 0 and 1 as "successful" runs
                        #     print("SEMGRP ERROR:")
                        #     print(quality_res.stderr)
                         
                        # try:
                        #     findings = json.loads(quality_res.stdout).get("results", [])
                        # except json.decoder.JSONDecodeError as e:
                        #     clean_stdout = quality_res.stdout[:500].encode('ascii', 'replace').decode('ascii') 
                        #     print(f"semgrep output was not valid JSON: {e}\noutput snippet:\n{clean_stdout}")

                    except subprocess.TimeoutExpired:
                        print(f"Skipping file {rel_path} due to lizard timeout")
                        stats["lizard_timeouts"] += 1
                        
                        continue
                    except Exception as e:
                        print(f"Error running lizard on {rel_path}: {e}")
                        continue

                    lines =baseline_lines

                    lizard_functions = metrics.get("functions_info", [])

                    # Tree-sitter owns function boundaries; lizard is kept only for
                    # cyclomatic complexity / nloc / parameter counts.
                    for ts_func in find_functions(baseline_content, file_extension):
                        func_name = ts_func["name"]
                        body_start_line = ts_func["start_line"]
                        end_line = ts_func["end_line"]

                        # Walk back over any doc block sitting above the declaration.
                        start_line = find_documentation_header(lines, body_start_line, file_extension)

                        function_line_range = set(range(start_line, end_line + 1))
                        # Check if every single line in the function is present in the PR's changed lines
                        if not function_line_range.issubset(changed_lines):
                            continue

                        # Body comes from the node's byte range so a callback starting
                        # mid-line (".then(function () {") is captured from the correct
                        # offset; any doc block above it is prepended by line.
                        doc_prefix = "\n".join(lines[start_line - 1:body_start_line - 1])
                        code_text = textwrap.dedent(
                            (doc_prefix + "\n" + ts_func["text"]) if doc_prefix.strip() else ts_func["text"]
                        )

                        doc_list = extract_documentation_text(code_text, file_extension)
                        doc_text = " ".join(doc_list)
                        code_text_no_documentation = strip_comments(code_text, file_extension)

                        func = match_lizard_metrics(lizard_functions, body_start_line, end_line)
                    

                        findings = {}
                        findings_count = 0
                        for config_name, findings_t in raw_findings.items():
                            temp = []
                            for f in findings_t:
                                f_start_line = f.get('start', {}).get('line', 0)
                                if f_start_line >= start_line and f_start_line <= end_line:
                                    temp.append(f)
                                    
                                    findings_count += 1
                            if temp:
                                findings[config_name] = temp
                        # print(f"Previous function location: {func['location']}, extracted start line: {start_line}, end line: {end_line}")
                        # print("========================================")
                        # print("DOC TEXT:")
                        # print(doc_text)
                        # print("--------------------------------------")
                        # print("CODE TEXT:")
                        # print(code_text)
                        # print("========================================")
                        # break
                 
              

                        
                        if merge_sha is None or merged_at is None:
                            # If PR isn't merged, we can't look into the future
                            turnover_data = [-1, -1, -1, -1, -1]
                        else:

                            turnover_data = getTurnover(local_repo, rel_path, func_name, doc_text,  merge_sha, merged_at)
                        #print(turnover_data)
                        results.append({
                            "repo": row['repo_url'],
                                "pull_request": pr_number,
                                "label": row['agent'],
                                "file_path": rel_path,
                                "function_name": func_name,
                                "function_start_line": start_line,
                                "function_end_line": end_line,
                            #   "function_length": func["length"],
                                "function": code_text,
                                "loc": end_line - body_start_line + 1,
                                "sloc": func["nloc"] if func else np.nan,
                                "cyclomatic_complexity": func["ccn"] if func else np.nan,
                                "num_parameters": func["params"] if func else np.nan,
                                "doc_lines": len(doc_list),
                                "doc_text": doc_text,
                                "doc_entropy": round(calculate_entropy(doc_text), 4) if doc_text else np.nan,
                                "total_entropy": round(calculate_entropy(code_text), 4),
                                "doc_readability": textstat.flesch_reading_ease(doc_text) if doc_text else np.nan,
                                "semgrep_findings": findings,
                                "semgrep_findings_count": findings_count,
                                "doc_code_overlap": round(doc_code_overlap(doc_text, code_text_no_documentation), 4) if doc_text else np.nan,
                                "doc_redundancy": round(doc_redundancy(doc_text), 4) if doc_text else np.nan,
                                "pr_date_merged": merged_at,
                                "pr_date_created": created_at,
                                "pr_date_closed": closed_at,
                                "turnover_c5": turnover_data[0],
                                "turnover_c10": turnover_data[1],
                                "turnover_c20": turnover_data[2],
                                "turnover_m1": turnover_data[3],
                                "turnover_m3": turnover_data[4]
                        })
                    
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                stats["files_processed"] += 1
        except Exception as e:
            print(f"Skipping PR {row.get('number')}: {e}")
        #print(f"Finished processing PR {row.get('number')}, {total_smells} smells and {total_vulns} vulns found in pr.")
        # try:
        #     if local_repo and os.path.exists(local_repo):
        #         if not remove_repo_tree(local_repo):
        #             print(f"WARNING: could not fully delete {local_repo}; a stale "
        #                   f"checkout may be reused on a later PR")
        # except Exception as e:
        #     print(f"Warning: failed to delete repo {local_repo}: {e}")
        return {"data": results, "stats": stats}

# --- Main Execution ---
if __name__ == "__main__":
    os.environ["PYTHONUTF8"] = "1"
    # df = pd.read_parquet(r"G:\663P\dataset\data\all_pull_request.parquet").head(1000)#683
    # df = pd.read_parquet(r"G:\663P\dataset\data\all_pull_request_ballanced.parquet")
    df = pd.read_parquet(r"G:\663P\dataset\data\human_baseline_2021.parquet").head(1000) # done 1k
  
 
    miner = AiDevMiner()
    # WE MINE AGENT PRS, THEN WE MINE DEV PRS. DEV PRS ARE SAVED IN "dev_final_dataset_subset_b.csv"
    # WHILE AGENT GO TO "agent_final_dataset_subset_b.csv"

    # output_path = OUTPUT_DIR / "aidev_final_dataset.csv"
    # stats_path = OUTPUT_DIR / "mining_stats_human.json"
    output_path = OUTPUT_DIR / "dev_dataset_subset_data.csv"
    stats_path = OUTPUT_DIR / "stats_dev_subset_data.json"
    # output_path = OUTPUT_DIR / "agent_dataset_subset_data.csv"
    # stats_path = OUTPUT_DIR / "stats_agent_subset_data.json"
    if stats_path.exists():
        with open(stats_path, 'r') as sj:
            final_stats = Counter(json.load(sj))
    else:
        final_stats = Counter()

    with open(output_path, 'a', encoding='utf-8', newline='') as f:
        writer = None
        file_needs_header = f.tell() == 0
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_pr = {}
            for i, (_, row) in enumerate(df.iterrows()):
                token = TOKENS[i % len(TOKENS)]
                future = executor.submit(miner.process_pr, row, token)
                future_to_pr[future] = row.get('number', 'Unknown')


            for future in tqdm.tqdm(as_completed(future_to_pr), total=len(future_to_pr), desc="Mining PRs"):
                try:
                    results_list_raw = future.result()
                    results_list = results_list_raw["data"]
                    final_stats.update(results_list_raw["stats"])
                    if not results_list:
                        continue
                        
                    for row_result in results_list:
                        if writer is None:
                            writer = csv.DictWriter(f, fieldnames=row_result.keys())
                            if file_needs_header:
                                writer.writeheader()
                                file_needs_header = False
                        
                        writer.writerow(row_result)
                
                except Exception as e:
                    pr_num = future_to_pr[future]
                    print(f"\n[ERROR] PR {pr_num} generated an exception: {e}")

    with open(stats_path, 'w') as sj:
        json.dump(dict(final_stats), sj, indent=4)
    print("\nDataset generated successfully.")