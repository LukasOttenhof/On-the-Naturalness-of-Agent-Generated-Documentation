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
# --- Configuration ---
REPO_BASE_DIR = Path("./cloned_repos")
OUTPUT_DIR = Path("./dataset/data")
REPO_BASE_DIR.mkdir(exist_ok=True)

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
    """
    Finds a commit within a 'soft boundary' window.
    Example: 1 month out + 14 day window.
    """
    merge_date = datetime.fromisoformat(merged_at_str.replace("Z", "+00:00"))
    results = {}

    for months in [1, 3]:
        # The start of our search (e.g., exactly 1 month later)
        start_search = merge_date + relativedelta(months=months)
        # The end of our 'soft boundary' (e.g., 1 month + 14 days)
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

def strip_comments(text):
    # 1. Remove multi-line comments /* ... */
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    # 2. Split into lines to handle inline comments
    lines = text.splitlines()
    clean_lines = []
    
    for line in lines:
        # Remove // or # style comments
        # This regex looks for // or # and grabs everything until the end of the line
        line = re.sub(r'(//|#).*$', '', line)
        
        # Remove triple quote docstrings if they are on a single line 
        # (e.g., """ doc """)
        line = re.sub(r'(""".*?"""|' + "'''.*?''')", '', line)
        
        # Only keep the line if it isn't empty after stripping whitespace
        if line.strip():
            clean_lines.append(line.rstrip())
            
    return "\n".join(clean_lines)

def tokenize(text): #this tokenizer will split words based on the aplabetical content, 
    if not text: # "def tokenize(text): boom1 bang" -> print(tokenize("def tokenize(text): boom1 bang"))
        return []
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower())


def find_documentation_header(lines, start_line):
    """
    Search upward from start_line to find docstring/documentation blocks.
    Handles Python docstrings (''' or \"\"\"), C-style block comments (/* */),
    and single-line comments (#, //).
    Returns the adjusted start_line that includes the docstring header.
    
    Parameters:
        lines : list        -> List of file lines (0-indexed)
        start_line : int    -> Current start line of the function (1-indexed)
    
    Returns:
        int : The line number where the docstring/documentation begins (1-indexed)
    """
    i = start_line - 2  # Start from line before function definition (0-indexed)
    doc_start = start_line  # Default to original start if no doc found
    in_block_comment = False
    block_delimiter = None
    
    # Search upward for documentation
    while i >= 0:
        line = lines[i].strip()
        
        # Check for Python docstrings (triple quotes)
        if '"""' in line or "'''" in line:
            quote = '"""' if '"""' in line else "'''"
            doc_start = i + 1  # Mark this as start of doc (1-indexed)
            
            # Count quotes to see if it's opening or closing
            count = line.count(quote)
            if count == 1:
                # Single occurrence means we're inside a docstring, keep searching up
                in_block_comment = True
                block_delimiter = quote
            elif count >= 2:
                # Double occurrence means docstring is complete on this line, we're done
                return doc_start
        
        # Check for C-style block comments (/* */)
        elif '/*' in line:
            doc_start = i + 1  # Mark this as start of doc (1-indexed)
            return doc_start
        elif '*/' in line and in_block_comment and block_delimiter == '/*':
            # Continue searching, we found the end of block comment
            pass
        
        # Check for single-line comments (# or //)
        elif line.startswith(('#', '//')) and not in_block_comment:
            doc_start = i + 1  # Mark this as start of doc (1-indexed)
            # Continue searching to find all consecutive comment lines
        
        # Stop if we hit a blank line or non-comment line (but continue if in block)
        elif line == '' and not in_block_comment:
            i -= 1
            continue
        elif not (line.startswith(('#', '//', '*')) or in_block_comment):
            # Hit a non-comment line, stop searching
            break
        
        i -= 1
    
    return doc_start

def extract_documentation(lines, start, end):
    """
    Extracts all documentation (header + internal) from the given range.
    Uses regex to pull comments out of code lines (inline comments).
    """
    # 1. Get the raw block of text for the function range
    # start and end are 1-indexed from lizard/find_documentation_header
    raw_block = "\n".join(lines[start-1:end])
    
    extracted_docs = []

    # 2. Extract Multi-line C-style /* ... */ blocks
    c_blocks = re.findall(r'/\*.*?\*/', raw_block, flags=re.DOTALL)
    extracted_docs.extend([b.strip() for b in c_blocks])
    
    # 3. Process line-by-line for #, //, and Python triple quotes
    for line in raw_block.splitlines():
        line = line.strip()
        
        # Capture Python docstrings (multiline or single line)
        # Regex finds """text""" or '''text'''
        py_doc = re.findall(r'(""".*?"""|' + "'''.*?'''" + r'|"""[\s\S]*?"""|' + "'''[\\s\\S]*?''')", line)
        if py_doc:
            extracted_docs.extend(py_doc)
            continue # Move to next line if this was just a docstring line

        # Capture # or // comments (including inline ones)
        # Look for the symbol and grab everything after it
        comment_match = re.search(r'(//|#)(.*)$', line)
        if comment_match:
            # group(0) includes the // or #
            extracted_docs.append(comment_match.group(0).strip())

    return extracted_docs, lines

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

    # fallback: match by filename
    filename = Path(rel_path).name

    matches = [f for f in files if filename == Path(f).name]

    if len(matches) == 1:
        return matches[0]
 
    return None

def getTurnover(local_repo: Path, rel_path: str, func_name: str, pr_doc_text: str, merge_sha: str, merged_at: str):
    month_shas = get_time_based_shas(local_repo, merged_at, window_days=14)
    number_shas = get_target_shas(local_repo, merge_sha)
    shas = [number_shas[0], number_shas[1], number_shas[2], month_shas['1_month_mark'], month_shas['3_month_mark']]

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
                # print("could not find file in new tree")
                res.append(None)
                continue


            cmd_show = ["git", "-C", str(local_repo), "show", f"{sha}:{path_at_commit}"]
            file_res = subprocess.run(cmd_show, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            
            if file_res.returncode != 0: 
                # print(f"error pulling future commit for {rel_path} using file SHA")
                print(file_res.stderr.strip())
                res.append(None)
                continue
            
            content = file_res.stdout
            lines = content.splitlines()

            with tempfile.NamedTemporaryFile(suffix=Path(rel_path).suffix.lower(), delete=False, mode='w', encoding='utf-8') as tf:
                tf.write(content)
                tmp_path = tf.name

            try:     
                cp = subprocess.run([sys.executable, "-m", "lizard", str(tmp_path)], capture_output=True, text=True, timeout=60)
                # if cp.returncode != 0:
                #     print(f"Lizard execution error for {sha}: {cp.stderr}")
                #     res.append(None)
                #     continue

                metrics = parse_detailed_lizard(cp.stdout)
                
                # IMPROVED MATCHING: Lizard location can be "name@line-line@path" or "path:line name"
                func = None
                for f in metrics.get("functions_info", []): 
                    # Parse location: can be "path/to/file.py:42 func_name" or "func_name@start-end@path"    
                    location = f["location"]
                    if '@' in location and '-' in location:
                        # Handle nested or ranged format: func_name@start-end@path
                        parts = location.split('@')
                        if len(parts) >= 3:
                            e_name = parts[0]
                            if func_name in e_name or e_name in func_name:
                                func = f
                                break

                if func:
                    # Parse start line safely
                    loc_full = func["location"]
                    if '@' in loc_full:
                        # format: name@start-end@path
                        f_start = int(loc_full.split('@')[1].split('-')[0])
                    else:
                        # format: path:line name
                        f_start = int(loc_full.rsplit(' ', 1)[0].split(':')[-1])
                    
                    f_end = f_start + func["length"] - 1
                    
                    adj_start = find_documentation_header(lines, f_start)
                    doc_list, _ = extract_documentation(lines, adj_start, f_end)
                    future_doc_text = " ".join(doc_list)
                    
                    future_tokens = set(tokenize(future_doc_text))
                    if not future_tokens:
                        res.append((sha, 1.0)) # 100% turnover if docs were deleted
                    else:
                        overlap = len(pr_tokens.intersection(future_tokens))
                        turnover = 1.0 - (overlap / len(pr_tokens))
                        res.append((sha, round(turnover, 4)))
                else:
                    res.append(None) # file existed but function was removec
                    # res.append((sha, 1.0)) # file existed but function was removec

            except Exception as e:
                print(f"Lizard processing error: {e}")
                res.append(None)
            finally:
                # Cleanup without printing "failed"
                if os.path.exists(tmp_path): 
                    os.remove(tmp_path)
                    
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
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        local_path = REPO_BASE_DIR / repo_name
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
        #print(url) # https://api.github.com/repos/Draco3310/Gal-Friday2/pulls/41/files
        files = [] # each file has the follwoing "sha": (i belive this is blob id not commit sha)"filename":  "status":  "additions": "deletions": "changes": "blob_url" "raw_url" "contents_url" "patch":
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
        # Ensure Windows handles the UTF-8 rule files correctly
        os.environ["PYTHONUTF8"] = "1"
        
        extension_map = {
            '.py': 'python',
            '.js': 'javascript', '.jsx': 'javascript',
            '.java': 'java',
            '.cs': 'csharp',
            '.go': 'golang',
            # Change 'cpp' to 'c' (Semgrep uses 'c' ruleset for both C and C++)
            '.c': 'c', '.cpp': 'c', '.cc': 'c', '.h': 'c', '.hpp': 'c',
            '.kt': 'kotlin', '.kts': 'kotlin',
            '.php': 'php',
            '.scala': 'scala',
            '.swift': 'swift'
        }
        
        ext = Path(file_path).suffix.lower()
        lang_config = extension_map.get(ext)
        
        # Safely build configs
        configs = ["p/security-audit", "p/default", "p/owasp-top-ten"]
        if lang_config:
            configs.append(f"p/{lang_config}")
        else:
            print(f"Warning: No language-specific rules found for {ext}")

        all_findings = {}

        for config in configs:
            cmd = [
                "semgrep",
                f"--config={config}",
                "--json",
                "--quiet",
                "--no-git-ignore", # Essential for mining arbitrary files
                str(file_path)
            ]

            try:
                # Execute the command
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

                # Parse the JSON output
                if res.stdout.strip():
                    data = json.loads(res.stdout)
                    all_findings[config] = data.get("results", [])
                else:
                    all_findings[config] = []

            except FileNotFoundError:
                print("Semgrep executable not found; please install it.")
                return {}
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
              
            
            # Use 'head' for the contributor's code, or 'merge_commit_sha' for the final result
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

                
                # 'git show' can take a blob SHA directly to get the content
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

                    for func in metrics.get("functions_info", []): 
                        # Parse location: can be "path/to/file.py:42 func_name" or "func_name@start-end@path"
                    
                    
                        location = func["location"]
                            
                        if '@' in location and '-' in location:
                            # Handle nested or ranged format: func_name@start-end@path
                            parts = location.split('@')
                            if len(parts) >= 3:
                                func_name = parts[0]
                                start_end = parts[1]
                                path = '@'.join(parts[2:])
                                if '-' in start_end:
                                    try:
                                        start_line, end_line = map(int, start_end.split('-'))
                                        start_line = find_documentation_header(lines, start_line)
                                    except ValueError:
                                        continue
                                else:
                                    continue
                            else:
                                continue
                        else:
                            # Handle standard format: "path:line func_name"
                            if ' ' not in location:
                                continue
                            file_line, func_name = location.rsplit(' ', 1)
                            if ':' not in file_line:
                                continue
                            _, line_str = file_line.rsplit(':', 1)
                            try:
                                start_line = int(line_str) # get function start line
                                end_line = start_line + func["length"] - 1
                                    
                            except ValueError:
                                continue
                        
                        # if not set(range(start_line, end_line + 1)).intersection(changed_lines): # filter out unchanged functions
                            #continue
                        function_line_range = set(range(start_line, end_line + 1))
                        # Check if every single line in the function is present in the PR's changed lines
                        if not function_line_range.issubset(changed_lines):
                            continue

                        start_line = find_documentation_header(lines, start_line)

                        doc_list, doc_lines = extract_documentation(lines, start_line, end_line)
                        doc_text = " ".join(doc_list)

                        raw_code = "\n".join(lines[start_line-1:end_line])
                        code_text = textwrap.dedent(raw_code)
                        code_text_no_documentation = strip_comments(code_text)
                    

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
                 
                        # Inside process_pr loop:

                        
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
                                "loc": func["length"],
                                "sloc": func["nloc"],
                                "cyclomatic_complexity": func["ccn"],
                                "num_parameters": func["params"],
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
        try:
            if local_repo and os.path.exists(local_repo):
                shutil.rmtree(local_repo)
        except Exception as e:
            print(f"Warning: failed to delete repo {local_repo}: {e}")
        return {"data": results, "stats": stats}

# --- Main Execution ---
if __name__ == "__main__":
    os.environ["PYTHONUTF8"] = "1"
    # df = pd.read_parquet(r"G:\663P\dataset\data\all_pull_request.parquet").head(1000)#683
    df = pd.read_parquet(r"G:\663P\dataset\data\all_pull_request_ballanced.parquet")
    # df = pd.read_parquet(r"G:\663P\dataset\data\human_baseline_2021.parquet").head(1000) # done 1k
  
 
    miner = AiDevMiner()
    
    # output_path = OUTPUT_DIR / "aidev_final_dataset.csv"
    # stats_path = OUTPUT_DIR / "mining_stats_human.json"
    # output_path = OUTPUT_DIR / "dev_final_dataset_subset_b.csv"
    # stats_path = OUTPUT_DIR / "mining_stats_dev_subset_b.json"
    output_path = OUTPUT_DIR / "agent_final_dataset_subset_b.csv"
    stats_path = OUTPUT_DIR / "mining_stats_agent_subset_b.json"
    if stats_path.exists():
        with open(stats_path, 'r') as sj:
            final_stats = Counter(json.load(sj))
    else:
        final_stats = Counter()

    with open(output_path, 'a', encoding='utf-8', newline='') as f:
        writer = None
        file_needs_header = f.tell() == 0
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 1. Map futures with their tokens
            future_to_pr = {}
            for i, (_, row) in enumerate(df.iterrows()):
                token = TOKENS[i % len(TOKENS)]
                future = executor.submit(miner.process_pr, row, token)
                future_to_pr[future] = row.get('number', 'Unknown')

            # 2. Wrap with tqdm for a real-time progress bar
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