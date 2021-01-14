import argparse
import ast
import importlib
import inspect
import re
import sys
import tokenize
from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path
from shutil import get_terminal_size
from typing import List

if not sys.version_info >= (3, 8):
    # Due to ast.Node.end_lineno
    raise RuntimeError("This script requires Python version 3.8 or higher to run")


class Bcolor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'


levels = {
    'i': 'info',
    'w': 'warning',
    'e': 'error',
    'c': 'critical',
    'x': 'exception',
}


def clear_terminal():
    print("\n" * get_terminal_size().lines, end='')


def confirm_action(desc='Really execute?') -> bool:
    """
    Return True if user confirms with 'Y' input
    """
    inp = ''
    while inp.lower() not in ['y', 'n']:
        inp = input(desc + ' Y/N: ')
    return inp == 'y'


@dataclass
class PrintStatement:
    lineix: int  # Starting line index of original print statement, 0 indexed
    end_lineix: int
    whitespace: str  # The whitespace predecing 'print' in the orginal statement
    arg_line: str  # The new argument string, i.e. `logging.info(arg_line)`
    f_string: bool  # Whether the arg line should be an f_string
    comment: str  # The comment(s) from the original print statement, joined together
    source_context: str  # Function or class name in which the print statement occurs


def modify(
    dir_: Path,
    paths: List[Path],
    default_level: str = 'info',
    accept_all: bool = False,
    context_lines: int = 13,
    comment_sep: str = ' \\ ',
    **_kwargs,
) -> None:
    for path in paths:
        fpath = str(path.relative_to(dir_))
        text = path.read_text()
        lines = text.splitlines()
        edited = False  # Is there an accepted edit to write to disk?

        tree = ast.parse(text)

        # Make backlinks in order to later determine the context (function or class name)
        # of the print statement
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node

        # Find all the nodes in the AST with a print statement,
        print_nodes = [
            node for node in ast.walk(tree)
            if (isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'print')
        ]
        if not print_nodes:
            continue

        # Fetch comments
        lineix_to_comment = dict()  # Line index to comment
        with path.open() as fil:
            for toktype, tok, (start_rowno, _), _, _ in tokenize.generate_tokens(fil.readline):
                if toktype == tokenize.COMMENT:
                    lineix_to_comment[start_rowno - 1] = re.sub(r'#\s*', '', tok)

        print_statements = []
        for node in print_nodes:
            has_vars = False
            terms = []  # (term, is_var)

            for arg in node.args:
                term = ast.get_source_segment(text, arg)
                if isinstance(arg, ast.JoinedStr):
                    # Is an f-string
                    has_vars = True
                    n_quotes = len(re.match(r'^f("""|\'\'\'|"|\')', term).group(1))
                    # Remove 'f' and quotes from string
                    term = term[1 + n_quotes:-n_quotes]
                    # Escape newlines etc
                    term = term.encode('unicode_escape').decode('utf-8')
                    terms.append((term, True))
                elif term.startswith('"') or term.startswith("'"):
                    term = arg.value.encode('unicode_escape').decode('utf-8')
                    terms.append((term, False))
                else:
                    terms.append(('{' + term + '}', True))
                    has_vars = True

            # Escape {} in non-f strings
            if has_vars:
                for i, (term, is_var) in enumerate(terms):
                    if not is_var:
                        terms[i] = (term.replace('{', '{{').replace('}', '}}'), is_var)

            # Use custom separator from print statement
            sep = ' '
            for kw in node.keywords:
                if kw.arg == 'sep':
                    sep = getattr(kw.value, "id",
                                  getattr(kw.value, "value",
                                          ' ')).encode('unicode_escape').decode('utf-8')

            arg_line = sep.join(t[0] for t in terms)
            # Line numbers are 1-indexed
            lineix, end_lineix = node.lineno - 1, node.end_lineno - 1

            # Find the whitespace predecing 'print' in the orginal statement
            old_line = lines[lineix]
            whitespace = re.match(r'(\s*)print', old_line).group(1)

            comments = [
                lineix_to_comment[i]
                for i in range(lineix, end_lineix + 1)
                if i in lineix_to_comment
            ]
            comment = comment_sep.join(comments) or None

            source_context = node
            while not isinstance(
                source_context, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef, ast.Module)
            ):
                source_context = source_context.parent
            if isinstance(source_context, ast.Module):
                source_context = ''
            else:
                source_context = '/' + source_context.name

            print_statements.append(
                PrintStatement(
                    lineix, end_lineix, whitespace, arg_line, has_vars, comment, source_context
                )
            )

        # Find out if logging is already imported, and if so, as what.
        has_logging_imported = False
        import_nodes = []
        logging_asname = 'logging'
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                import_nodes.append(node)
                if node.names[0].name == 'logging':
                    has_logging_imported = True
                    logging_asname = node.names[0].asname or 'logging'
        first_import_node = min(import_nodes, default=None, key=attrgetter('lineno'))
        first_import_lineno = first_import_node.lineno if first_import_node else 1

        def get_line(stmt: PrintStatement, level: str) -> str:
            """
            Construct the full source code line with whitespace,
            logging statement and optionally a comment
            """
            f_string = 'f' if stmt.f_string else ''
            comment = f'  # {stmt.comment}' if stmt.comment else ''
            return stmt.whitespace + f'{logging_asname}.{level}({f_string}"' \
                + stmt.arg_line + '")' + comment

        print_statements.sort(key=lambda st: st.lineix)
        # Multiline print statements are always squashed to a single-line statement,
        # so there might be superfluous lines to delete
        to_del_lineixs = []
        n_changes = 0
        for stmt in print_statements:
            line = get_line(stmt, default_level)

            if not accept_all:
                clear_terminal()

                print(
                    Bcolor.HEADER, f"{fpath}{stmt.source_context}:"
                    f"{stmt.lineix+1}-{stmt.end_lineix+1}", Bcolor.ENDC
                )
                print()
                print_context(lines, stmt.lineix, stmt.end_lineix, line, context_lines)
                print()

                inp = None
                while inp not in ['', 'y', 'n', 'A', 'i', 'w', 'e', 'c', 'x', 'q']:
                    inp = input(
                        Bcolor.OKCYAN + "Accept change? ("
                        f"y = yes ({default_level}) [default], "
                        "n = no, "
                        "A = yes to all, "
                        "i = yes (info), "
                        "w = yes (warning), "
                        "e = yes (error), "
                        "c = yes (critical), "
                        "x = yes (exception), "
                        "q = quit): " + Bcolor.ENDC
                    )
                if inp in ('q', 'Q'):
                    sys.exit(0)
                elif inp in ['i', 'w', 'e', 'c', 'x']:
                    level = levels[inp]
                    line = get_line(stmt, level)
                elif inp == 'A':
                    accept_all = True
            if accept_all or inp in ['', 'y', 'A', 'i', 'w', 'e', 'c', 'e', 'x']:
                lines[stmt.lineix] = line
                edited = True
                to_del_lineixs.extend(range(stmt.lineix + 1, stmt.end_lineix + 1))
                n_changes += 1
        for index in sorted(to_del_lineixs, reverse=True):
            del lines[index]
        if edited:
            # insert import statement, if necessacy:
            if not has_logging_imported:
                lines.insert(first_import_lineno - 1, 'import logging')
                print("Added `import logging`. And ", end=' ')

            path.write_text('\n'.join(lines))
            print(f"Wrote {n_changes} changes to {fpath}")


def print_context(
    lines: List[str], lineix: int, end_lineix: int, new_line: int, context_lines: int = 13
) -> None:
    """
    Print the source code diff, with `context_lines` number of lines before and after
    the modified part
    """
    lines = lines.copy()
    lines.insert(end_lineix + 1, new_line)
    lines = [' ' + ll for ll in lines]
    lines[end_lineix + 1] = Bcolor.OKGREEN + '+' + lines[end_lineix + 1][1:] + Bcolor.ENDC
    for lno in range(lineix, end_lineix + 1):
        lines[lno] = Bcolor.FAIL + '-' + lines[lno][1:] + Bcolor.ENDC
    lines = lines[max(0, lineix - context_lines):end_lineix + context_lines]
    print('\n'.join(lines))


def get_module_files(module: str) -> List[Path]:
    """
    Find all .py files in the directory of an importable module
    """
    mod = importlib.import_module(module)
    moddir = Path(inspect.getsourcefile(mod)).parent
    paths = get_paths(moddir)
    return moddir, paths


def get_paths(dir_: Path) -> List[Path]:
    return [x.resolve() for x in dir_.glob('**/*.py')]


def cli(args_=None):
    parser = argparse.ArgumentParser(
        description='Refactoring tool to replace print with logging',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--file', type=Path)
    group.add_argument(
        '-d',
        '--directory',
        type=Path,
        help='Find all .py paths in the directory and its subdirectories',
    )
    group.add_argument(
        '-m',
        '--module',
        type=str,
        help='Find all .py paths in the directory of the module and its subdirectories',
    )

    parser.add_argument(
        '-l',
        '--level',
        default='info',
        dest='default_level',
        choices=list(levels.values()),
        help="Default logging level",
    )
    parser.add_argument(
        '--accept_all',
        default=False,
        action='store_true',
        help="Auto accept all changes",
    )
    parser.add_argument(
        '--no_single_var_fstrings',
        default=False,
        action='store_true',
        help="Do not convert \"print(x)\" into \"logging.info(f'{x}')\"",
    )
    parser.add_argument(
        '--context_lines',
        default=13,
        type=int,
        help="Number of lines before and after change in diff"
    )
    parser.add_argument(
        '--comment_sep',
        default=' \\ ',
        type=str,
        help="Separator to use when joining multiline comments"
    )

    args = vars(parser.parse_args(args_))

    # Find the source code paths to modify
    if fil := args.get('file'):
        fil = fil.resolve()
        paths = [fil]
        dir_ = fil.parent
    else:
        if dir_ := args.get('directory'):
            dir_ = dir_.resolve()
            paths = get_paths(dir_)
        elif module := args.get('module'):
            dir_, paths = get_module_files(module)

        for p in paths:
            print(str(p.relative_to(dir_)))

        if not confirm_action('Continue with above paths?'):
            sys.exit(0)

    modify(dir_=dir_, paths=paths, **args)


if __name__ == '__main__':
    cli()
