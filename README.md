## print_to_logging

[![asciicast](https://asciinema.org/a/mSw4uEVJrvxoKW6so3b5RPjAj.png](https://asciinema.org/a/mSw4uEVJrvxoKW6so3b5RPjAj)

This is an interactive tool for refactoring Python code.
The mission is to replace `print` statements with calls to `logging` instead.

`print_to_logging` features:
- it will ask you to accept or modify any proposed change
- support for multiline strings and multiline print statements
- preservation of comments
- automatic f-string generation
- preservation of custom print separators
- interactive choice of logging level (info, warning, error, etc.)
- showing a diff for before and after change

To try it out, clone the repo:

`git clone https://github.com/tsoernes/print_to_logging.git & cd print_to_logging`

Then run the program on the test file:

`python print_to_logging.py -f test.py`


or on a directory of python files:

`python print_to_logging.py -d ~/code/my-python-project`


See `python print_to_logging.py --help`:
``` python
usage: print_to_logging.py [-h] (-f FILE | -d DIRECTORY | -m MODULE) [-l {info,warning,error,critical,exception}] [--accept_all] [--context_lines CONTEXT_LINES]

Refactoring tool to replace print with logging

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE
  -d DIRECTORY, --directory DIRECTORY
                        Find all .py paths in the directory and its subdirectories
  -m MODULE, --module MODULE
                        Find all .py paths in the directory of the module and its subdirectories
  -l {info,warning,error,critical,exception}, --level {info,warning,error,critical,exception}
                        Default logging level (info).
  --accept_all          Auto accept all changes.
  --context_lines CONTEXT_LINES
                        Number of lines before and after change in diff
```

Requires Python version 3.8 or greater to run.
