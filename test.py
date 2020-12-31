"""
This is a test file to run on.

print_to_logging will automatically insert `import logging` if necessary,
and it will do so, after this module docstring
"""
import re

print("This should be a simple one. Notice the file name and line number up there ^")
print("and your options to choose logging.info, logging.error etc at the bottom.")

x = 1
print(x, "f-strings are automatically created, if needed")

y = 1
supports = 'supports'
print(x, f" it also {supports} custom separators ", y, sep='\n')

print(
    '(although the `end`, `file`, and `flush` arguments to print are ignored.)',
    end='xxx',
    file=None,
    flush=True,
)

print('It keeps')  # comments intact.

print("Multiline "
      'strings '
      """are supported""")  # yapf: disable

statements = "statements"
print("Multiline",
      f"{statements}",
      2)  # yapf: disable

print(
    "Even multiline comments",  # comment A
    "are kept, but joined into a single comment",  # comment B
)  # Comment C
