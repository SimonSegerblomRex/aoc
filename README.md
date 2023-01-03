Advent of Code
==============
My solutions to [Advent of Code](https://adventofcode.com/).
The purpose of this repistory is mainly just for version control.

Background
----------
I signed up for Advent of Code in 2022, the first year [Axis](https://www.axis.com/)
was an official [sponsor](https://adventofcode.com/2022/sponsors) of the event.
Local time the puzzles are published 6:00 am.

## Retrospective
### 2022
The first week I could finish the puzzles before it was time to bring
the boys to school.
Some days I finished the puzzle on the bus on my way to work, or on the lunch break.
For many of the later puzzles I needed to spend time in the evening as well...
Day 16 was really tough since I had no previous experience with search algorithms,
but I managed to solve it with an ugly brute-force solution.
Then day 19 was even worse and I ended up saving it for Christmas day.
It was the only puzzle I needed to consult the subreddit for to get some hints of
of how to speed up my solution.
I was happy to be able to solve day 18 and day 24 without using search algorithms.
Next year I need to write less buggy code...
In the end I had a lof of fun, but I'm not sure if I'll manage to wake up 6:00 am every
morning next time.

Structure
---------
The solution for each day can be found in a Python script named `YYYY/DD.py`.
(Using this naming convention (with filenames starting with a digit) the
scripts can't be imported as modules, but they are really just intended to
be run as scripts, and any reusable code would be placed in a separate module.)

Each solution contains asserts for my private input data that you'd need to modify.

`template.py` contains the skeleton code used as a base for my solutions.

Libraries used
--------------
Many solutions just rely on Python's standard library, but there're some
third-party libraries that I tend to use as well:
* [advent-of-code-data](https://github.com/wimglenn/advent-of-code-data)
* [numpy](https://github.com/numpy/numpy)
* [scipy](https://github.com/scipy/scipy)
* [sympy](https://github.com/sympy/sympy)
