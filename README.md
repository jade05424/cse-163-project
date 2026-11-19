# Student mathematics performance EDA

This CSE 163 Part 2 project uses the mathematics-course file from the
[UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance).

Run from the project directory with Python 3.12.

```sh
python -m pip install -r requirements.txt
python eda.py
python -m unittest -v test_eda
python -m flake8 eda.py test_eda.py
```

The script writes the basic EDA tables and plots to the project directory.
Results for New Library and Result Validity go into `results/`.
`test_students.csv` contains a small dataset for checking the summaries by hand.
