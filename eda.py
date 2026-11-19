"""Summarize the math-course data and make the EDA plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

QUANT = ['G3', 'absences', 'age']
NUMERIC_COLUMNS = [
    'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
    'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
    'absences', 'G1', 'G2', 'G3',
]
CATEGORIES = {
    'studytime': dict(enumerate(['<2', '2-5', '5-10', '>10'], start=1)),
    'Medu': dict(enumerate(['None', 'Primary', 'Grades 5-9',
                            'Secondary', 'Higher'])),
    'Fedu': dict(enumerate(['None', 'Primary', 'Grades 5-9',
                            'Secondary', 'Higher'])),
    'school': {'GP': 'Gabriel Pereira', 'MS': 'Mousinho da Silveira'},
}
MISSING = ['', '?', 'NA', 'N/A', 'NULL', 'None', 'NaN']


def summarize(data):
    """Return missing counts, seven statistics, and full category counts."""
    statistics = {}
    for column in QUANT:
        values = data[column]
        statistics[column] = {
            'mean': values.mean(),
            'std': values.std(),
            'min': values.min(),
            '25%': values.quantile(0.25),
            '50%': values.median(),
            '75%': values.quantile(0.75),
            'max': values.max(),
        }
    quantitative = pd.DataFrame(statistics).T
    categories = {}
    for column, labels in CATEGORIES.items():
        counts = {}
        for level in labels:
            counts[level] = len(data.loc[data[column] == level])
        categories[column] = counts
    return data.isna().sum(), quantitative, categories


# Challenge Goals: New Library and Result Validity.


def rank_associations(data):
    """Compute marginal Spearman correlations and within-school directions."""
    rows = []
    for sample, records in [('All', data), ('G3 > 0', data[data.G3 > 0])]:
        for school in ['All', *CATEGORIES['school']]:
            group = records if school == 'All' else records[
                records.school == school
            ]
            for predictor in ['studytime', 'absences']:
                pair = group[[predictor, 'G3']].dropna()
                rho = stats.spearmanr(pair[predictor], pair.G3).statistic
                rows.append((sample, school, predictor, len(pair), rho))
    columns = ['sample', 'school', 'predictor', 'n', 'rho']
    return pd.DataFrame(rows, columns=columns)


def rank_dot(x, y, axis=-1):
    """Find the Pearson correlation of ranks, with ties averaged."""
    x = stats.rankdata(x, axis=axis)
    y = stats.rankdata(y, axis=axis)
    x = x - x.mean(axis=axis, keepdims=True)
    y = y - y.mean(axis=axis, keepdims=True)
    denominator = np.sqrt((x*x).sum(axis=axis) * (y*y).sum(axis=axis))
    return np.divide((x*y).sum(axis=axis), denominator,
                     out=np.full_like(denominator, np.nan),
                     where=denominator > 0)


def correlation_inference(data):
    """Run paired bootstrap intervals and pairing permutation tests."""
    rng = np.random.default_rng(163)
    rows = []
    for sample, records in [('All', data), ('G3 > 0', data[data.G3 > 0])]:
        for predictor in ['studytime', 'absences']:
            x, y = records[[predictor, 'G3']].dropna().to_numpy().T
            interval = stats.bootstrap(
                (x, y), rank_dot, paired=True, n_resamples=5000,
                method='percentile', batch=250, rng=rng
            ).confidence_interval
            p_value = np.nan
            if sample == 'All':
                result = stats.permutation_test(
                    (x,), lambda shuffled, axis: rank_dot(shuffled, y, axis),
                    permutation_type='pairings', n_resamples=9999,
                    batch=250, rng=rng
                )
                p_value = result.pvalue
            rows.append((sample, predictor, interval.low, interval.high,
                         p_value))
    columns = ['sample', 'predictor', 'lower', 'upper', 'p_raw']
    results = pd.DataFrame(rows, columns=columns)
    primary = results['sample'].eq('All')
    results.loc[primary, 'p_holm'] = multipletests(
        results.loc[primary, 'p_raw'], method='holm'
    )[1]
    return results


def model_diagnostics(data, output):
    """Check identification, leverage, residual spread and nonlinearity."""
    formula = (
        'G3 ~ C(Medu, Treatment(1)) + C(Fedu, Treatment(1))'
        ' + C(studytime) + C(school) + absences + age'
    )
    for label, sample in [('all', data), ('positive', data[data.G3 > 0])]:
        model = smf.ols(formula, data=sample).fit(cov_type='HC3', use_t=True)
        intervals = model.conf_int()
        intervals.columns = ['lower', 'upper']
        intervals.insert(0, 'estimate', model.params)
        intervals.to_csv(output / f'coefficients_{label}.csv')
        model.wald_test_terms(scalar=True).table.to_csv(
            output / f'joint_tests_{label}.csv'
        )
        if label == 'all':
            primary = model

    x = primary.model.exog
    rank = np.linalg.matrix_rank(x)
    fitted = primary.fittedvalues
    residual = primary.resid
    influence = primary.get_influence()
    leverage = influence.hat_matrix_diag
    cooks = influence.cooks_distance[0]
    complete = data.loc[fitted.index]
    columns = ['fitted', 'residual', 'leverage', 'cooks_distance']
    diagnostics = pd.DataFrame(
        np.column_stack([fitted, residual, leverage, cooks]),
        columns=columns, index=fitted.index
    )
    diagnostics.to_csv(output / 'model_diagnostics.csv', index_label='row')
    spread = diagnostics.groupby(pd.qcut(fitted, 4), observed=True)
    spread.residual.std().to_csv(output / 'residual_spread.csv')

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout='constrained')
    axes[0].scatter(fitted, residual, color='teal')
    axes[0].set(xlabel='Fitted grade', ylabel='Residual',
                title='Residuals vs. fitted grades')
    axes[1].scatter(complete.absences, residual, color='teal')
    axes[1].set(xlabel='Recorded absences', title='Residuals vs. absences')
    fig.savefig(output / 'diagnostics.png', dpi=220)
    plt.close(fig)
    return rank, x.shape[1], float(leverage.max()), float(cooks.max())


def main():
    """Read the CSV, make the EDA outputs, and run the challenge analyses."""
    data = pd.read_csv('student-mat.csv', sep=';', na_values=MISSING)

    for column in data.columns:
        if column not in NUMERIC_COLUMNS:
            text = data[column].str.strip()
            data[column] = text
            for marker in MISSING:
                data.loc[text == marker, column] = float('nan')

    limits = {'G3': (0, 20), 'absences': (0, 93), 'age': (15, 22)}
    for column, (low, high) in limits.items():
        for value in data[column].dropna():
            if value < low or value > high:
                print(f'Invalid range in {column}: {value}')
                return
            if value % 1 != 0:
                print(f'Expected an integer in {column}: {value}')
                return
    for column, labels in CATEGORIES.items():
        for value in data[column].dropna():
            if value not in labels:
                print(f'Invalid category in {column}: {value}')
                return

    missing, quantitative, categories = summarize(data)
    with open('missing.csv', 'w') as file:
        file.write(',missing\n')
        for column, count in missing.items():
            file.write(f'{column},{count}\n')

    with open('quantitative.csv', 'w') as file:
        file.write(',mean,std,min,25%,50%,75%,max\n')
        for variable in QUANT:
            line = variable
            for statistic in quantitative.columns:
                line += f',{quantitative.loc[variable, statistic]}'
            file.write(f'{line}\n')

    with open('categories.csv', 'w') as file:
        file.write('variable,level,count\n')
        for column, counts in categories.items():
            for level, count in counts.items():
                file.write(f'{column},{level},{count}\n')

    with open('study_groups.csv', 'w') as file:
        file.write('studytime,size,mean,median\n')
        for level, group in data.groupby('studytime'):
            grades = group['G3']
            mean = grades.mean()
            median = grades.median()
            file.write(f'{level},{len(group)},{mean},{median}\n')

    with open('zero_counts.csv', 'w') as file:
        file.write('absences,False,True\n')
        for no_absences in [False, True]:
            group = data.loc[(data['absences'] == 0) == no_absences]
            nonzero = len(group.loc[group['G3'] != 0])
            zero = len(group.loc[group['G3'] == 0])
            file.write(f'{no_absences},{nonzero},{zero}\n')

    with open('education_groups.csv', 'w') as file:
        file.write('variable,level,size,mean\n')
        for column in ['Medu', 'Fedu']:
            for level, group in data.groupby(column):
                mean = group['G3'].mean()
                file.write(f'{column},{level},{len(group)},{mean}\n')

    # Plots for the basic EDA.
    plt.figure(figsize=(7.2, 2.65))
    groups = []
    labels = []
    for level, hours in CATEGORIES['studytime'].items():
        grades = data.loc[data['studytime'] == level, 'G3'].dropna()
        groups.append(grades)
        labels.append(f'{hours} h')

    plt.boxplot(groups, tick_labels=labels)
    plt.xlabel('Weekly study time')
    plt.ylabel('Final grade')
    plt.title('Final grades by weekly study time')
    plt.tight_layout()
    plt.savefig('studytime.png')
    plt.close()

    plt.figure(figsize=(7.2, 2.65))
    positive = data.loc[data['G3'] > 0]
    zero = data.loc[data['G3'] == 0]
    plt.scatter(positive['absences'], positive['G3'], color='teal',
                label='Positive grade')
    plt.scatter(zero['absences'], zero['G3'], color='orange',
                label='Zero grade')
    plt.xlabel('Recorded absences')
    plt.ylabel('Final grade')
    plt.title('Final grades and school absences')
    plt.legend()
    plt.tight_layout()
    plt.savefig('absences.png')
    plt.close()

    print('Rows and columns:', data.shape)
    print('Missing cells:', missing.sum())

    # Challenge Goals: write the test and regression results.
    output = Path('results')
    output.mkdir(parents=True, exist_ok=True)
    rank_associations(data).to_csv(
        output / 'rank_associations.csv', index=False
    )
    correlation_inference(data).to_csv(
        output / 'rank_inference.csv', index=False
    )
    rank, columns, leverage, cooks = model_diagnostics(data, output)
    print(f'Design rank: {rank} of {columns}')
    print(f'Maximum leverage: {leverage:.3f}')
    print(f'Maximum Cook distance: {cooks:.3f}')


if __name__ == '__main__':
    main()
