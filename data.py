import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup
from requests import get

pd.set_option('display.max_colwidth', None)

__author__ = 'kqureshi'


def fetch_probability(school_id: str) -> pd.DataFrame:
    """

    :param school_id:
    :return:
    """

    url = 'https://nces.ed.gov/collegenavigator/?q=&id={school_id}#admsns'.format(school_id=str(school_id))
    data = get(url)
    soup = BeautifulSoup(data.text, "html.parser")
    df = pd.read_html(str(soup), flavor="bs4")
    for j in df:
        cols = list(j.columns)
        first = cols[0]
        data = j[j[first] == 'Percent admitted']
        if len(data) > 0:
            return j


def fetch_distribution() -> pd.Series:
    """

    :return:
    """
    ids = list(pd.read_csv('school_ids.csv').UNITID.astype(str))
    data_list = {}
    for k in ids:
        try:
            data = fetch_probability(school_id=k)
            data_list[k] = float(data.iloc[1].loc['Total'].replace('%', ''))
        except Exception as e:
            print(e)
            pass
    return pd.Series(data_list)


def plot_distribution(probabilities: pd.Series) -> None:
    """

    :param probabilities:
    :return:
    """
    display(
        probabilities.describe(percentiles=[j / 10 for j in range(10)]).to_frame().rename(columns={0: 'p'}).T.astype(
            int))
    fig, ax = plt.subplots()
    ax.hist(probabilities, color='blue', alpha=0.5, bins=30)
    ax.axvspan(0, 25, alpha=0.1, color='red', label='Reach')
    ax.axvspan(25, 75, alpha=0.1, color='orange', label='Target')
    ax.axvspan(75, 100, alpha=0.1, color='green', label='Safety')
    plt.title('Admission Probability by Category')
    plt.legend()
    plt.ylabel('Count')
    plt.xlabel('Probability')
    plt.show()
