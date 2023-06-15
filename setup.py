from setuptools import setup, find_packages

setup(
    name='TMDC',
    version='1.0',
    packages=find_packages(),
    authors=["Orestis D. Papagiannopoulos", "Vasileios Pezoulas ", "Costas Papaloukas ", "Dimitrios I. Fotiadis"],
    author_email="orepap@uoi.gr",
    install_requires=["scikit-learn", "Numpy", "matplotlib"],
    description='Timeseries Multi-Dimensional Clustering of Gene Expression Data from Systemic Autoinflammatory Diseases'
)

