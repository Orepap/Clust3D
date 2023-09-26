from setuptools import setup, find_packages

setup(
    name='TMDC',
    version='1.0',
    packages=find_packages(),
    authors=["Orestis D. Papagiannopoulos", "Vasileios Pezoulas ", "Costas Papaloukas ", "Dimitrios I. Fotiadis"],
    author_email="orepap@uoi.gr",
    install_requires=["scikit-learn >= 1.0.2",
                      "numpy >= 1.21.6",
                      "pandas >= 1.4.0",
                      "matplotlib >= 3.5.1"],
    description='Timeseries Multi-Dimensional Clustering of Gene Expression Data from Systemic Autoinflammatory Diseases'
)

