try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'pycal',
    'author': 'Nicholas Tarabelloni',
    'url': 'https://github.com/ntarabelloni/pycal',
    'download_url': 'https://github.com/ntarabelloni/pycal',
    'author_email': 'nicholas.tarabelloni@gmail.com',
    'version': '0.1',
    'install_requires': [''],
    'packages': ['pycal'],
    'scripts': [],
    'name': 'pycal'
}

setup(**config )
