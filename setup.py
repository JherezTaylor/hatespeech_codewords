try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('README.md') as f:
    README = f.read()

with open('LICENSE') as f:
    LICENSE = f.read()

config = {
    'description': 'Preprocessing modules for hatespeech detection project',
    'author': 'Jherez Taylor',
    'url': 'https://github.com/JherezTaylor/thesis-preprocessing',
    'download_url': 'https://github.com/JherezTaylor/thesis-preprocessing',
    'author_email': 'jherez.taylor@gmail.com',
    'long_description': README,
    'license': LICENSE,
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['hatespeech_core'],
    'name': 'Hatespeech Preprocessing'
}

setup(**config)
