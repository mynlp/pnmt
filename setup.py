#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='Pre-train-OpenNMT-py',
    description='A python onmt extension that supports pre-train model implementation of OpenNMT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='o_2.2.0_p_0.0.1',
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=[
        "torch>=1.6.0",
        "torchtext==0.5.0",
        "configargparse",
        "tensorboard>=2.3",
        "flask",
        "waitress",
        "pyonmttok>=1.23,<2",
        "pyyaml",
        "transformers"
    ],
    entry_points={
        "console_scripts": [
            "pnmt_server=pnmt.bin.server:main",
            "pnmt_train=pnmt.bin.train:main",
            "pnmt_translate=pnmt.bin.translate:main",
            "pnmt_translate_dynamic=pnmt.bin.translate_dynamic:main",
            "pnmt_release_model=pnmt.bin.release_model:main",
            "pnmt_average_models=pnmt.bin.average_models:main",
            "pnmt_build_vocab=pnmt.bin.build_vocab:main"
        ],
    }
)
