import os
import re
import sys
from pathlib import Path
from setuptools import find_packages, setup

pwd = os.path.dirname(__file__)
version_file = 'version.py'


def readme():
    with open(os.path.join(pwd, 'README.md'), encoding='utf-8') as f:
        return f.read()


def get_version():
    file_path = os.path.join(pwd, version_file)
    pattern = re.compile(r"\s*__version__\s*=\s*'([0-9A-Za-z.-]+)'")
    with open(file_path, 'r') as f:
        for line in f:
            m = pattern.match(line)
            if m:
                return m.group(1)
    raise RuntimeError(f'No version found in {file_path}')


def parse_requirements(fname='requirements.txt'):
    """只处理纯 Python 的依赖"""
    requirements = []
    if os.path.exists(fname):
        with open(fname, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # 忽略 -e 或 git+ 形式的依赖
                    if line.startswith('-e ') or 'git+' in line:
                        continue
                    requirements.append(line)
    return requirements


setup(
    name='lmdeploy',
    version=get_version(),
    description='A toolset for compressing, deploying and serving LLM',
    long_description=readme(),
    long_description_content_type='text/markdown',
    author='OpenMMLab',
    author_email='openmmlab@gmail.com',
    packages=find_packages(exclude=()),
    include_package_data=True,
    install_requires=parse_requirements('requirements/runtime_cuda.txt'),  # 或者 runtime_cpu.txt
    extras_require={
        'all': parse_requirements('requirements/runtime_cuda.txt'),
        'lite': parse_requirements('requirements/lite.txt'),
        'serve': parse_requirements('requirements/serve.txt'),
    },
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
    ],
    entry_points={'console_scripts': ['lmdeploy = lmdeploy.cli:run']},
)
