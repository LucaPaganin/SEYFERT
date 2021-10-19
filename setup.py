import setuptools
from distutils.core import setup
import re
from pathlib import Path

version_regex = re.compile(r'^VERSION = [\'\"]?([0-9\.dev]+)[\'\"]?$', re.MULTILINE)
version = version_regex.search(open("./seyfert/__init__.py").read()).groups()[0]

setup(
    name='seyfert',
    version=version,
    license='GNU General Public License',
    long_description=open('README.md').read(),
    packages=setuptools.find_packages(),
    scripts=[str(f) for f in Path('bin').iterdir()],
    python_requires='>3.5',
    package_data={
        'seyfert': [
            'data/config_files/*',
            'data/fishers/*',
            'data/input_data/*',
            'data/tables/*',
            'data/test_data/*',
        ]}
)
