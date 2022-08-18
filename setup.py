import setuptools
from setuptools.command.develop import develop
from distutils.core import setup
import re
from pathlib import Path

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):

        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            # Mark us as not a pure python package
            self.root_is_pure = False

        def get_tag(self):
            python, abi, plat = _bdist_wheel.get_tag(self)
            if plat == "linux_x86_64":
                plat = "manylinux1_x86_64"
                python = 'cp310'
                abi = 'none'
            elif plat == "win_amd64":
                python = 'cp310'
                abi = 'none'
            elif plat == "linux_x86":
                plat = "manylinux1_x86"
                python = 'cp310'
                abi = 'none'
            elif plat == "win_32":
                python = 'cp310'
                abi = 'none'
            # We don't contain any python source
            return python, abi, plat
except ImportError:
    bdist_wheel = None

version_regex = re.compile(r'^VERSION = [\'\"]?([0-9\.dev]+)[\'\"]?$', re.MULTILINE)
version = version_regex.search(open("./seyfert/__init__.py").read()).groups()[0]

setup(
    name='seyfert',
    cmdclass={
        'bdist_wheel': bdist_wheel,
        'develop': develop
    },
    version=version,
    license='GNU General Public License',
    long_description=open('README.md').read(),
    packages=setuptools.find_packages(),
    scripts=[str(f) for f in Path('bin').iterdir()],
    python_requires='>3.8',
    package_data={
        'seyfert': [
            'data/config_files/*',
            'data/fishers/*',
            'data/input_data/*',
            'data/tables/*',
            'data/test_data/*',
        ]}
)
