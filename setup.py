import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='pynemo',
    version='0.0.1',
    author='Yarone Tokayer',
    author_email='yarone.tokayer@yale.edu',
    description='A collection of functions for issuing commands for N body simulations using NEMO programs',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/yaronetokayer/pynemo',
    project_urls = {
        "Bug Tracker": "https://github.com/yaronetokayer/pynemo/issues"
    },
    license='MIT',
    packages=[
        'pynemo'
    ],
    install_requires=[
        'numpy',
        'astropy',
        'agama'
    ],
)
