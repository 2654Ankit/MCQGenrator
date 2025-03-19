# this file is used for local file in venv

from setuptools import find_packages,setup

setup(
    name = 'mcqgenrator',
    version='0.0.1',
    author='ankit',
    author_email = 'ankitkrpandey1904@gmail.com',
    install_requires =['langchain','streamlit','python-dotenv','PyPDF2'],
    packages = find_packages()
)