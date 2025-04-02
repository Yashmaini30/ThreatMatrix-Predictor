"""
This Setup.py file is essential part of packaging and
distributing python projects. It is used to setupTools
(or disutils in older versions of python) to define the
configuration of the project.
"""

from setuptools import setup, find_packages
from typing import List

def get_requirements()->List[str]:
    """
    This function will return the list of requirements
    """
    requirement_list:List[str]=[]
    try:
        with open("requirements.txt", "r") as files:
            lines=files.readlines()
            for line in lines:
                requirement=line.strip()
                if requirement and requirement!="-e .":
                    requirement_list.append(requirement)

    except FileNotFoundError:
        print("requirements.txt not found")
    return requirement_list


setup(
    name="Network Security",
    version="0.0.1",
    author="Yash Maini",
    author_email="mainiyash2@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)

    