from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='rl_package',
      version="0.0.1",
      description="Reinforcement Learning traffic light",
      author="PPFA",
      install_requires=requirements,
      packages=find_packages(),
      #test_suite="tests")
)
