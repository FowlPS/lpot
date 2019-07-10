from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='lpot',
    version='0.0.1',
    description='A simple auto-ml solution based on sklearn',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Przemysław Sadownik',
    author_email='przemek.sadownik@gmail.com',
    packages=['lpot'],
    install_requires=['sklearn', 'numpy'],  # external packages as dependencies
)
