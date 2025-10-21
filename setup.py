import setuptools

with open("README.md","r") as fh:
	long_description=fh.read()
    
setuptools.setup(
    name="omnidiff",
    version="0.2.1",
    author="monkeyluffy824",
    author_email="panangipallisaicharan@gmail.com",
    description="A scalar-valued autograd engine, consider this like an extension of micrograd with support of more complex functions",
    long_description=long_description,
    url="https://github.com/monkeyluffy824/omnidiff",
    packages=setuptools.find_packages(),
    license='MIT',
    license_files=('LICENSE.txt',),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)