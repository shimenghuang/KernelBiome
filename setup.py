import setuptools

setuptools.setup(
    name="kernelbiome",
    version="1.0.0",
    author="Shimeng Huang, Niklas Pfister, Elisabeth Ailer, Niklas Pfister",
    author_email="shimeng@math.ku.dk, np@math.ku.dk, elisabeth.ailer@helmholtz-muenchen.de",
    description="A kernel-based nonparametric regression and classification framework for compositional data.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/shimenghuang/KernelBiome",
    project_urls={
        "Bug Tracker": "https://github.com/shimenghuang/KernelBiome/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(include="kernelbiome*"),
    python_requires=">=3.9",
    install_requires=[
        'numpy>=1.22.3',
        'pandas>=1.3.5',
        'scikit-learn>=1.0.1',
        'jaxlib>=0.1.75',
        'jax>=0.2.26',
        'matplotlib>=3.5.1'
    ]
)
