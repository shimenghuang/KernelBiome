import setuptools

setuptools.setup(
    name="kernelbiome",
    version="0.1.0",
    author="Shimeng Huang, Elisabeth Ailer",
    author_email="shimeng@math.ku.dk, elisabeth.ailer@helmholtz-muenchen.de",
    description="A kernel-based nonparametric regression and classication framework for compositional data.",
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
    package_dir={"": "."},
    packages=setuptools.find_packages(where="kernelbiome"),
    python_requires=">=3.9",
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'jax'
    ]
)
