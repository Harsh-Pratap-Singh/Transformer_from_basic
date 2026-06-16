from setuptools import setup, find_packages

setup(
    name="english_hindi_transformer",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "pandas",
        "numpy",
        "pyyaml",
        "sacrebleu",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "train-transformer=scripts.train:main",
            "translate=scripts.translate:main",
        ],
    },
    author="Harsh Pratap Singh",
    description="English to Hindi Transformer using character‑level tokenization",
    license="MIT",
)