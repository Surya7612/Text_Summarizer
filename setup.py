import setuptools

with open("README.md","r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "Text_Summarizer"
AUTHOR_USER_NAME = "Surya7612"
SRC_REPO = "text-summarizer"
AUTHOR_EMAIL = "nediyadethsurya@gmail.com"

setuptools.setup(
    name=REPO_NAME,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Text summarizer package for NLP applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=[
        "fastapi",
        "uvicorn",
        "transformers[sentencepiece]",
        "datasets",
        "tqdm",
        "pandas",
        "nltk",
        "pydantic",
        "pyyaml",
        "ensure",
        "box"
    ],
    entry_points={
        "console_scripts": [
            "text_summarizer_train=text_summarizer.pipeline.stage_01_data_ingestion:main",
            "text_summarizer_validate=text_summarizer.pipeline.stage_02_data_validation:main",
            "text_summarizer_transform=text_summarizer.pipeline.stage_03_data_transformation:main",
            "text_summarizer_train_model=text_summarizer.pipeline.stage_04_model_trainer:main",
            "text_summarizer_evaluate_model=text_summarizer.pipeline.stage_05_model_evaluation:main"
        ]
    }
)