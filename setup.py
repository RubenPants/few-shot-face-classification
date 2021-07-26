"""Setup module for this Python package."""
import pathlib

from setuptools import find_packages, setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

INSTALL_REQUIRES = [
    'facenet-pytorch~=2.5.2',
    'Pillow',
    'matplotlib',
    'scikit-learn',
    'tqdm',
    'numpy',
]

setup(
        name="few_shot_face_classification",
        version="0.0.1",
        description="Library to recognise and classify faces.",
        long_description=README,
        long_description_content_type="text/markdown",
        url="https://github.com/RubenPants/FewShotFaceClassification",
        author="RubenPants",
        author_email="broekxruben@gmail.com",
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        install_requires=INSTALL_REQUIRES,
        include_package_data=True,
)
