from setuptools import setup, find_packages

setup(
    name="conditional_diffusion_motion",
    version="0.1.0",
    description="A package to simulate diffusion motion with conditional diffusion",
    author="anon",
    author_email="anon",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
)