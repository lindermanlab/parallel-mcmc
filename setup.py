from setuptools import setup, find_packages

# Define the base requirements with specific versions
base_requirements = [
]

# Create flexible requirements by removing version specifiers
flex_requirements = [req.split("==")[0] for req in base_requirements]

setup(
    name="parallel-mcmc",
    version="0.1",
    description="Algorithms for Parallel Evaluation of MCMC Samplers",
    packages=find_packages(),
    install_requires=base_requirements,  # Use specific versions by default
    extras_require={
        "cr": ["python==3.12.1"],  # Python 3.12.1 for v1.0.0, commit 458ad76
        "flex": flex_requirements,  # Flexible versions without specifiers
    },
)