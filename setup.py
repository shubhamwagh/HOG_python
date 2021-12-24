from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='HOG_python',
    version='1.0.0',
    description='Histogram of Gradients in Python from scratch',
    author='Shubham Wagh',
    author_email='shubhamwagh48@gmail.com',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.6, <4',
    data_files=[('data', ['data/car.jpg', 'data/people.jpg']), ('examples', ['examples/Hog_car.png', 'examples/HOG_people.png', 'examples/HOG_implementation.png'])]
)