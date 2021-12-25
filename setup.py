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
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Shubham Wagh',
    author_email='shubhamwagh48@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "Programming Language :: Python :: 3.10",
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='HOG_python, setuptools, development',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.6, <4',
    install_requires=['scikit-image'],
    data_files=[('data', ['data/car.jpg', 'data/people.jpg']), ('examples', ['examples/Hog_car.png', 'examples/HOG_people.png', 'examples/HOG_implementation.png'])],
    entry_points={  # Optional
        'console_scripts': [
            'HOG_python=HOG_python.__main__:main',
        ],
    },
)
