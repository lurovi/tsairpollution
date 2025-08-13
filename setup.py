from setuptools import setup, find_packages

setup(
    name='tsairpoll',
    version='1.0.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            # If you have any command-line scripts
            # 'command-name = mylibrary.module:function',
        ],
    },
    author= 'Luigi Rovito', 
    author_email='luigirovito2@gmail.com',
    description='TS Air Pollution',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/lurovi/ts-air-pollution',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ]
)
