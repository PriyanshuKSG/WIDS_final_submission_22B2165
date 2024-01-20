from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Education',
  'Operating System :: Microsoft :: Windows :: Windows 11',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]
 
setup(
  name='basic_binary_classification',
  version='0.0.1',
  description='A very basic NN',
  long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
  url='',  
  author='Priyanshu K.S. Gangavati ',
  author_email='priyanshupreetamsg@gmail.com',
  license='MIT', 
  classifiers=classifiers,
  keywords='neural_network', 
  packages=find_packages(),
  install_requires=['numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',] 
)