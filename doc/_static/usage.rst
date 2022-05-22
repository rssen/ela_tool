Installation and Usage
=======================

Installation
-------------

.. warning::
   Installation requires Python 3.9 and is only tested on Linux (Debian 10 and Fedora 35)!

- Clone the project.
- Create and activate a python 3 virtualenv.
- Use pip to install dependencies.

.. code-block::

   $ git clone ...
   $ python3 -m venv ./venv
   $ . ./venv/bin/activate
   $ pip install Cython==0.29.23 numpy==1.20.3 setuptools wheel
   $ pip install .


Usage
-----

.. code-block::

   jpeg-cli -h
   usage: jpeg-cli [-h] -f FILE [-o] [-s] [-q QUALITY] [-e] [-m MULTIPLIER]

   JPEG encoder and decoder

   optional arguments:
     -h, --help            show this help message and exit
     -f FILE, --file FILE  path to the image
     -o, --output          write intermediate results to files
     -s, --slow            use slow dct calculation (1:1 application of the formula from the lecture notes)
     -q QUALITY, --quality QUALITY
                           quality value between 0 and 100
     -e, --ela             perform ELA with ELA image as output
     -m MULTIPLIER, --multiplier MULTIPLIER
                           multiplier m > 0
