.. src documentation master file, created by
   sphinx-quickstart on Sun May 22 12:58:34 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ELA Tool Documentation
=======================================

What is ELA?
------------
Using Error Level Analysis (ELA) it is possible to detect some forms of image manipulation.

.. |jet| image:: ../Testbilder/jet.jpg
   :align: middle

..  |jet_ela| image:: ../Testbilder/jet_ela.png
   :align: middle

+-----------+-----------+
| |jet|     | |jet_ela| |
+-----------+-----------+


How can I use this tool?
------------------------
This program was developed to run on Linux with Python 3.9.
If you *really* want to use it, please check out the usage documentation.
No support will be provided.

.. include::
   _static/usage.rst

Why this project?
---------------------
This project is the result of a university assignment.
It thus had to fulfill certain requirements and features that would not be included
in a "normal" tool.
**I chose to publish it, as I needed a project to play around with for testing Sphinx
documentation** :-).


Project Structure
==================

.. autosummary::
   :toctree: _autosummary
   :template: custom_module.rst
   :recursive:

   src
   bin


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
