.. SEYFERT documentation master file, created by
   sphinx-quickstart on Tue Jun 23 09:30:38 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SEYFERT's documentation!
===================================

SEYFERT (SurvEY FishEr foRecast Tool) is a code to perform forecast of cosmological parameters measurement for several cosmological probes. Up to now are included:

    - Weak Lensing
    - Spectroscopic Galaxy Clustering
    - Photometric Galaxy Clustering
    - Void Clustering

The code is written in Python in a modular way. It is actually interfaced with the Boltzmann solver `CAMB <https://camb.readthedocs.io/en/latest/#python-camb>`_, altough other codes can be easily added. The code can be installed executing::

    ./installer.sh

Cosmology modules
=================

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   modules/modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
