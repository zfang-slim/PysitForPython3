## Installation instruction

You may install the Pysit toolbox directly on your system and use the python of
your system. However, we recommend to install the Pysit toolbox on a virtual environment
to avoid possible conflicts. There are several different virtual environments that
you may choose. In this instruction, we provide a way to use the miniconda. Additionally,
although this version of Pysit is for Python 3. The installation of a required
external toolbox -- petsc -- still needs Python 2.  

1. Please go to the website <https://conda.io/miniconda.html> and select the miniconda
   that works for your platform. There are two version `Python 3.7` and `Python 2.7`.
   We recommend to install `Python 3.7`. Therefore, the default Python for a new virtual
   environment is `Python 3.7`. Download the one that best fits you to your local directory,
   for example `~/Download` (In this example, we assume that the installation file of miniconda
   is located at `~/Download`. You can change it to any directory that you want). You will
   find a file named similar as `Miniconda3-latest-MacOSX-x86_64.sh` in your directory.

2. Open your terminal. Go to the directory `~/Download` by the following command:

    `cd ~/Download`

   then, start to install Miniconda by the following command:

   `source ./Miniconda3-latest-MacOSX-x86_64.sh`

   You can check if the miniconda has been installed successfully by the command

   `which conda`

   If it has been installed, then you will the following output

   `/YOURHOMEDIRECTORY/miniconda3/bin/conda`

   Otherwise, you may have some problems with the installation.

3. Create a Python3 virtual environment named with `myenv` with necessary packages
   by the following command

    `conda create -n myenv numpy=1.14.5 scipy=1.1.0 matplotlib=2.2.2 pyamg=3.3.2 cython=0.28.3`


4. Activate your environment by the following command:

    `source activate myenv`

5. Install obspy by pip:

    `pip install obspy`

5. Install necessary external softwares including petsc and mumps. First go to
   your Pysit directory

   `cd /PATHTOPYSIT`

   then, run the following command:

   `source ./install_petsc4py_linux.sh` for Linux platform,

   `source ./install_petsc4py_OSx.sh` for Mac OS platform.

6. Install the Pysit toolbox. First, make sure to go to your Pysit directory by

    `cd /PATHTOPYSIT`

   Then, run the following command:

   `python setup.py install`
