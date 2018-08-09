# location of the Python header files

PYTHON_VERSION = 3.7
# You need to find the correct Python include path that includes all the Python
# header files for the version of Python you want to use.
PYTHON_INCLUDE = /usr/local/Cellar/python/3.7.0/Frameworks/Python.framework/Versions/3.7/include/python3.7m

# The name of the directory for Python config may differ from version to version.
# For Python 2, it is `config` while for Python 3.7, it is config-3.7m-darwin
PYTHON_CONFIG_DIR = /usr/local/Cellar/python/3.7.0/Frameworks/Python.framework/Versions/3.7/lib/python3.7/config-3.7m-darwin

# The Boost-Python lib should be consistant with the version of Python thay you
# want to use. For example, if you want to use Python3.7, then here should be
# boos_python37
BOOST_PYTHON_LIB = boost_python37


# location of the Boost Python include files and library
BOOST_INC = /usr/local/Cellar/boost/1.67.0_1/include/boost
BOOST_LIB = /usr/local/Cellar/boost-python3/1.67.0_1/lib

# compile mesh classes
WRAPPER = constant_density_acoustic_time_scalar_boostpython_wrapper
TARGET = _constant_density_acoustic_time_scalar_boostpython_wrapper

#-$(PYTHON_VERSION)

$(TARGET).so: $(WRAPPER).o
	g++ -shared -Wl $(WRAPPER).o -L$(BOOST_LIB) -l$(BOOST_PYTHON_LIB) -L$(PYTHON_CONFIG_DIR) -lpython$(PYTHON_VERSION) -o $(TARGET).so

$(WRAPPER).o: $(WRAPPER).cpp
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c $(WRAPPER).cpp
