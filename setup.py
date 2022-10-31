from setuptools import setup, find_packages

def get_packages():
    mypackages = find_packages(where = "src")
    print(f"found packages {mypackages}")
    return mypackages

setup(
   name='process_webcam_ml',
   version='1.0',
   packages=get_packages(),  #same as name
#    install_requires=['wheel', 'bar', 'greek'], #external packages as dependencies
)