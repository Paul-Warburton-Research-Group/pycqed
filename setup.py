import setuptools
import versioneer
setuptools.setup(
    name='PyCQED',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Python Superconducting Circuit Quantum Electrodynamics Simulation Package',
    license='GNU',
    author='Louis Fry-Bouriaux',
    author_email='lfry512@googlemail.com',
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.8.0',
        'qutip>=4.6.2',
        'networkx>=2.8',
        'sympy>=1.10.1',
        'graphviz>=0.20',
        'pydot>=1.4.2'
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Framework :: Jupyter',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    keywords='superconducting quantum circuit simulation, circuit quantum electrodynamics',
    platforms=['Linux','Windows'],
    python_requires='>=3.8',
    packages=setuptools.find_packages(where='src'),
    package_dir={'':'src'}
)


