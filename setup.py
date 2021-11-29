from setuptools import setup
import versioneer
setup(
    name='PyCQED',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Python Circuit Quantum Electrodynamics Simulation Package',
    license='GNU',
    author='Louis Fry-Bouriaux',
    author_email='lfry512@googlemail.com',
    install_requires=[
        'numpy>=1.17.0',
        'scipy>=1.3.0',
        'qutip>=4.4.1',
        'networkx>=2.3',
        'sympy>=1.4',
        'matplotlib>=3.0.3',
        'SchemDraw>=0.4.0'
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Framework :: Jupyter',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    keywords='quantum circuit simulation, circuit quantum electrodynamics',
    platforms=['Linux','Windows'],
    python_requires='!=2.*, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, >=3.4',
    packages=['pycqed'],
    package_dir={'pycqed':'pycqed/src'}
)


