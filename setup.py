from setuptools import setup

setup(name='mlp',
      packages=['mlp'],
      version='0.0.1dev1',
      entry_points={
            'console_scripts': ['mlp-cli=mlp.cmd:main']
      }
)
    