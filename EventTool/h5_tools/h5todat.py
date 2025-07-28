"""
@Time ： 2023/3/23 13:17
@Auth ： Haoyue Liu
@File ：h5todat.py
"""
from pandas import HDFStore

store = HDFStore('inputFile.hd5')
store['table1Name'].to_csv('outputFileForTable1.dat')

