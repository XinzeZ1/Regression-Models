import sys
sys.path.append("D:\\workfiles\\450\\Test02\\")

import numpy as np
import pandas as pd
from aaa import car


audi = car("audi a4", "blue")
ferrari = car("ferrari 488", "green")
 
audi.show()     # same output as car.show(audi)
ferrari.show() 