# 一つの特徴だけStandardizationさせる

from IPython import embed
import numpy as np

def scale_one(x):
    # 標準偏差が 1 で平均が 0 の母集団を作る
    new = x - np.mean(x)
    return new / np.std(new)

scaled = scale_one(np.array([1,2,0]))
print(scaled) # [ 0.          1.22474487 -1.22474487]
assert(np.mean(scaled) == 0)
assert(str(np.std(scaled)) == '1.0') # floatの状態だとイコールにならないのでstringに直す
