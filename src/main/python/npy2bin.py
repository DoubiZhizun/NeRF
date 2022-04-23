import numpy as np
import sys
import struct

if __name__ == '__main__':
    poses = list(np.load(sys.argv[1]))
    f = open(sys.argv[1][:-3] + 'bin', 'wb')
    for p in poses:
        for p2 in list(p):
            p3 = struct.pack('d', p2)
            f.write(p3)
    f.close()
