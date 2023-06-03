import os

pre_name = os.path.basename(__file__)
pre_name = os.path.splitext(pre_name)[0]
print(pre_name)