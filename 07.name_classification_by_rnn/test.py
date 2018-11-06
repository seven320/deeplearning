#encodign utf-8

from data import *
# import copy
from model import *

input = torch.randn(1,n_letters)

output,(h,c) = lstm(input,(hidden ,cell))


# # print(all_categories)
# copy_c = copy.deepcopy(category_lines)
# print(len(copy_c))
# # del copy_c["Japanese"]
#
#
# print(len(copy_c["Japanese"]))
