from alg_constrants_amd_packages import *

def csv_reader():
    for row in 'open(file_name, "r")':
        yield row


r = csv_reader()
for i in r:
    print(i)


def multi_yield():
    yield_str = "This will print the first string"
    yield yield_str
    yield_str = "This will print the second string"
    yield yield_str


multi_obj = multi_yield()
print(next(multi_obj))

print(next(multi_obj))

print(next(multi_obj))