"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

def append_the_data_into_csv(filename:str, data) -> None:
    with open(filename, "a+") as fp:
        fp.write(data)