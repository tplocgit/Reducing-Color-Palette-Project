import os


def make_list_input_file(directory: str):
    input_list = []
    with os.scandir(directory) as i:
        for entry in i:
            if entry.is_file():
                input_list.append(directory + '\\' + entry.name)

    return input_list
