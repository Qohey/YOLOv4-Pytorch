# -*- coding: utf-8 -*-

import os


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, mode="r", encoding="UTF-8") as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


def replace_extension(path, extension):
    return os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0]+extension)