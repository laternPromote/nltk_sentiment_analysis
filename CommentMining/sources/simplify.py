# -*- coding: UTF-8 -*-
__author__ = 'geyan'


# 使用该函数时，要在txt文件后加一个回车，否则utf8无法解码繁体中文
def tradToSimp(f):
    table = {}
    file = open("utftable.txt")
    line = file.readline()
    while line:
        line = line.strip()
        # line = unicode(line, "utf-8")
        assert len(line) == 3 and line[1] == "="
        table[line[2]] = line[0]
        line = file.readline()
    file.close()

    file = open(f)
    line = file.readline()
    while line:
        line = line[:-1]
        # line = unicode(line, "utf-8")
        output = [table.get(char, char) for char in line]
        print("".join(output).encode("utf-8"))
        line = file.readline()
    file.close()

    print('file has been simplified')
