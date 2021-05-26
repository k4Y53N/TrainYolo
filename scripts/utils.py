def printdic(dic: dict, space=0):
    for key in dic.keys():
        print(f'{key}'.upper(), end=':')
        if type(dic[key]) == dict:
            print('{')
            print('\t' * (space + 1), end='')
            printdic(dic[key], space + 1)
            print('\r', end='')
            print('\t' * space, end='')
            print('}')
        else:
            print(dic[key])
        print('\t' * space, end='')
