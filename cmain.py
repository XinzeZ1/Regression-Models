import csv


def getFunc(file, Y_name):
    with open(file, 'r') as f:
        header = next(csv.reader(f))
        if 'ID' in header:
            del header[header.index('ID')]
        try:
            del header[header.index(Y_name)]
        except ValueError:
            raise ValueError(f'The Y column "{Y_name}" is not exist!!!')
        valueDict = {'Y': Y_name}
        for i, column in enumerate(header):
            valueDict[f'X{i + 1}'] = column
        return f'Y~{"+".join([f"X{i}" for i in range(1, len(header) + 1)])}', valueDict


function, valueDict = getFunc('play.csv', 'X1')
print(function)
print(valueDict)
