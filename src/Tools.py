def buildPath(*args: str) -> str:

    path = "/".join(args)
    while("\\" in path or "//" in path):
        path = path.replace("\\","/").replace("//", "/")

    return path

def reducedDateFormat(year: int, month: int, day: int) -> str:

    return "-".join([str(year), str(month), str(day)])

def sortDictionary(dictionary: dict) -> dict:

    return {i: dictionary[i] for i in list(dictionary.keys()).sort()}
