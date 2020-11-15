ENDC = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"
INDENT = "\t"

class level:
    info = "\033[94m[DEBUG]" + ENDC
    success = BOLD + "\033[92m[+]" + ENDC
    warn = "\033[93m[WARN]" + ENDC
    fail = BOLD + "\033[91m[FAIL]" + ENDC


def debug( text:str, level:str = level.info, indent:int = 0, end:str = "\n"):
    print( INDENT*indent + level + " " + text, end=end )
